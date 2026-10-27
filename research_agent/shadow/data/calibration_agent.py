"""One bounded Calibration Agent attempt over point-in-time read-only tools."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from shadow.data.calibration_models import CalibrationOutput
from shadow.data.queries import CalibrationTools

CALIBRATION_MODEL = "deepseek/deepseek-v4-flash-0731"
CALIBRATION_TEMPERATURE = 0.2   # the protocol fixes every LLM at 0.2; 0.0 made DeepSeek loop upstream
CALIBRATION_MAX_TOKENS = 32768   # same cap as TRADINGAGENTS_MAX_TOKENS in .env
MAX_TOOL_STEPS = 12
RATE_LIMIT_RETRIES = 3
API_ERROR_CODES = (401, 402, 403, 408, 429)   # key, credit, access, timeout, rate limit

RISK_LEVELS = """Risk level is the overall risk appetite for the next four-week cycle:
1 Capital preservation: be highly selective; favor strong evidence and contained downside.
2 Conservative: favor clear evidence and manageable downside; reduce/reject unresolved risks.
3 Balanced: balance risk control and benchmark outperformance; this is the initial level.
4 Opportunistic: accept moderate uncertainty when the risk-adjusted opportunity compensates.
5 High risk tolerance: accept the most compensated uncertainty/volatility, while obeying every
  portfolio, position, drawdown and evidence control. Level is not a position-size multiplier.
The level range is 1-5 and may move by at most one from the previous level per cycle.
"""

SYSTEM_PROMPT = """You are the Calibration Agent for a simulated trading risk team.
Investigate the evidence using only the five read-only tools provided. The cycle date and
experiment version are fixed by the caller. Begin with cycle_summary and past_memos, then inspect
trades and subgroups that help test explanations. Consider supporting and contradicting cases.
New subgroup patterns are hypotheses, not established facts; check prior hypotheses only against
later resolved outcomes. Statistics are evidence, not automatic advice. Thin samples are warnings.

Return one JSON object matching the supplied schema. Give advice for all five categories.
Every citation ID must have appeared in a tool result. Put each numeric fact in numeric_claims,
including the tool call sequence and JSON pointer to the exact returned number. Tool messages wrap
raw tool output as {"tool_call_seq": N, "result": RAW_JSON}; JSON pointers must be relative to
RAW_JSON and must omit the wrapper prefix. For example, use /by_category/event_risk/all_history/n_scored,
not /result/by_category/event_risk/all_history/n_scored. Apply the same raw-result rule to hypothesis
evidence pointers. Do not write decimals or percentages in free-text fields; write them as
numeric_claims. Whole numbers such as a risk level or a day count may appear in text. If
evidence is weak, say so. Include a conflict_rationale when departing from prior advice despite
P(success rate > 50%) >= 0.85. Choose one overall risk level and explain it in risk_level_rationale.
If is_first_cycle is true, risk_level MUST be exactly 3 regardless of evidence. Do not invent numbers,
citations, trades, or news, and do not ask to modify the database.

""" + RISK_LEVELS


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CycleArgs(_Args):
    window_days: int = Field(default=28, ge=1, le=365)


class SearchArgs(_Args):
    filters: dict | None = None
    limit: int = Field(default=20, ge=1, le=100)
    order: Literal["recent", "worst", "best"] = "recent"


class ReadArgs(_Args):
    decision_id: str = Field(min_length=1, max_length=160)
    max_chars: int = Field(default=1500, ge=100, le=5000)


class SubgroupArgs(_Args):
    filters: dict | None = None
    split_by: str | None = None


class PastMemosArgs(_Args):
    limit: int = Field(default=3, ge=1, le=12)


@dataclass
class AttemptResult:
    output: CalibrationOutput | None
    tool_calls: list[dict]
    error: str | None = None
    status: str = "ok"


@dataclass
class ToolRuntime:
    """Model-visible tool specs and their bound, read-only implementation."""

    tools: CalibrationTools
    calls: list[dict] = field(default_factory=list)
    specs: list[StructuredTool] = field(init=False)

    def __post_init__(self):
        self.specs = [
            StructuredTool.from_function(name="cycle_summary", description="Get this-cycle and historical statistics, evidence coverage, memo and portfolio summary.", args_schema=CycleArgs, func=self._cycle_summary),
            StructuredTool.from_function(name="search_trades", description="Search resolved trades using the supported category/range/text filters.", args_schema=SearchArgs, func=self._search_trades),
            StructuredTool.from_function(name="read_trade", description="Read one resolved trade, its proposal, reports, conditions, sources and news seen.", args_schema=ReadArgs, func=self._read_trade),
            StructuredTool.from_function(name="subgroup_stats", description="Compute statistics for a supported subgroup and optional split.", args_schema=SubgroupArgs, func=self._subgroup_stats),
            StructuredTool.from_function(name="past_memos", description="Read prior memos and hypothesis status as of the bound cycle date.", args_schema=PastMemosArgs, func=self._past_memos),
        ]

    def _cycle_summary(self, window_days: int = 28):
        return self.tools.cycle_summary(window_days=window_days)

    def _search_trades(self, filters: dict | None = None, limit: int = 20, order: str = "recent"):
        return self.tools.search_trades(filters=filters, limit=limit, order=order)

    def _read_trade(self, decision_id: str, max_chars: int = 1500):
        return self.tools.read_trade(decision_id=decision_id, max_chars=max_chars)

    def _subgroup_stats(self, filters: dict | None = None, split_by: str | None = None):
        return self.tools.subgroup_stats(filters=filters, split_by=split_by)

    def _past_memos(self, limit: int = 3):
        return self.tools.past_memos(limit=limit)

    def execute(self, name: str, args: dict) -> tuple[int, dict]:
        spec = next((t for t in self.specs if t.name == name), None)
        seq = len(self.calls) + 1
        if spec is None:
            result = {"error": f"unknown tool {name!r}"}
            self.calls.append({"seq": seq, "tool": name, "args": args, "result": result,
                               "called_at": datetime.now(timezone.utc).isoformat()})
            return seq, result
        try:
            result = spec.invoke(args)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"text": result}
        except Exception as exc:  # tool errors are returned to the agent and kept in the trace
            result = {"error": type(exc).__name__, "message": str(exc)[:1000]}
        self.calls.append({"seq": seq, "tool": name, "args": args, "result": result,
                           "called_at": datetime.now(timezone.utc).isoformat()})
        return seq, result


def _retryable_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    return status == 429 or "429" in str(exc)


def _invoke(chain, messages):
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return chain.invoke(messages)
        except Exception as exc:
            if not _retryable_rate_limit(exc) or attempt >= RATE_LIMIT_RETRIES:
                raise
            time.sleep(min(2 ** attempt, 8))


def _parse_json(content) -> CalibrationOutput:
    if isinstance(content, list):
        content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
            for pos, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(candidate[pos:])
                    break
                except json.JSONDecodeError:
                    continue
        if isinstance(payload, dict):
            return CalibrationOutput.model_validate(payload)
    raise ValueError("model response did not contain a valid JSON object")


def run_attempt(llm, tools: CalibrationTools, *, as_of: str, experiment_version: str,
                previous_risk_level: int | None, attempt_no: int) -> AttemptResult:
    runtime = ToolRuntime(tools)
    bound = llm.bind_tools(runtime.specs)
    user_payload = {
        "as_of": as_of,
        "experiment_version": experiment_version,
        "attempt": attempt_no,
        "is_first_cycle": previous_risk_level is None,
        "previous_risk_level": previous_risk_level,
        "output_schema": CalibrationOutput.model_json_schema(),
    }
    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=False))]
    try:
        for _ in range(MAX_TOOL_STEPS):
            response: AIMessage = _invoke(bound, messages)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                output = _parse_json(response.content)
                return AttemptResult(output=output, tool_calls=runtime.calls)
            messages.append(response)
            for call in calls:
                seq, result = runtime.execute(call.get("name", ""), call.get("args") or {})
                messages.append(ToolMessage(
                    content=json.dumps({"tool_call_seq": seq, "result": result}, ensure_ascii=False, default=str),
                    tool_call_id=call.get("id", f"tool_{seq}"),
                ))
        return AttemptResult(output=None, tool_calls=runtime.calls,
                             error=f"exceeded {MAX_TOOL_STEPS} model turns", status="step_limit")
    except Exception as exc:
        code = getattr(exc, "status_code", None) or 0
        # No credit, bad key, rate limit, server or connection trouble: not a model attempt, so the cycle is redone.
        api = isinstance(exc, openai.APIConnectionError) or code in API_ERROR_CODES or code >= 500
        return AttemptResult(output=None, tool_calls=runtime.calls,
                             error=f"{type(exc).__name__}: {str(exc)[:2000]}",
                             status="api_error" if api else "failed")


def build_calibration_llm():
    """Build the pinned OpenRouter model without making a request."""
    from shadow.data.paths import load_env
    from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client

    load_env()
    config = {
        "llm_provider": "openrouter",
        "deep_think_llm": CALIBRATION_MODEL,
        "quick_think_llm": CALIBRATION_MODEL,
        "temperature": CALIBRATION_TEMPERATURE,
        "max_tokens": CALIBRATION_MAX_TOKENS,
        "llm_max_retries": 0,  # bounded 429 retry is handled above
    }
    client = create_llm_client("openrouter", CALIBRATION_MODEL,
                               **build_llm_kwargs(config))
    return client.get_llm()

