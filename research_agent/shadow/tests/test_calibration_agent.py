import json

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from shadow.data.calibration_agent import _parse_json, run_attempt
from shadow.data.calibration_models import CATEGORIES


class FakeTools:
    def cycle_summary(self, window_days=28):
        return {"as_of": "2026-09-28", "experiment_version": "v", "window_days": window_days,
                "by_category": {c: {"all_history": {"n_scored": 0,
                                                          "beta": {"p_rate_above_half": 0.5}}}
                                for c in CATEGORIES}}

    def past_memos(self, limit=3):
        return {"memos": [], "hypotheses": []}

    def search_trades(self, **kwargs):
        return {"trades": []}

    def read_trade(self, **kwargs):
        return {}

    def subgroup_stats(self, **kwargs):
        return {"overall": {"n": 0}}


class FakeLLM:
    def __init__(self):
        self.turn = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.turn += 1
        if self.turn == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "cycle_summary", "args": {"window_days": 28}, "id": "call_1", "type": "tool_call"
            }])
        if self.turn == 2:
            return AIMessage(content="", tool_calls=[{
                "name": "past_memos", "args": {"limit": 3}, "id": "call_2", "type": "tool_call"
            }])
        data = {
            "summary": "Evidence is thin across categories.",
            "risk_level": 3,
            "risk_level_rationale": "Keep the balanced stance while evidence is thin.",
            "advice": {c: {"advice": "keep", "confidence": "low",
                           "reasoning": "There is not enough resolved evidence to change course."}
                       for c in CATEGORIES},
            "numeric_claims": [], "hypotheses": [], "hypothesis_checks": [],
        }
        return AIMessage(content=json.dumps(data))


def test_attempt_calls_read_only_tools_then_returns_structured_output():
    result = run_attempt(FakeLLM(), FakeTools(), as_of="2026-09-28",
                         experiment_version="v", previous_risk_level=3, attempt_no=1)
    assert result.status == "ok"
    assert result.output is not None
    assert set(result.output.advice) == set(CATEGORIES)
    assert [call["tool"] for call in result.tool_calls] == ["cycle_summary", "past_memos"]
    assert [call["seq"] for call in result.tool_calls] == [1, 2]


_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


class FailingLLM:
    def __init__(self, exc):
        self.exc = exc

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        raise self.exc


@pytest.mark.parametrize("exc, status", [
    (openai.APIStatusError("Insufficient credits", response=httpx.Response(402, request=_REQUEST), body=None),
     "api_error"),
    (openai.APIConnectionError(request=_REQUEST), "api_error"),
    (openai.InternalServerError("upstream down", response=httpx.Response(502, request=_REQUEST), body=None),
     "api_error"),
    (openai.BadRequestError("prompt too long", response=httpx.Response(400, request=_REQUEST), body=None),
     "failed"),                                                     # a rerun would fail the same way
    (ValueError("model response did not contain a valid JSON object"), "failed"),
])
def test_api_trouble_is_told_apart_from_a_failed_model_attempt(exc, status):
    result = run_attempt(FailingLLM(exc), FakeTools(), as_of="2026-09-28",
                         experiment_version="v", previous_risk_level=3, attempt_no=1)
    assert (result.output, result.status) == (None, status)


def test_parse_json_extracts_fenced_object_with_surrounding_text():
    data = {
        "summary": "Evidence is mixed.",
        "risk_level": 3,
        "risk_level_rationale": "Keep the balanced stance while evidence is mixed.",
        "advice": {c: {"advice": "keep", "confidence": "low", "reasoning": "Limited evidence."}
                   for c in CATEGORIES},
        "numeric_claims": [], "hypotheses": [], "hypothesis_checks": [],
    }
    parsed = _parse_json("Here is the memo:\n```json\n" + json.dumps(data) + "\n```\nDone.")
    assert parsed.risk_level == 3
    assert set(parsed.advice) == set(CATEGORIES)
