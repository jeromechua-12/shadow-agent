"""Deterministic checks for Calibration Agent outputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date

from shadow.data.calibration_models import CATEGORIES, CalibrationOutput

POSTERIOR_REVIEW_THRESHOLD = 0.85
MAX_RISK_LEVEL = 5
MIN_RISK_LEVEL = 1
DEFAULT_RISK_LEVEL = 3
MAX_DRAWDOWN_FOR_MORE_RISK = -0.10
RECENT_SNAPSHOT_DAYS = 7
TOOLS = ("cycle_summary", "search_trades", "read_trade", "subgroup_stats", "past_memos")
# Statistics in prose (decimals, percentages) must be numeric_claims; whole numbers like "level 3" may stay.
UNSTRUCTURED_NUMBER = re.compile(
    r"(?<![A-Za-z_\d.])(?:\d+(?:\.\d+)?\s?(?:%|percent\b)|\d+\.\d+(?![A-Za-z_\d]))", re.IGNORECASE)


@dataclass
class SafetyResult:
    passed: bool
    blockers: list[dict] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)
    accepted_risk_level: int | None = None
    evidence_score: tuple[int, int, int] = (0, 0, 0)


def json_pointer_value(value, pointer: str):
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with '/'")
    current = value
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(pointer)
    return current


def _collect_ids(value, key: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == "id" or child_key.endswith("_id"):
                if isinstance(child, str):
                    found.add(child)
                elif isinstance(child, list):
                    found.update(x for x in child if isinstance(x, str))
            found.update(_collect_ids(child, child_key))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_ids(child, key))
    return found


def _prior_advice(tool_calls: list[dict]) -> dict[str, str]:
    past = next((c.get("result") for c in tool_calls if c.get("tool") == "past_memos"), None)
    if not isinstance(past, dict):
        return {}
    memos = past.get("memos") or []
    if not memos:
        return {}
    latest = max(memos, key=lambda m: str(m.get("cycle_date") or ""))
    return {str(x.get("category")): str(x.get("advice")) for x in latest.get("advice") or []}


def _category_stats(tool_calls: list[dict]) -> dict:
    summary = next((c.get("result") for c in tool_calls if c.get("tool") == "cycle_summary"), None)
    if not isinstance(summary, dict):
        return {}
    by_category = summary.get("by_category") or {}
    out = {}
    for category, detail in by_category.items():
        history = (detail or {}).get("all_history") or {}
        beta = history.get("beta") or {}
        probability = beta.get("p_rate_above_half")
        if isinstance(probability, (int, float)) and math.isfinite(probability):
            out[category] = {"p_rate_above_half": float(probability),
                             "n_scored": int(history.get("n_scored") or 0)}
    return out


def validate_output(
    output: CalibrationOutput,
    tool_calls: list[dict],
    *,
    previous_risk_level: int | None,
    decision_drawdown: float | None,
    snapshot_date: date | None,
    as_of: date,
) -> SafetyResult:
    """Check claims/limits; hold an increase when its required snapshot is stale."""
    blockers: list[dict] = []
    flags: list[dict] = []
    level = output.risk_level

    if not MIN_RISK_LEVEL <= level <= MAX_RISK_LEVEL:
        blockers.append({"code": "risk_level_range", "message": "risk level must be 1-5"})
    if previous_risk_level is None and level != DEFAULT_RISK_LEVEL:
        blockers.append({"code": "first_level", "message": "first risk level must be 3"})
    if previous_risk_level is not None and abs(level - previous_risk_level) > 1:
        blockers.append({"code": "risk_level_step", "message": "risk level may change by one step per cycle"})
    if previous_risk_level is not None and level > previous_risk_level:
        if decision_drawdown is not None and decision_drawdown < MAX_DRAWDOWN_FOR_MORE_RISK:
            blockers.append({"code": "drawdown_limit", "message": "cannot increase risk after drawdown beyond 10%"})
        elif snapshot_date is None or (as_of - snapshot_date).days > RECENT_SNAPSHOT_DAYS:
            flags.append({"code": "risk_increase_unverified",
                          "message": "no recent decision-book snapshot; proposed increase held at prior level",
                          "proposed_risk_level": level, "accepted_risk_level": previous_risk_level})
            level = previous_risk_level

    call_by_seq = {int(call["seq"]): call for call in tool_calls if "seq" in call}
    for call in tool_calls:
        if call.get("tool") not in TOOLS:
            blockers.append({"code": "unknown_tool", "tool": call.get("tool"), "tool_call_seq": call.get("seq")})
    if not any(call.get("tool") == "cycle_summary" for call in tool_calls):
        blockers.append({"code": "missing_cycle_summary", "message": "cycle_summary must be consulted"})
    if not any(call.get("tool") == "past_memos" for call in tool_calls):
        blockers.append({"code": "missing_past_memos", "message": "past_memos must be consulted"})

    known_ids = set()
    for call in tool_calls:
        known_ids.update(_collect_ids(call.get("result")))
    for category, advice in output.advice.items():
        for record_id in (*advice.supporting_ids, *advice.contradicting_ids):
            if record_id not in known_ids:
                blockers.append({"code": "unknown_citation", "category": category, "record_id": record_id})

    prose = [output.summary, output.risk_level_rationale]
    for advice in output.advice.values():
        prose.extend([advice.reasoning, *advice.conditions, *advice.watch_next])
        if advice.conflict_rationale:
            prose.append(advice.conflict_rationale)
    prose.extend(h.statement for h in output.hypotheses)
    prose.extend(claim.statement for claim in output.numeric_claims)
    if any(UNSTRUCTURED_NUMBER.search(text) for text in prose):
        blockers.append({"code": "unstructured_numeric_claim",
                         "message": "put numeric facts only in numeric_claims, linked to a tool result"})

    for claim in output.numeric_claims:
        call = call_by_seq.get(claim.tool_call_seq)
        if call is None:
            blockers.append({"code": "unknown_numeric_source", "statement": claim.statement,
                             "tool_call_seq": claim.tool_call_seq})
            continue
        try:
            actual = json_pointer_value(call.get("result"), claim.json_pointer)
        except (ValueError, KeyError, IndexError, TypeError):
            blockers.append({"code": "invalid_numeric_source", "statement": claim.statement,
                             "tool_call_seq": claim.tool_call_seq, "json_pointer": claim.json_pointer})
            continue
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), claim.value, rel_tol=1e-6, abs_tol=1e-6
        ):
            blockers.append({"code": "numeric_claim_mismatch", "statement": claim.statement,
                             "claimed": claim.value, "tool_value": actual})

    # Only hypotheses past_memos returned (same version, earlier cycle) can be checked, once each.
    known_hypotheses = {h.get("hypothesis_id") for call in tool_calls if call.get("tool") == "past_memos"
                        and isinstance(call.get("result"), dict)
                        for h in call["result"].get("hypotheses") or [] if isinstance(h, dict)}
    checked = [check.hypothesis_id for check in output.hypothesis_checks]
    for hypothesis_id in sorted({h for h in checked if checked.count(h) > 1}):
        blockers.append({"code": "duplicate_hypothesis_check", "hypothesis_id": hypothesis_id})
    for check in output.hypothesis_checks:
        if check.hypothesis_id not in known_hypotheses:
            blockers.append({"code": "unknown_hypothesis", "hypothesis_id": check.hypothesis_id})
        call = call_by_seq.get(check.evidence_tool_call_seq)
        if call is None:
            blockers.append({"code": "unknown_hypothesis_evidence", "hypothesis_id": check.hypothesis_id})
            continue
        try:
            json_pointer_value(call.get("result"), check.evidence_json_pointer)
        except (ValueError, KeyError, IndexError, TypeError):
            blockers.append({"code": "invalid_hypothesis_evidence", "hypothesis_id": check.hypothesis_id,
                             "tool_call_seq": check.evidence_tool_call_seq,
                             "json_pointer": check.evidence_json_pointer})
        for record_id in check.evidence_record_ids:
            if record_id not in known_ids:
                blockers.append({"code": "unknown_hypothesis_citation", "hypothesis_id": check.hypothesis_id,
                                 "record_id": record_id})

    stats = _category_stats(tool_calls)
    previous_advice = _prior_advice(tool_calls)
    for category in CATEGORIES:
        category_stats = stats.get(category)
        if category_stats and category_stats["n_scored"] < 10:
            flags.append({"code": "thin_evidence", "category": category,
                          "n_scored": category_stats["n_scored"]})
        if (category_stats
                and category_stats["p_rate_above_half"] >= POSTERIOR_REVIEW_THRESHOLD
                and previous_advice.get(category)
                and output.advice[category].advice != previous_advice[category]
                and not output.advice[category].conflict_rationale):
            flags.append({"code": "unexplained_strong_statistics_departure", "category": category,
                          "posterior_probability": category_stats["p_rate_above_half"],
                          "previous_advice": previous_advice[category],
                          "proposed_advice": output.advice[category].advice})

    cited_count = sum(len(a.supporting_ids) + len(a.contradicting_ids) for a in output.advice.values())
    covered_categories = sum(bool(a.supporting_ids or a.contradicting_ids) for a in output.advice.values())
    score = (covered_categories, cited_count, len(output.numeric_claims))
    return SafetyResult(not blockers, blockers, flags, level, score)

