from datetime import date

from shadow.data.calibration_models import CATEGORIES, CalibrationOutput
from shadow.data.calibration_safety import validate_output


def output(**changes):
    data = {
        "summary": "The evidence is mixed.",
        "risk_level": 3,
        "risk_level_rationale": "Keep the balanced stance while evidence is mixed.",
        "advice": {
            category: {"advice": "keep", "confidence": "low", "reasoning": "Evidence is limited."}
            for category in CATEGORIES
        },
        "numeric_claims": [],
        "hypotheses": [],
        "hypothesis_checks": [],
    }
    data.update(changes)
    return CalibrationOutput.model_validate(data)


def summary_call(probability=0.5, n=3):
    by_category = {
        category: {"all_history": {"beta": {"p_rate_above_half": 0.5}, "n_scored": 0}}
        for category in CATEGORIES
    }
    by_category["event_risk"] = {
        "all_history": {"beta": {"p_rate_above_half": probability}, "n_scored": n}
    }
    return {"seq": 1, "tool": "cycle_summary", "args": {},
            "result": {"by_category": by_category}}


def past_call(advice="keep"):
    return {"seq": 2, "tool": "past_memos", "args": {},
            "result": {"memos": [{"cycle_date": "2026-08-31", "advice": [
                {"category": "event_risk", "advice": advice}
            ]}]}}


def test_missing_snapshot_holds_proposed_increase_but_keeps_memo():
    result = validate_output(
        output(risk_level=4), [summary_call(), past_call()], previous_risk_level=3,
        decision_drawdown=None, snapshot_date=None, as_of=date(2026, 9, 28),
    )
    assert result.passed
    assert result.accepted_risk_level == 3
    assert any(flag["code"] == "risk_increase_unverified" for flag in result.flags)


def test_large_drawdown_blocks_risk_increase():
    result = validate_output(
        output(risk_level=4), [summary_call(), past_call()], previous_risk_level=3,
        decision_drawdown=-0.11, snapshot_date=date(2026, 9, 28), as_of=date(2026, 9, 28),
    )
    assert not result.passed
    assert any(blocker["code"] == "drawdown_limit" for blocker in result.blockers)


def test_unknown_citation_blocks_output():
    result = validate_output(
        output(advice={**output().advice, "event_risk": output().advice["event_risk"].model_copy(
            update={"supporting_ids": ["dec_missing"]})}),
        [summary_call(), past_call()], previous_risk_level=3, decision_drawdown=0.0,
        snapshot_date=date(2026, 9, 28), as_of=date(2026, 9, 28),
    )
    assert not result.passed
    assert any(blocker["code"] == "unknown_citation" for blocker in result.blockers)


def test_numeric_claim_must_match_exact_tool_result():
    model_output = output(numeric_claims=[{
        "statement": "The posterior probability is",
        "value": 0.9,
        "unit": "fraction",
        "tool_call_seq": 2,
        "json_pointer": "/probability",
    }])
    calls = [summary_call(), past_call(), {"seq": 3, "tool": "subgroup_stats", "args": {},
                              "result": {"probability": 0.8}}]
    model_output = model_output.model_copy(update={"numeric_claims": [model_output.numeric_claims[0].model_copy(
        update={"tool_call_seq": 3})]})
    result = validate_output(model_output, calls, previous_risk_level=3,
                             decision_drawdown=0.0, snapshot_date=date(2026, 9, 28),
                             as_of=date(2026, 9, 28))
    assert not result.passed
    assert any(blocker["code"] == "numeric_claim_mismatch" for blocker in result.blockers)


def _codes(model_output, calls):
    result = validate_output(model_output, calls, previous_risk_level=3, decision_drawdown=0.0,
                             snapshot_date=date(2026, 9, 28), as_of=date(2026, 9, 28))
    return {blocker["code"] for blocker in result.blockers}


def _check(hypothesis_id):
    return {"hypothesis_id": hypothesis_id, "status": "held", "evidence_tool_call_seq": 2,
            "evidence_json_pointer": "/hypotheses"}


def test_hypothesis_checks_must_name_a_known_hypothesis_once():
    calls = [summary_call(), past_call()]
    calls[1]["result"]["hypotheses"] = [{"hypothesis_id": "hyp_known"}]
    assert not _codes(output(hypothesis_checks=[_check("hyp_known")]), calls)
    assert "unknown_hypothesis" in _codes(output(hypothesis_checks=[_check("hyp_made_up")]), calls)
    assert "duplicate_hypothesis_check" in _codes(
        output(hypothesis_checks=[_check("hyp_known"), _check("hyp_known")]), calls)


def test_a_tool_outside_the_five_blocks_output():
    calls = [summary_call(), past_call(), {"seq": 3, "tool": "drop_table", "args": {}, "result": {}}]
    assert "unknown_tool" in _codes(output(), calls)


def test_whole_numbers_may_appear_in_text_but_statistics_may_not():
    calls = [summary_call(), past_call()]
    assert not _codes(output(risk_level_rationale="Hold level 3 on the first cycle; watch 10-day outcomes."), calls)
    assert "unstructured_numeric_claim" in _codes(output(summary="The win rate was 0.62."), calls)
    assert "unstructured_numeric_claim" in _codes(output(summary="The win rate was 62%."), calls)


def test_85_percent_signal_flags_unexplained_departure_from_prior_advice():
    base = output()
    advice = dict(base.advice)
    advice["event_risk"] = advice["event_risk"].model_copy(update={"advice": "loosen"})
    calls = [summary_call(probability=0.9, n=12), {
        "seq": 2, "tool": "past_memos", "args": {},
        "result": {"memos": [{"cycle_date": "2026-08-31", "advice": [
            {"category": "event_risk", "advice": "keep"}
        ]}]},
    }]
    result = validate_output(output(advice=advice), calls, previous_risk_level=3,
                             decision_drawdown=0.0, snapshot_date=date(2026, 9, 28),
                             as_of=date(2026, 9, 28))
    assert result.passed
    assert any(flag["code"] == "unexplained_strong_statistics_departure"
               for flag in result.flags)


def test_conflict_rationale_suppresses_unexplained_departure_flag():
    base = output()
    advice = dict(base.advice)
    advice["event_risk"] = advice["event_risk"].model_copy(
        update={"advice": "loosen", "conflict_rationale": "The macro-news subgroup missed gains."})
    calls = [summary_call(probability=0.9, n=12), {
        "seq": 2, "tool": "past_memos", "args": {},
        "result": {"memos": [{"cycle_date": "2026-08-31", "advice": [
            {"category": "event_risk", "advice": "keep"}
        ]}]},
    }]
    result = validate_output(output(advice=advice), calls, previous_risk_level=3,
                             decision_drawdown=0.0, snapshot_date=date(2026, 9, 28),
                             as_of=date(2026, 9, 28))
    assert result.passed
    assert not any(flag["code"] == "unexplained_strong_statistics_departure"
                   for flag in result.flags)
