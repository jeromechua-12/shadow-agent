"""Calibration runner: one model attempt per cycle, one repair turn on failed checks. Nothing touches the network."""

import json

from shadow.data import calibration_runner
from shadow.data.calibration_agent import AttemptResult
from shadow.data.db import connect, init_db


def test_one_attempt_per_cycle(tmp_path, monkeypatch):
    path = tmp_path / "c.duckdb"
    init_db(path).close()
    calls = []

    def fake_attempt(model, tools, **kw):
        calls.append(kw["attempt_no"])
        return AttemptResult(output=None, tool_calls=[], error="no output", status="error")

    monkeypatch.setattr(calibration_runner, "run_attempt", fake_attempt)
    result = calibration_runner.run_cycle(as_of="2026-09-28", experiment_version="feedback", db_path=path,
                                          llm=object())
    assert calls == [1]
    assert result["attempts"] == 1 and result["status"] == "no_valid_memo"
    con = connect(path, read_only=True)
    assert con.execute("SELECT count(*) FROM calibration_attempts").fetchone() == (1,)
    con.close()


def test_a_memo_the_writer_rejects_is_reported_not_raised(tmp_path, monkeypatch):
    from shadow.data.calibration_models import CATEGORIES, CalibrationOutput
    from shadow.data.writers import InvalidRecord

    path = tmp_path / "c.duckdb"
    init_db(path).close()
    output = CalibrationOutput.model_validate({
        "summary": "Evidence is mixed.", "risk_level": 3, "risk_level_rationale": "First cycle.",
        "advice": {c: {"advice": "keep", "confidence": "low", "reasoning": "Thin."} for c in CATEGORIES}})
    calls = [{"seq": 1, "tool": "cycle_summary", "args": {}, "result": {}},
             {"seq": 2, "tool": "past_memos", "args": {}, "result": {"memos": [], "hypotheses": []}}]

    def reject(*args, **kwargs):
        raise InvalidRecord("memo", ["writer said no"])

    monkeypatch.setattr(calibration_runner, "run_attempt",
                        lambda model, tools, **kw: AttemptResult(output=output, tool_calls=calls))
    monkeypatch.setattr(calibration_runner, "insert_memo", reject)
    result = calibration_runner.run_cycle(as_of="2026-09-28", experiment_version="feedback", db_path=path,
                                          llm=object())
    assert result["status"] == "memo_rejected" and result["error"] == "writer said no"
    con = connect(path, read_only=True)
    status, passed, blockers = con.execute(
        "SELECT status, safety_passed, safety_blockers FROM calibration_attempts").fetchone()
    con.close()
    assert (status, passed) == ("memo_rejected", False)               # the row explains the missing memo
    assert {"code": "writer_rejected", "problems": ["writer said no"]} in json.loads(blockers)


def test_a_valid_attempt_writes_one_memo(tmp_path, monkeypatch):
    from shadow.data.calibration_models import CATEGORIES, CalibrationOutput

    path = tmp_path / "c.duckdb"
    init_db(path).close()
    output = CalibrationOutput.model_validate({
        "summary": "Evidence is mixed.", "risk_level": 3, "risk_level_rationale": "First cycle.",
        "advice": {c: {"advice": "keep", "confidence": "low", "reasoning": "Thin."} for c in CATEGORIES}})
    calls = [{"seq": 1, "tool": "cycle_summary", "args": {}, "result": {}},
             {"seq": 2, "tool": "past_memos", "args": {}, "result": {"memos": [], "hypotheses": []}}]
    monkeypatch.setattr(calibration_runner, "run_attempt",
                        lambda model, tools, **kw: AttemptResult(output=output, tool_calls=calls))
    result = calibration_runner.run_cycle(as_of="2026-09-28", experiment_version="feedback", db_path=path,
                                          llm=object())
    assert result["status"] == "memo_written"
    con = connect(path, read_only=True)
    assert con.execute("SELECT memo_id FROM memos").fetchall() == [(result["memo_id"],)]
    assert con.execute("SELECT status, safety_passed FROM calibration_attempts").fetchone() == ("ok", True)
    con.close()


def test_a_failed_check_gets_one_repair_turn(tmp_path, monkeypatch):
    from shadow.data.calibration_models import CATEGORIES, CalibrationOutput

    path = tmp_path / "c.duckdb"
    init_db(path).close()
    payload = {"summary": "Evidence is mixed.", "risk_level": 3, "risk_level_rationale": "First cycle.",
               "advice": {c: {"advice": "keep", "confidence": "low", "reasoning": "Thin."} for c in CATEGORIES}}
    fixed = CalibrationOutput.model_validate(payload)
    broken = CalibrationOutput.model_validate({**payload, "summary": "Success rate is 0.49."})
    calls = [{"seq": 1, "tool": "cycle_summary", "args": {}, "result": {}},
             {"seq": 2, "tool": "past_memos", "args": {}, "result": {"memos": [], "hypotheses": []}}]
    seen = []

    def fake_repair(model, tools, prior, blockers):
        seen.extend(b["code"] for b in blockers)
        return AttemptResult(output=fixed, tool_calls=calls)

    monkeypatch.setattr(calibration_runner, "run_attempt",
                        lambda model, tools, **kw: AttemptResult(output=broken, tool_calls=calls))
    monkeypatch.setattr(calibration_runner, "repair_attempt", fake_repair)
    result = calibration_runner.run_cycle(as_of="2026-09-28", experiment_version="feedback", db_path=path,
                                          llm=object())
    assert seen == ["unstructured_numeric_claim"]
    assert result["status"] == "memo_written" and result["attempts"] == 2
    con = connect(path, read_only=True)
    assert con.execute("SELECT attempt_no, status FROM calibration_attempts ORDER BY attempt_no").fetchall() == [
        (1, "invalid"), (2, "ok")]
    con.close()


def test_calibration_llm_uses_the_protocol_temperature(monkeypatch):
    from shadow.data import calibration_agent
    from tradingagents.llm_clients import factory

    seen = {}

    class Client:
        def get_llm(self):
            return "llm"

    def fake_client(provider, model, **kwargs):
        seen.update(kwargs, provider=provider, model=model)
        return Client()

    monkeypatch.setattr(factory, "create_llm_client", fake_client)
    assert calibration_agent.build_calibration_llm() == "llm"
    assert seen["temperature"] == 0.2 and seen["provider"] == "openrouter"
