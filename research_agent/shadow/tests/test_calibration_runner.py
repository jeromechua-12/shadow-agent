"""Calibration runner: one model attempt per cycle. Fake attempt; nothing touches the network."""

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
