"""Cache runner bookkeeping: skip cached, rerun incomplete, never overwrite. No LLM."""

import json

import pytest

from shadow.data import cache_runner


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOW_UPSTREAM_CACHE_DIR", str(tmp_path))
    return tmp_path


def write(ticker, date, **record):
    path = cache_runner.cache_path(ticker, date)
    cache_runner._write_atomic(path, {"run_id": "r1", **record})
    return path


def test_plan_skips_cached_and_reruns_only_incomplete(cache_dir):
    write("NVDA", "2026-09-01", complete=True, incomplete_reasons=[])
    write("NVDA", "2026-09-02", complete=False, incomplete_reasons=["empty fundamentals_report"])
    dates = ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert [d for _, d, _ in cache_runner.plan(["NVDA"], dates, rerun_incomplete=False)] == ["2026-09-03"]
    assert [d for _, d, _ in cache_runner.plan(["NVDA"], dates, rerun_incomplete=True)] == ["2026-09-02", "2026-09-03"]


def test_supersede_keeps_old_file(cache_dir):
    path = write("NVDA", "2026-09-01", complete=False)
    cache_runner._supersede(path, json.loads(path.read_text()))
    assert not path.exists()
    assert (path.parent / "_superseded" / "2026-09-01.r1.json").exists()


def test_incomplete_reasons():
    outputs = {"market_report": "m", "sentiment_report": "s", "news_report": "n",
               "fundamentals_report": "", "investment_plan": "p", "trader_investment_plan": "t",
               "investment_debate": {"bull_history": "\nBull Analyst: case", "bear_history": "\nBear Analyst: "}}
    calls = [
        {"method": "get_macro_indicators", "status": "not_configured", "retryable": True, "detail": "no FRED key"},
        {"method": "fetch_reddit_posts", "status": "unavailable", "retryable": False, "detail": "recent only"},
    ]
    assert cache_runner._incomplete_reasons(outputs, calls) == [
        "empty bear_history", "empty fundamentals_report", "get_macro_indicators not_configured: no FRED key"]
    assert "trader proposal not structured (free-text fallback)" in cache_runner._incomplete_reasons(
        outputs, calls, {"trader_proposal": None})
    assert not any("trader proposal" in r for r in cache_runner._incomplete_reasons(
        outputs, calls, {"trader_proposal": {"action": "Buy"}}))


def test_parallel_workers_retry_rate_limits(cache_dir, monkeypatch):
    """3 pairs on 2 workers; the first attempt of one pair is rate limited."""
    import threading

    monkeypatch.setattr(cache_runner, "RATE_LIMIT_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(cache_runner, "STAGGER_S", 0)
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)
    attempts, lock = [], threading.Lock()

    def fake_run(ticker, date, temperature=0.2):
        with lock:
            attempts.append((ticker, date))
            first = attempts.count((ticker, date)) == 1
        if ticker == "AAPL" and first:
            raise RuntimeError("Error code: 429 - rate limited")
        return {"run_id": f"{ticker}{date}", "complete": True, "incomplete_reasons": [], "model": {"temperature": temperature},
                "duration_s": 0, "usage": {}}

    monkeypatch.setattr(cache_runner, "run_upstream", fake_run)
    rc = cache_runner.main(["--tickers", "NVDA", "AAPL", "MSFT", "--dates", "2026-09-01",
                            "--max-runs", "4", "--workers", "2"])
    assert rc == 0
    assert len(attempts) == 4
    assert attempts.count(("AAPL", "2026-09-01")) == 2
    for t in ("NVDA", "AAPL", "MSFT"):
        assert cache_runner.read_cache(t, "2026-09-01")["complete"]


def looped_or_ok(calls):
    """Fake run: loops to the token cap at temperature 0.2, completes at 0.6."""
    def fake_run(ticker, date, temperature=0.2):
        calls.append(temperature)
        looped = temperature < 0.6
        return {"run_id": f"t{temperature}", "complete": not looped,
                "incomplete_reasons": ["empty fundamentals_report"] if looped else [],
                "model": {"temperature": temperature}, "duration_s": 0, "usage": {"cost_usd": 0.03},
                "llm_calls": [{"node": "Fundamentals Analyst", "finish_reason": "length" if looped else "stop",
                               "content_chars": 0 if looped else 9000, "tool_calls": 0}]}
    return fake_run


def test_loop_at_token_cap_retries_once_at_higher_temperature(cache_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(cache_runner, "run_upstream", looped_or_ok(calls))
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)
    assert cache_runner.main(["--tickers", "NVDA", "--dates", "2026-09-02", "--max-runs", "2"]) == 0
    assert calls == [0.2, 0.6]
    saved = cache_runner.read_cache("NVDA", "2026-09-02")
    assert saved["complete"] and saved["model"]["temperature"] == 0.6
    assert saved["retry_of"]["run_id"] == "t0.2" and "Fundamentals Analyst" in saved["retry_of"]["reason"]
    assert (cache_dir / "NVDA" / "_superseded" / "2026-09-02.t0.2.json").exists()


def test_max_runs_stops_before_token_cap_retry_when_budget_is_one(cache_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(cache_runner, "run_upstream", looped_or_ok(calls))
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)

    assert cache_runner.main([
        "--tickers", "NVDA", "--dates", "2026-09-02", "--max-runs", "1",
    ]) == 0

    saved = cache_runner.read_cache("NVDA", "2026-09-02")
    assert calls == [0.2]
    assert not saved["complete"]
    assert "max-runs budget exhausted before token-cap retry" in saved["incomplete_reasons"]


def test_attempt_budget_reserves_one_start_for_each_pair(cache_dir, monkeypatch):
    monkeypatch.setattr(cache_runner, "RATE_LIMIT_BASE_DELAY_S", 0)
    monkeypatch.setattr(cache_runner, "STAGGER_S", 0)
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)
    attempts = []

    def fake_run(ticker, date, temperature=0.2):
        attempts.append(ticker)
        if ticker == "NVDA":
            raise RuntimeError("Error code: 429 - rate limited")
        return {"run_id": "msft", "complete": True, "incomplete_reasons": [],
                "model": {"temperature": temperature}, "duration_s": 0, "usage": {}}

    monkeypatch.setattr(cache_runner, "run_upstream", fake_run)
    rc = cache_runner.main(["--tickers", "NVDA", "MSFT", "--dates", "2026-09-01",
                            "--max-runs", "3", "--workers", "1"])

    assert rc == 1
    assert attempts == ["NVDA", "NVDA", "MSFT"]
    assert cache_runner.read_cache("MSFT", "2026-09-01")["complete"]
    failed = list((cache_dir / "NVDA" / "_failed").glob("*.json"))
    assert "budget exhausted" in failed[0].read_text()


def test_rerun_of_a_looped_pair_starts_hot(cache_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(cache_runner, "run_upstream", looped_or_ok(calls))
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)
    looped = looped_or_ok([])("NVDA", "2026-09-02", 0.2)
    cache_runner._write_atomic(cache_runner.cache_path("NVDA", "2026-09-02"), looped)
    assert cache_runner.main(["--tickers", "NVDA", "--dates", "2026-09-02", "--rerun-incomplete"]) == 0
    assert calls == [0.6]


def test_rerun_flag_and_sample_cache(cache_dir, tmp_path, monkeypatch, capsys):
    sample = tmp_path / "sample"
    monkeypatch.setenv("SHADOW_SAMPLE_CACHE_DIR", str(sample))
    assert cache_runner.main(["--tickers", "JPM", "--dates", "2026-09-03", "--sample", "--dry-run"]) == 0
    assert f"cache: {sample}" in capsys.readouterr().out
    # A complete cached pair is only rerun when asked.
    cache_runner._write_atomic(sample / "JPM" / "2026-09-03.json", {"run_id": "old", "complete": True})
    assert cache_runner.plan(["JPM"], ["2026-09-03"], rerun_incomplete=True) == []
    assert cache_runner.plan(["JPM"], ["2026-09-03"], False, rerun=True)[0][2].startswith("rerun requested")


def test_refuses_more_runs_than_allowed(cache_dir, capsys):
    rc = cache_runner.main(["--tickers", "NVDA", "AAPL", "--dates", "2026-09-01"])
    assert rc == 2
    assert "2 ticker/date pairs need one attempt each" in capsys.readouterr().out
