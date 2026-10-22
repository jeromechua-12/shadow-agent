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


def test_parallel_workers_retry_rate_limits(cache_dir, monkeypatch):
    """3 pairs on 2 workers; the first attempt of one pair is rate limited."""
    import threading

    monkeypatch.setattr(cache_runner, "RATE_LIMIT_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(cache_runner, "STAGGER_S", 0)
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)
    attempts, lock = [], threading.Lock()

    def fake_run(ticker, date):
        with lock:
            attempts.append((ticker, date))
            first = attempts.count((ticker, date)) == 1
        if ticker == "AAPL" and first:
            raise RuntimeError("Error code: 429 - rate limited")
        return {"run_id": f"{ticker}{date}", "complete": True, "incomplete_reasons": [],
                "duration_s": 0, "usage": {}}

    monkeypatch.setattr(cache_runner, "run_upstream", fake_run)
    rc = cache_runner.main(["--tickers", "NVDA", "AAPL", "MSFT", "--dates", "2026-09-01",
                            "--max-runs", "3", "--workers", "2"])
    assert rc == 0
    assert attempts.count(("AAPL", "2026-09-01")) == 2
    for t in ("NVDA", "AAPL", "MSFT"):
        assert cache_runner.read_cache(t, "2026-09-01")["complete"]


def test_refuses_more_runs_than_allowed(cache_dir, capsys):
    rc = cache_runner.main(["--tickers", "NVDA", "AAPL", "--dates", "2026-09-01"])
    assert rc == 2
    assert "refusing: 2 runs planned" in capsys.readouterr().out
