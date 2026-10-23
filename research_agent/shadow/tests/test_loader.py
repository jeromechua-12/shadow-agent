"""Loader: cache file + evidence index + news store -> DuckDB, idempotently. Fake files only."""

import json

import numpy as np
import pandas as pd
import pytest

from shadow.data import evidence, loader
from shadow.data.alpaca_news import AlpacaNewsStore
from shadow.data.checks import run_checks
from shadow.data.conditions import price_conditions
from shadow.data.db import init_db


def fake_calls(run_id="run_a"):
    news = {"call_id": f"{run_id}_news", "kind": "router", "node": "tools_news", "method": "get_news",
            "args": ["NVDA", "2026-08-25", "2026-09-01"], "kwargs": {}, "fetched_at": "2026-10-22T10:00:00+00:00",
            "seconds": 0.1, "vendor": "alpaca", "status": "ok", "retryable": False, "detail": None,
            "attempts": [{"vendor": "alpaca", "status": "ok"}], "log": [],
            "items": [{"source": "alpaca_news", "key": "NVDA", "ids": ["1", "2"]}], "output": "## NVDA News ..."}
    fred = {**news, "call_id": f"{run_id}_fred", "method": "get_macro_indicators", "args": ["vix", "2026-09-01", 90],
            "vendor": None, "status": "not_configured", "retryable": True, "detail": "FRED_API_KEY not set",
            "attempts": [{"vendor": "fred", "status": "not_configured"}], "items": [], "output": "DATA_UNAVAILABLE: ..."}
    social = {**news, "call_id": f"{run_id}_st", "kind": "social", "node": "Sentiment Analyst",
              "method": "fetch_stocktwits_messages", "args": ["NVDA"], "vendor": "stocktwits",
              "status": "blocked", "retryable": False, "attempts": [], "items": [], "output": "<stocktwits unavailable>"}
    for c in (news, fred, social):
        c["output_chars"] = len(c["output"])
        c["output_sha256"] = evidence.hashlib.sha256(c["output"].encode()).hexdigest()
    return [news, fred, social]


def fake_record(run_id="run_a", created="2026-10-22T10:00:00+00:00", complete=True, evidence_index=None):
    return {
        "cache_version": 1, "run_id": run_id, "ticker": "NVDA", "trade_date": "2026-09-01",
        "created_at": created, "duration_s": 400.0, "complete": complete, "incomplete_reasons": [],
        "model": {"provider": "openrouter", "deep_think_llm": "deepseek/x", "quick_think_llm": "deepseek/x",
                  "temperature": 0.2, "max_tokens": 16384},
        "settings": {"analysts": ["market"], "max_debate_rounds": 1, "data_vendors": {"news_data": "yfinance"},
                     "tool_vendors": {"get_news": "alpaca"}},
        "code_version": {"commit": "abc", "dirty": False},
        "usage": {"llm_calls": 2, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0172},
        "llm_calls": [{"node": "Market Analyst", "finish_reason": "stop", "output_tokens": 5, "cost_usd": 0.01},
                      {"node": "Trader", "finish_reason": "tool_calls", "output_tokens": 0, "cost_usd": 0.0072}],
        "outputs": {"instrument_context": "NVIDIA", "market_report": "m", "sentiment_report": "s",
                    "news_report": "n", "fundamentals_report": "f",
                    "investment_debate": {"bull_history": "\nBull Analyst: up", "bear_history": "\nBear Analyst: down",
                                          "history": "", "judge_decision": "plan"},
                    "investment_plan": "plan", "trader_investment_plan": "sell it"},
        "structured": {"sentiment_report": {"overall_band": "Bullish", "overall_score": 6.5, "confidence": "low"},
                       "research_plan": {"recommendation": "Hold"},
                       "trader_proposal": {"action": "Sell", "entry_price": 217.2, "stop_loss": 231.0,
                                           "position_sizing": "trim", "reasoning": "r"}},
        "data_sources": {"evidence_index": evidence_index, "summary": {},
                         "calls": [evidence.summary(c) for c in fake_calls(run_id)]},
    }


@pytest.fixture
def setup(tmp_path):
    warehouse, cache = tmp_path / "warehouse", tmp_path / "upstream_cache"
    store = AlpacaNewsStore(warehouse / "raw" / "alpaca_news")
    store._append("NVDA", [
        {"id": 1, "created_at": "2026-08-31T14:00:00Z", "headline": "Nvidia earnings beat", "summary": "s",
         "source": "benzinga", "symbols": ["NVDA"], "fetched_at": "2026-10-22T09:00:00+00:00"},
        {"id": 2, "created_at": "2026-09-01T20:00:00Z", "headline": "Chip stocks slide", "summary": "",
         "source": "benzinga", "symbols": ["NVDA", "AMD"], "fetched_at": "2026-10-22T09:00:00+00:00"}])
    index = evidence.persist(fake_calls(), "run_a", "NVDA", "2026-09-01", root=warehouse / "raw")
    rel = index.relative_to(warehouse).as_posix()
    (cache / "NVDA").mkdir(parents=True)
    (cache / "NVDA" / "2026-09-01.json").write_text(json.dumps(fake_record(evidence_index=rel)), encoding="utf-8")
    con = init_db(tmp_path / "l.duckdb")
    yield con, warehouse, cache
    con.close()


def market(ticker, day):
    inputs = [{"input": i, "source": s, "series": x, "status": "ok", "payload": f"{x} data"}
              for i, s, x in (("prices", "yfinance", ticker), ("vix", "fred", "VIXCLS"), ("yield_curve", "fred", "T10Y2Y"))]
    return {"values": {"close": 217.2, "atr_pct": 0.031, "rsi_14": 55.0, "vix": 16.34, "yield_curve": 0.4},
            "inputs": inputs}


def load(con, warehouse, cache):
    return loader.load_all(con, cache_dir=cache, store_dir=warehouse / "raw" / "alpaca_news",
                           market=market, warehouse=warehouse)


def test_loads_every_table_and_is_idempotent(setup):
    con, warehouse, cache = setup
    assert load(con, warehouse, cache) == {"news_items_added": 2, "runs_loaded": 1, "runs_already_loaded": 0, "links_added": 2, "failed": []}
    assert load(con, warehouse, cache) == {"news_items_added": 0, "runs_loaded": 0, "runs_already_loaded": 1, "links_added": 0, "failed": []}

    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    assert q("SELECT run_id, temperature, cost_usd FROM runs") == [("run_a", 0.2, 0.0172)]
    assert q("SELECT count(*) FROM llm_calls") == [(2,)]
    assert q("SELECT action, direction, entry_price, stop_loss, structured_ok FROM trader_proposals") == [
        ("Sell", -1, 217.2, 231.0, True)]
    assert q("SELECT method, agent, status, retryable, vendors_tried FROM evidence_calls ORDER BY method") == [
        ("fetch_stocktwits_messages", "sentiment", "blocked", False, ["stocktwits"]),
        ("get_macro_indicators", "news", "not_configured", True, ["fred"]),
        ("get_news", "news", "ok", False, ["alpaca"])]
    assert q("SELECT item_id, rank FROM run_text_items ORDER BY rank") == [("alpaca_news:1", 1), ("alpaca_news:2", 2)]
    assert q("SELECT count(*) FROM agent_outputs") == [(8,)]
    cond = con.execute("SELECT vix, atr_pct, sentiment_score, sentiment_confidence, news_items_seen, "
                       "missing_sources FROM latest_conditions").fetchone()
    assert cond == (16.34, 0.031, 6.5, "low", 2, ["fetch_stocktwits_messages", "get_macro_indicators"])


def test_payloads_resolve_and_checks_pass(setup):
    con, warehouse, cache = setup
    load(con, warehouse, cache)
    paths = con.execute("SELECT payload_path FROM evidence_calls").fetchall()
    assert all(p[0] and (warehouse / "raw" / p[0]).exists() for p in paths)
    errors = [f for f in run_checks(con, warehouse) if f.count and f.severity == "error"]
    assert errors == []
    warns = {f.check for f in run_checks(con, warehouse) if f.count}
    assert "retryable_data_failures" in warns  # the FRED call


def test_superseded_runs_are_history_and_current_view_picks_latest(setup):
    con, warehouse, cache = setup
    old = fake_record(run_id="run_old", created="2026-10-20T10:00:00+00:00", complete=False)
    (cache / "NVDA" / "_superseded").mkdir()
    (cache / "NVDA" / "_superseded" / "2026-09-01.run_old.json").write_text(json.dumps(old), encoding="utf-8")
    assert load(con, warehouse, cache)["runs_loaded"] == 2
    assert con.execute("SELECT run_id FROM current_runs").fetchall() == [("run_a",)]


def test_missing_evidence_index_falls_back_to_cache_summary(setup):
    con, warehouse, cache = setup
    record = fake_record(run_id="run_b", evidence_index="raw/evidence/NVDA/2026-09-01/nope.jsonl")
    (cache / "NVDA" / "2026-09-02.json").write_text(json.dumps(record), encoding="utf-8")
    load(con, warehouse, cache)
    assert con.execute("SELECT count(*) FROM evidence_calls WHERE run_id = 'run_b'").fetchone() == (3,)


def test_append_rows_guards_names_and_duplicates(setup):
    con, *_ = setup
    with pytest.raises(ValueError, match="unknown table"):
        loader.append_rows(con, "runs; DROP TABLE runs", [{"run_id": "x"}])
    with pytest.raises(ValueError, match="no column"):
        loader.append_rows(con, "memos", [{"memo_id": "m", "evil) VALUES (1); --": 1}])
    row = {"snapshot_date": "2026-09-01", "book": "shadow", "experiment_version": "v", "cash": 1.0,
           "value": 1.0, "drawdown": 0.0}
    assert loader.append_rows(con, "book_snapshots", [row]) == 1
    with pytest.raises(Exception, match="(?i)constraint|duplicate"):
        loader.append_rows(con, "book_snapshots", [row])
    assert loader.append_rows(con, "book_snapshots", [row], ignore_existing=True) == 0


def test_price_conditions():
    n = 260
    close = pd.Series(np.linspace(100, 200, n))  # steady climb
    prices = pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close})
    c = price_conditions(prices)
    assert c["close"] == 200
    assert c["rsi_14"] == 100.0                      # no down days
    assert c["atr_pct"] == pytest.approx(2.0 / 200, rel=0.05)
    assert c["pct_vs_sma200"] == pytest.approx(200 / close.iloc[-200:].mean() - 1)
    short = price_conditions(prices.iloc[:30])
    assert short["pct_vs_sma200"] is None and short["rsi_14"] is not None
