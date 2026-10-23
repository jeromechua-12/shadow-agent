"""Regression tests for the review findings (numbered as in the review). Fake data only."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from shadow.data import loader
from shadow.data.alpaca_news import AlpacaNewsStore
from shadow.data.checks import run_checks
from shadow.data.db import connect, init_db
from shadow.data.loader import append_rows
from shadow.data.queries import CalibrationTools
from shadow.data.writers import (
    InvalidRecord,
    insert_book_snapshot,
    insert_decision,
    insert_memo,
    insert_outcome,
)

# NYSE days 2026-08-31 .. 2026-11-30, with Labor Day (09-07) and Thanksgiving (11-26) closed.
CAL = [d.date() for d in pd.bdate_range("2026-08-31", "2026-11-30") if d.date() not in
       (date(2026, 9, 7), date(2026, 11, 26))]
RATINGS = {"volatility": 1, "event_risk": 1, "valuation": 1, "liquidity": 0, "concentration": 0}
ADVICE = {c: {"advice": "keep", "confidence": "low", "reasoning": "r", "cited_ids": []}
          for c in ("volatility", "event_risk", "valuation", "liquidity", "concentration")}


def add_run(con, run_id, day, ticker="NVDA", complete=True, started="2026-10-22T10:00:00+00:00", action="Buy"):
    append_rows(con, "runs", [{"run_id": run_id, "ticker": ticker, "trade_date": day, "started_at": started,
                               "cache_version": 1, "complete": complete}])
    append_rows(con, "trader_proposals", [{"run_id": run_id, "ticker": ticker, "trade_date": day, "action": action,
                                           "direction": 1, "entry_price": 100.0, "stop_loss": 95.0,
                                           "structured_ok": True}])


def decide(con, run_id, version="v1", **over):
    return insert_decision(con, **{"run_id": run_id, "experiment_version": version, "verdict": "approved",
                                   "proposed_size": 1.0, "final_size": 1.0, "risk_ratings": RATINGS,
                                   "main_reason": "none", **over})


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "r.duckdb")
    yield con
    con.close()


# ---- 1. outcome dates follow the trading calendar ---------------------------------

def test_1_outcome_dates_derived_from_calendar_across_a_holiday(con):
    add_run(con, "r1", "2026-09-04")  # Friday before Labor Day
    (d,) = decide(con, "r1")
    out = insert_outcome(con, decision_id=d, stock_return=0.02, spy_return=0.0, calendar=CAL)
    entry, resolved = con.execute("SELECT entry_date, resolved_on FROM outcomes").fetchone()
    assert entry == date(2026, 9, 8)                     # Monday is closed
    assert resolved == CAL[CAL.index(entry) + 10] == date(2026, 9, 22)
    assert out["label"] == "correct_approval"


@pytest.mark.parametrize("over, message", [
    ({"entry_date": "2026-09-07"}, "next trading day"),
    ({"resolved_on": "2026-09-21"}, "10th trading day after entry"),
    ({"horizon_days": 5}, "horizon_days must be 10"),
])
def test_1_outcome_dates_must_match_calendar(con, over, message):
    add_run(con, "r1", "2026-09-04")
    (d,) = decide(con, "r1")
    with pytest.raises(InvalidRecord, match=message):
        insert_outcome(con, decision_id=d, stock_return=0.02, spy_return=0.0, calendar=CAL, **over)


def test_1_outcome_not_resolved_yet(con):
    add_run(con, "r1", "2026-09-04")
    (d,) = decide(con, "r1")
    with pytest.raises(InvalidRecord, match="not resolved"):
        insert_outcome(con, decision_id=d, stock_return=0.0, spy_return=0.0, calendar=CAL[:8])


# ---- 2. text links survive a reload ---------------------------------------------------

def _cache_with_served_items(tmp_path, ids):
    call = {"call_id": "c1", "kind": "router", "node": "tools_news", "method": "get_news",
            "args": ["NVDA", "2026-08-25", "2026-09-01"], "kwargs": {}, "fetched_at": "2026-10-22T10:00:00+00:00",
            "vendor": "alpaca", "status": "ok", "retryable": False, "attempts": [{"vendor": "alpaca", "status": "ok"}],
            "items": [{"source": "alpaca_news", "key": "NVDA", "ids": ids}]}
    record = {"cache_version": 1, "run_id": "run_a", "ticker": "NVDA", "trade_date": "2026-09-01",
              "created_at": "2026-10-22T10:00:00+00:00", "complete": True, "incomplete_reasons": [],
              "model": {}, "settings": {}, "outputs": {"market_report": "m"}, "structured": {},
              "data_sources": {"calls": [call]}}
    cache = tmp_path / "cache"
    (cache / "NVDA").mkdir(parents=True)
    (cache / "NVDA" / "2026-09-01.json").write_text(json.dumps(record), encoding="utf-8")
    return cache


def _article(i, created="2026-08-31T14:00:00Z"):
    return {"id": i, "created_at": created, "headline": f"h{i}", "source": "benzinga", "symbols": ["NVDA"],
            "fetched_at": "2026-10-22T09:00:00+00:00"}


def test_2_reload_relinks_items_added_to_the_store_later(con, tmp_path):
    cache = _cache_with_served_items(tmp_path, ["1", "2"])
    store = AlpacaNewsStore(tmp_path / "news")
    store._append("NVDA", [_article(1)])
    loader.load_all(con, cache_dir=cache, store_dir=store.root)
    assert con.execute("SELECT item_id, rank FROM run_text_items").fetchall() == [("alpaca_news:1", 1)]
    assert {f.check for f in run_checks(con, tmp_path) if f.count} >= {"served_items_not_linked"}

    store._append("NVDA", [_article(2)])
    summary = loader.load_all(con, cache_dir=cache, store_dir=store.root)
    assert summary["links_added"] == 1
    assert con.execute("SELECT item_id, rank FROM run_text_items ORDER BY rank").fetchall() == [
        ("alpaca_news:1", 1), ("alpaca_news:2", 2)]
    assert "served_items_not_linked" not in {f.check for f in run_checks(con, tmp_path) if f.count}


# ---- 3. conditions: missing inputs recorded, backfill later, inputs saved as evidence ----

def _provider(fred_ok):
    def market(ticker, as_of):
        inputs = [{"input": "prices", "source": "yfinance", "series": ticker, "status": "ok",
                   "payload": "Date,Close\n2026-09-01,217.2\n"}]
        for name, series, value in (("vix", "VIXCLS", 16.34), ("yield_curve", "T10Y2Y", 0.41)):
            inputs.append({"input": name, "source": "fred", "series": series,
                           "status": "ok" if fred_ok else "not_configured",
                           "detail": None if fred_ok else "FRED_API_KEY not set",
                           "payload": json.dumps({"value": value}) if fred_ok else None})
        return {"values": {"close": 217.2, "atr_pct": 0.03, "vix": 16.34 if fred_ok else None,
                           "yield_curve": 0.41 if fred_ok else None}, "inputs": inputs}
    return market


def test_3_missing_fred_is_recorded_and_backfilled_later(con, tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOW_WAREHOUSE_DIR", str(tmp_path / "wh"))
    cache = _cache_with_served_items(tmp_path, [])
    loader.load_all(con, cache_dir=cache, store_dir=tmp_path / "news", market=_provider(fred_ok=False))
    vix, status = con.execute("SELECT vix, input_status FROM latest_conditions").fetchone()
    assert vix is None and json.loads(status)["vix"] == "not_configured"
    assert "conditions_inputs_missing" in {f.check for f in run_checks(con, tmp_path / "wh") if f.count}
    rows = con.execute("SELECT input, status, payload_path FROM condition_inputs ORDER BY input").fetchall()
    assert rows[0][:2] == ("prices", "ok") and rows[0][2] and rows[1][:2] == ("vix", "not_configured")

    assert loader.backfill_conditions(con, market=_provider(fred_ok=True)) == 1
    assert con.execute("SELECT count(*) FROM conditions").fetchone() == (2,)   # append-only: both kept
    assert con.execute("SELECT vix FROM latest_conditions").fetchone() == (16.34,)
    assert loader.backfill_conditions(con, market=_provider(fred_ok=True)) == 0  # nothing left to fill
    assert "conditions_inputs_missing" not in {f.check for f in run_checks(con, tmp_path / "wh") if f.count}


# ---- 4. one decision set per version + ticker + date, only on current complete runs ----

def test_4_decision_needs_a_current_complete_run(con):
    add_run(con, "old", "2026-09-01", started="2026-10-20T10:00:00+00:00")
    add_run(con, "new", "2026-09-01", started="2026-10-21T10:00:00+00:00")
    add_run(con, "broken", "2026-09-02", complete=False)
    with pytest.raises(InvalidRecord, match="not the current run"):
        decide(con, "old")
    with pytest.raises(InvalidRecord, match="not the current run"):
        decide(con, "broken")
    decide(con, "new")


def test_4_one_decision_set_per_version_ticker_date(con):
    add_run(con, "first", "2026-09-01", started="2026-10-20T10:00:00+00:00")
    decide(con, "first")
    add_run(con, "rerun", "2026-09-01", started="2026-10-21T10:00:00+00:00")  # becomes current
    with pytest.raises(InvalidRecord, match="already has a decision set"):
        decide(con, "rerun")
    decide(con, "rerun", version="v2")  # another version may decide on it
    found = {f.check: f for f in run_checks(con) if f.count}
    assert "decision_run_not_current" in found  # the v1 decision now points at a superseded run


# ---- 5 + 6. news cutoff: same rule everywhere, in UTC whatever the session timezone ----

def _news_db(path):
    con = init_db(path)
    add_run(con, "r1", "2026-09-01")
    (d,) = decide(con, "r1")
    append_rows(con, "outcomes", [{"decision_id": d, "entry_date": "2026-09-02", "resolved_on": "2026-09-16",
                                   "stock_return": 0.05, "spy_return": 0.0, "score": 0.05,
                                   "label": "correct_approval"}])
    append_rows(con, "evidence_calls", [{"call_id": "c1", "run_id": "r1", "kind": "router", "method": "get_news",
                                         "as_of": "2026-09-01", "fetched_at": "2026-10-22T10:00:00+00:00",
                                         "status": "ok", "retryable": False}])
    # 03:00 UTC on 09-02 is still 09-01 in New York and Los Angeles, but after the UTC cutoff.
    append_rows(con, "text_items", [
        {"item_id": "n:early", "source": "n", "source_id": "early", "created_at": "2026-09-01T15:00:00+00:00",
         "first_seen_at": "2026-10-22T00:00:00+00:00", "title": "Fed holds rates"},
        {"item_id": "n:late", "source": "n", "source_id": "late", "created_at": "2026-09-02T03:00:00+00:00",
         "first_seen_at": "2026-10-22T00:00:00+00:00", "title": "Tariff shock overnight"}])
    append_rows(con, "run_text_items", [{"run_id": "r1", "call_id": "c1", "item_id": "n:early", "rank": 1},
                                        {"run_id": "r1", "call_id": "c1", "item_id": "n:late", "rank": 2}])
    con.close()


def test_5_news_keyword_uses_the_created_at_cutoff(tmp_path):
    _news_db(tmp_path / "n.duckdb")
    con = connect(tmp_path / "n.duckdb")
    tools = CalibrationTools(con, "2026-09-28", "v1")
    assert tools.subgroup_stats({"news_keyword": "fed"})["overall"]["n"] == 1
    assert tools.subgroup_stats({"news_keyword": "tariff"})["overall"]["n"] == 0


@pytest.mark.parametrize("tz", ["America/Los_Angeles", "Asia/Singapore"])
def test_6_cutoffs_are_utc_whatever_the_session_timezone(tmp_path, tz):
    _news_db(tmp_path / "n.duckdb")
    con = connect(tmp_path / "n.duckdb")
    con.execute(f"SET TimeZone = '{tz}'")
    tools = CalibrationTools(con, "2026-09-28", "v1")
    trade = tools.search_trades()["trades"][0]
    assert [n["item_id"] for n in tools.read_trade(trade["decision_id"])["news_seen"]] == ["n:early"]
    assert tools.subgroup_stats({"news_keyword": "tariff"})["overall"]["n"] == 0
    found = {f.check: f for f in run_checks(con) if f.count}
    assert found["news_after_trade_date"].examples == ["r1 n:late"]


def test_6_returned_timestamps_are_utc(tmp_path):
    _news_db(tmp_path / "n.duckdb")
    con = connect(tmp_path / "n.duckdb")
    con.execute("SET TimeZone = 'Asia/Singapore'")
    tools = CalibrationTools(con, "2026-09-28", "v1")
    news = tools.read_trade(tools.search_trades()["trades"][0]["decision_id"])["news_seen"]
    assert news[0]["created_at"] == "2026-09-01T15:00:00+00:00"


# ---- 7. no as-of-blind hypothesis view -----------------------------------------------

def test_7_hypothesis_status_view_is_gone(tmp_path):
    con = init_db(tmp_path / "h.duckdb")
    con.execute("CREATE OR REPLACE VIEW hypothesis_status AS SELECT 1 AS x")  # as an older schema had it
    con.close()
    con = init_db(tmp_path / "h.duckdb")
    views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    assert "hypothesis_status" not in views


# ---- 9. first memo at the default level; no increase without a recent snapshot ---------

def memo(con, day, level, version="v1", **kw):
    return insert_memo(con, experiment_version=version, cycle_date=day, risk_level=level, memo_text="m",
                       advice=ADVICE, safety_passed=True, **kw)


def test_9_first_memo_starts_at_default_level(con):
    with pytest.raises(InvalidRecord, match="first memo .* risk level 3"):
        memo(con, "2026-08-31", 4)
    memo(con, "2026-08-31", 3)


def test_9_increase_needs_a_recent_decision_book_snapshot(con):
    memo(con, "2026-08-31", 3)
    with pytest.raises(InvalidRecord, match="no decision-book snapshot"):
        memo(con, "2026-09-28", 4)
    insert_book_snapshot(con, experiment_version="v1", book="decision", snapshot_date="2026-09-10", cash=0.0, value=100.0)
    with pytest.raises(InvalidRecord, match="no decision-book snapshot"):
        memo(con, "2026-09-28", 4)                     # 18 days old: not recent
    insert_book_snapshot(con, experiment_version="v1", book="decision", snapshot_date="2026-09-25", cash=0.0, value=101.0)
    memo(con, "2026-09-28", 4)


# ---- 10. hypothesis checks: same version, later cycle ------------------------------------

def test_10_writer_rejects_checks_from_another_version(con):
    a = memo(con, "2026-08-31", 3, version="A",
             hypotheses=[{"statement": "s", "category": "event_risk", "subgroup": {"main_reason": "event_risk"}}])
    (hyp,) = a["hypothesis_ids"]
    with pytest.raises(InvalidRecord, match="belongs to experiment version 'A'"):
        memo(con, "2026-09-28", 3, version="B", hypothesis_checks=[{"hypothesis_id": hyp, "status": "held"}])
    memo(con, "2026-09-28", 3, version="A", hypothesis_checks=[{"hypothesis_id": hyp, "status": "held"}])


def test_10_check_flags_rows_written_around_the_writer(con):
    a = memo(con, "2026-08-31", 3, version="A", hypotheses=[{"statement": "s", "subgroup": {}}])
    (hyp,) = a["hypothesis_ids"]
    # Hand-built rows: an earlier memo, and another version's memo, both "checking" it.
    append_rows(con, "memos", [
        {"memo_id": "early", "experiment_version": "A", "cycle_date": "2026-08-24",
         "created_at": "2026-08-24T00:00:00+00:00", "risk_level": 3, "memo_text": "m", "safety_passed": True},
        {"memo_id": "other", "experiment_version": "B", "cycle_date": "2026-09-28",
         "created_at": "2026-09-28T00:00:00+00:00", "risk_level": 3, "memo_text": "m", "safety_passed": True}])
    append_rows(con, "hypothesis_checks", [{"hypothesis_id": hyp, "memo_id": m, "status": "held"}
                                           for m in ("early", "other")])
    found = {f.check: f for f in run_checks(con) if f.count}
    assert found["hypothesis_check_invalid"].count == 2


def test_10_past_memos_ignores_checks_from_other_versions(con):
    a = memo(con, "2026-08-31", 3, version="A",
             hypotheses=[{"statement": "s", "category": "event_risk", "subgroup": {}}])
    (hyp,) = a["hypothesis_ids"]
    append_rows(con, "memos", [{"memo_id": "b1", "experiment_version": "B", "cycle_date": "2026-09-14",
                                "created_at": "2026-09-14T00:00:00+00:00", "risk_level": 3, "memo_text": "m",
                                "safety_passed": True}])
    append_rows(con, "hypothesis_checks", [{"hypothesis_id": hyp, "memo_id": "b1", "status": "failed"}])
    (h,) = CalibrationTools(con, "2026-09-28", "A").past_memos()["hypotheses"]
    assert h["status"] == "pending"


def test_calendar_fixture_sanity():
    assert CAL[CAL.index(date(2026, 9, 4)) + 1] == date(2026, 9, 8)
    assert date(2026, 9, 4) + timedelta(days=3) not in CAL


# ---- 12. a social fetch that raises is still recorded ----------------------------------

def test_12_social_call_that_raises_is_recorded():
    from shadow.data import evidence

    def fetch_reddit_posts(ticker, **kwargs):
        raise TimeoutError("read timed out")

    wrapped = evidence._wrap_social("reddit", fetch_reddit_posts)
    with evidence.capture() as rec, pytest.raises(TimeoutError):
        wrapped("NVDA", start_date="2026-08-25", end_date="2026-09-01")
    (call,) = rec.calls
    assert (call["vendor"], call["status"], call["retryable"]) == ("reddit", "error", True)
    assert "TimeoutError: read timed out" in call["detail"]
    assert call["seconds"] is not None and call["output"] is None


# ---- 13. install() is safe to call from several threads ---------------------------------

HOOKED = [("tradingagents.agents.tools", "route_to_vendor"),
          ("tradingagents.agents.analysts.sentiment_analyst", "fetch_stocktwits_messages"),
          ("tradingagents.agents.analysts.sentiment_analyst", "fetch_reddit_posts"),
          ("tradingagents.agents.analysts.sentiment_analyst", "render_sentiment_report"),
          ("tradingagents.agents.managers.research_manager", "render_research_plan"),
          ("tradingagents.agents.trader.trader", "render_trader_proposal")]


def test_13_concurrent_install_wraps_each_hook_once(monkeypatch):
    import importlib
    import logging
    import threading
    import time

    from shadow.data import evidence

    evidence.install()
    for mod_name, attr in HOOKED:  # start from the unwrapped functions; restored after the test
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr)
        while hasattr(fn, "__shadow_wrapped__"):
            fn = fn.__shadow_wrapped__
        monkeypatch.setattr(mod, attr, fn)
    logger = logging.getLogger("tradingagents")
    monkeypatch.setattr(logger, "handlers", [h for h in logger.handlers
                                             if not isinstance(h, evidence._CallLogHandler)])
    monkeypatch.setattr(evidence, "_installed", False)
    real_table = evidence._wrap_vendor_table
    monkeypatch.setattr(evidence, "_wrap_vendor_table", lambda: (time.sleep(0.05), real_table()))

    barrier = threading.Barrier(8)
    threads = [threading.Thread(target=lambda: (barrier.wait(), evidence.install())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for mod_name, attr in HOOKED:
        fn = getattr(importlib.import_module(mod_name), attr)
        assert hasattr(fn, "__shadow_wrapped__"), attr
        assert not hasattr(fn.__shadow_wrapped__, "__shadow_wrapped__"), f"{attr} wrapped twice"
    assert sum(isinstance(h, evidence._CallLogHandler) for h in logger.handlers) == 1


# ---- 14. only a real HTTP 403 means blocked --------------------------------------------

def test_14_a_403_price_in_the_text_is_not_blocked():
    from shadow.data import evidence

    def fetch_stocktwits_messages(ticker, **kwargs):
        return "<stocktwits unavailable: last price $403.25, feed timed out>"

    with evidence.capture() as rec:
        evidence._wrap_social("stocktwits", fetch_stocktwits_messages)("NVDA")
    assert rec.calls[0]["status"] == "unavailable"


@pytest.mark.parametrize("log_line", [
    "StockTwits fetch failed for NVDA: HTTP Error 403: Forbidden",
    "Vendor failed: 403 Client Error: Forbidden for url: https://x",
])
def test_14_a_real_403_is_blocked(log_line):
    import logging

    from shadow.data import evidence

    evidence.install()

    def fetch_stocktwits_messages(ticker, **kwargs):
        logging.getLogger("tradingagents.dataflows.vendors.stocktwits").warning(log_line)
        return "<stocktwits unavailable: HTTPError>"

    with evidence.capture() as rec:
        evidence._wrap_social("stocktwits", fetch_stocktwits_messages)("NVDA")
    assert rec.calls[0]["status"] == "blocked"


def test_14_a_403_inside_logged_prices_is_not_blocked():
    import logging

    from shadow.data import evidence

    evidence.install()

    def fetch_stocktwits_messages(ticker, **kwargs):
        logging.getLogger("tradingagents.dataflows.vendors.stocktwits").warning(
            "StockTwits returned stale quote 403.25 for %s", ticker)
        return "<stocktwits unavailable: stale>"

    with evidence.capture() as rec:
        evidence._wrap_social("stocktwits", fetch_stocktwits_messages)("NVDA")
    assert rec.calls[0]["status"] == "unavailable"


# ---- 16. the Alpaca vendor refuses items after the analysis date itself --------------------

def test_16_alpaca_vendor_clamps_to_the_analysis_date(tmp_path, monkeypatch):
    from shadow.data import alpaca_news

    store = AlpacaNewsStore(tmp_path)
    days = [date(2026, 8, 25) + timedelta(days=i) for i in range(12)]
    store._append("NVDA", [_article(i, f"{d}T15:00:00Z") for i, d in enumerate(days)])
    store._append("_global", [_article(100 + i, f"{d}T15:00:00Z") for i, d in enumerate(days)])
    for key in ("NVDA", "_global"):
        store._mark_covered(key, None, days)
    monkeypatch.setattr(alpaca_news, "default_store", lambda: store)

    with alpaca_news.analysis_date("2026-08-28"):
        text = alpaca_news.get_news("NVDA", "2026-08-25", "2026-09-05")   # asks past the analysis date
        glob = alpaca_news.get_global_news("2026-09-05")
        later = alpaca_news.get_news("NVDA", "2026-09-01", "2026-09-05")  # wholly after it
    for t in (text, glob):
        published = [line.split("Published: ")[1][:10] for line in t.splitlines() if line.startswith("Published: ")]
        assert published and max(published) <= "2026-08-28"
    assert "to 2026-08-28" in text and "to 2026-08-28" in glob
    assert later.startswith("No news found")
    # Outside an analysis-date scope the vendor serves what it is asked for, as before.
    assert "2026-09-05" in alpaca_news.get_news("NVDA", "2026-08-25", "2026-09-05")


# ---- review's suggested checks ---------------------------------------------------------

def test_check_outcome_dates_follow_calendar(con):
    add_run(con, "r1", "2026-09-04")
    (d,) = decide(con, "r1")
    append_rows(con, "outcomes", [{"decision_id": d, "entry_date": "2026-09-07", "resolved_on": "2026-09-18",
                                   "stock_return": 0.0, "spy_return": 0.0, "score": 0.0, "label": "noise"}])
    found = {f.check: f for f in run_checks(con, calendar=CAL) if f.count}
    assert found["outcome_dates_off_calendar"].examples == [d]
    skipped = {f.check: f for f in run_checks(con)}["outcome_dates_off_calendar"]
    assert skipped.severity == "skipped"


def test_check_conditions_date_matches_run(con):
    add_run(con, "r1", "2026-09-04")
    append_rows(con, "conditions", [{"run_id": "r1", "computed_at": "2026-10-22T10:00:00+00:00",
                                     "as_of": "2026-09-05", "input_status": {"prices": "ok"}}])
    assert {f.check: f for f in run_checks(con) if f.count}["conditions_date_mismatch"].examples == ["r1"]
