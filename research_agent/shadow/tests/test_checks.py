"""Data checks: clean fake data passes; each injected problem is caught by the right check."""

import hashlib

import pytest

from shadow.data.checks import run_checks
from shadow.data.db import init_db
from shadow.data.loader import append_rows
from shadow.tests import fake_db


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "c.duckdb")
    fake_db.build(con)
    yield con
    con.close()


def flagged(con, warehouse=None):
    return {f.check: f for f in run_checks(con, warehouse) if f.count}


def add_run(con, run_id="r9", day="2026-09-11"):
    append_rows(con, "runs", [{"run_id": run_id, "ticker": "MSFT", "trade_date": day,
                               "started_at": "2026-10-22T10:00:00+00:00", "cache_version": 1, "complete": True}])


def add_decision(con, decision_id="D9", run_id="r9", day="2026-09-11", **over):
    row = {"decision_id": decision_id, "run_id": run_id, "experiment_version": "llm_stats", "ticker": "MSFT",
           "trade_date": day, "decided_at": f"{day}T20:00:00+00:00", "part": "approved_part",
           "verdict": "approved", "proposed_action": "Buy", "direction": 1, "proposed_size": 1.0,
           "final_size": 1.0, "part_size": 1.0, **over}
    append_rows(con, "decisions", [row])


def test_clean_fake_data_has_no_errors(con, tmp_path):
    errors = {k for k, f in flagged(con, tmp_path).items() if f.severity == "error"}
    assert errors == set()


def test_news_created_at_midnight_after_trade_date_is_lookahead(con):
    append_rows(con, "text_items", [{
        "item_id": "alpaca_news:99", "source": "alpaca_news", "source_id": "99",
        "created_at": "2026-09-03T00:00:00+00:00", "first_seen_at": "2026-10-22T00:00:00+00:00", "title": "late"}])
    append_rows(con, "run_text_items", [{"run_id": "r2", "call_id": "c_r2", "item_id": "alpaca_news:99", "rank": 2}])
    f = flagged(con)["news_after_trade_date"]
    assert f.count == 1 and f.examples == ["r2 alpaca_news:99"]


def test_data_call_with_future_date_argument(con):
    append_rows(con, "evidence_calls", [{
        "call_id": "c_bad", "run_id": "r1", "kind": "router", "method": "get_news",
        "args": {"args": ["NVDA", "2026-08-25", "2026-09-05"], "kwargs": {}}, "as_of": "2026-09-01",
        "fetched_at": "2026-10-22T10:00:00+00:00", "status": "ok", "retryable": False}])
    assert flagged(con)["data_call_asked_for_future"].examples == ["c_bad"]


def test_payload_altered_or_missing(con, tmp_path):
    raw = tmp_path / "raw" / "payloads" / "ab"
    raw.mkdir(parents=True)
    (raw / "good.txt").write_text("table", encoding="utf-8")
    good = hashlib.sha256(b"table").hexdigest()
    append_rows(con, "evidence_calls", [
        {"call_id": f"c_{n}", "run_id": "r1", "kind": "router", "method": "get_stock_data", "as_of": "2026-09-01",
         "fetched_at": "2026-10-22T10:00:00+00:00", "status": "ok", "retryable": False,
         "payload_path": path, "payload_sha256": sha}
        for n, path, sha in (("ok", "payloads/ab/good.txt", good),
                             ("tampered", "payloads/ab/good.txt", "0" * 64),
                             ("gone", "payloads/ab/missing.txt", good))])
    f = flagged(con, tmp_path)["payload_missing_or_altered"]
    assert f.count == 2


def test_label_and_score_must_follow_the_rule(con):
    add_run(con)
    add_decision(con)
    append_rows(con, "outcomes", [{"decision_id": "D9", "entry_date": "2026-09-14", "resolved_on": "2026-09-25",
                                   "stock_return": 0.05, "spy_return": 0.01, "score": 0.02, "label": "noise"}])
    found = flagged(con)
    assert found["outcome_score_mismatch"].examples == ["D9"]   # 0.02 != 0.05 - 0.01
    assert found["outcome_label_mismatch"].examples == ["D9"]   # +4% approved is correct_approval


def test_reduced_trade_parts_must_add_up(con):
    add_run(con)
    add_decision(con, "D9a", verdict="reduced", final_size=0.5, part_size=0.7)
    add_decision(con, "D9r", verdict="reduced", final_size=0.5, part_size=0.5, part="rejected_part")
    assert flagged(con)["part_sizes_do_not_add_up"].count == 1


def test_outcome_on_trade_date(con):
    add_run(con)
    add_decision(con)
    append_rows(con, "outcomes", [{"decision_id": "D9", "entry_date": "2026-09-11", "resolved_on": "2026-09-11",
                                   "stock_return": 0.0, "spy_return": 0.0, "score": 0.0, "label": "noise"}])
    assert flagged(con)["outcome_not_after_trade"].examples == ["D9"]


def test_stop_on_wrong_side_and_risk_jump_and_drawdown(con):
    add_run(con)
    append_rows(con, "trader_proposals", [{"run_id": "r9", "ticker": "MSFT", "trade_date": "2026-09-11",
                                           "action": "Buy", "direction": 1, "entry_price": 100.0,
                                           "stop_loss": 105.0, "structured_ok": True}])
    append_rows(con, "memos", [{"memo_id": "M9", "experiment_version": "llm_stats", "cycle_date": "2026-09-28",
                                "created_at": "2026-09-28T20:00:00+00:00", "risk_level": 5,
                                "previous_risk_level": 3, "memo_text": "x", "safety_passed": False}])
    append_rows(con, "book_snapshots", [{"snapshot_date": "2026-09-29", "book": "shadow",
                                         "experiment_version": "llm_stats", "cash": 0.0, "value": 1.0,
                                         "drawdown": 0.1}])
    found = flagged(con)
    assert found["stop_on_wrong_side"].examples == ["r9"]
    assert found["risk_level_jump"].examples == ["M9"]
    assert found["drawdown_above_zero"].count == 1


def test_decision_using_a_future_memo(con):
    add_run(con)
    add_decision(con, memo_id="M2")  # M2's cycle date is 10-26, after the 09-11 trade
    assert flagged(con)["decision_memo_from_future"].examples == ["D9"]
