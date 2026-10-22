"""Schema: creates cleanly, is idempotent, and its constraints reject bad rows."""

import duckdb
import pytest

from shadow.data.db import init_db

TABLES = {
    "schema_info", "runs", "llm_calls", "evidence_calls", "text_items", "run_text_items",
    "agent_outputs", "trader_proposals", "conditions", "decisions", "outcomes",
    "book_snapshots", "memos", "memo_advice", "memo_hypotheses", "hypothesis_checks",
    "memo_tool_calls",
}


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "t.duckdb")
    yield con
    con.close()


def add_run(con, run_id="r1", ticker="NVDA", day="2026-09-01", started="2026-10-22 10:00:00+00", complete=True):
    con.execute("INSERT INTO runs (run_id, ticker, trade_date, started_at, cache_version, complete) "
                "VALUES (?, ?, ?, ?, 1, ?)", [run_id, ticker, day, started, complete])


def test_creates_all_tables_and_is_idempotent(tmp_path):
    path = tmp_path / "t.duckdb"
    init_db(path).close()
    con = init_db(path)  # second run must not fail or duplicate
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                       "WHERE table_type = 'BASE TABLE'").fetchall()}
    assert names == TABLES
    assert con.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchall() == [("1",)]


def test_status_check_rejects_unknown_status(con):
    add_run(con)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO evidence_calls (call_id, run_id, kind, method, as_of, fetched_at, status, retryable) "
                    "VALUES ('c1', 'r1', 'router', 'get_news', '2026-09-01', now(), 'kinda_ok', false)")


@pytest.mark.parametrize("column, value", [("risk_volatility", 4), ("main_reason", "vibes"), ("part", "half")])
def test_decision_checks(con, column, value):
    add_run(con)
    row = {"decision_id": "d1", "run_id": "r1", "experiment_version": "v", "ticker": "NVDA",
           "trade_date": "2026-09-01", "decided_at": "2026-09-01 20:00:00+00", "part": "approved_part",
           "verdict": "approved", "proposed_action": "Buy", "direction": 1, "proposed_size": 1.0,
           "final_size": 1.0, "part_size": 1.0, column: value}
    cols = ", ".join(row)
    with pytest.raises(duckdb.ConstraintException):
        con.execute(f"INSERT INTO decisions ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def test_risk_level_limits(con):
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO memos (memo_id, experiment_version, cycle_date, created_at, risk_level, "
                    "memo_text, safety_passed) VALUES ('m1', 'v', '2026-09-28', now(), 6, 'x', true)")


def test_foreign_key_requires_run(con):
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO trader_proposals (run_id, ticker, trade_date, structured_ok) "
                    "VALUES ('missing', 'NVDA', '2026-09-01', false)")


def test_current_runs_picks_latest_complete(con):
    add_run(con, "old", started="2026-10-20 10:00:00+00")
    add_run(con, "new", started="2026-10-21 10:00:00+00")
    add_run(con, "newest_incomplete", started="2026-10-22 10:00:00+00", complete=False)
    assert con.execute("SELECT run_id FROM current_runs").fetchall() == [("new",)]


def test_hypothesis_status_is_latest_check(con):
    for memo, day in (("m1", "2026-09-28"), ("m2", "2026-10-26")):
        con.execute("INSERT INTO memos (memo_id, experiment_version, cycle_date, created_at, risk_level, "
                    "memo_text, safety_passed) VALUES (?, 'v', ?, now(), 3, 'x', true)", [memo, day])
    con.execute("INSERT INTO memo_hypotheses VALUES ('h1', 'm1', 'event_risk', 'macro news rejections miss gains', "
                "'{\"category\": \"event_risk\"}')")
    con.execute("INSERT INTO hypothesis_checks VALUES ('h1', 'm1', 'pending', NULL), ('h1', 'm2', 'held', NULL)")
    assert con.execute("SELECT status, checked_in_memo FROM hypothesis_status").fetchall() == [("held", "m2")]
