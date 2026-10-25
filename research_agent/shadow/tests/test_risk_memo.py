"""Current memo lookup: the latest memo written on or before the trading day, same version only."""

import pytest

from shadow.data.db import init_db
from shadow.data.risk_memo import current_memo
from shadow.data.writers import insert_memo

ADVICE = {c: {"advice": "keep", "confidence": "low", "reasoning": "few trades", "cited_ids": []}
          for c in ("volatility", "event_risk", "valuation", "liquidity", "concentration")}


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "m.duckdb")
    yield con
    con.close()


def add_memo(con, version, cycle_date, text):
    return insert_memo(con, experiment_version=version, cycle_date=cycle_date, risk_level=3,
                       memo_text=text, advice=ADVICE, safety_passed=True)["memo_id"]


def test_no_memos_returns_none(con):
    assert current_memo(con, "2026-09-01", "feedback") is None


def test_memo_starts_on_its_cycle_date(con):
    memo_id = add_memo(con, "feedback", "2026-08-31", "first memo")
    assert current_memo(con, "2026-08-30", "feedback") is None
    assert current_memo(con, "2026-08-31", "feedback") == {"memo_id": memo_id, "memo_text": "first memo"}


def test_memo_has_no_age_limit(con):
    memo_id = add_memo(con, "feedback", "2026-08-31", "first memo")
    assert current_memo(con, "2027-02-01", "feedback")["memo_id"] == memo_id


def test_newer_memo_replaces_older(con):
    add_memo(con, "feedback", "2026-08-31", "first memo")
    newer = add_memo(con, "feedback", "2026-09-28", "second memo")
    assert current_memo(con, "2026-09-28", "feedback")["memo_id"] == newer
    assert current_memo(con, "2026-09-27", "feedback")["memo_text"] == "first memo"


def test_other_version_is_ignored(con):
    add_memo(con, "other", "2026-08-31", "other memo")
    assert current_memo(con, "2026-09-01", "feedback") is None
