import pytest

from shadow.data.calibration_models import CATEGORIES
from shadow.data.db import init_db
from shadow.data.risk_memo import current_memo
from shadow.data.writers import insert_memo


@pytest.fixture
def con(tmp_path):
    c = init_db(tmp_path / "risk-memo.duckdb")
    yield c
    c.close()


def add_memo(con, cycle_date, version="feedback_A", level=3):
    advice = {c: {"advice": "keep", "confidence": "low", "reasoning": "Thin evidence."} for c in CATEGORIES}
    return insert_memo(con, experiment_version=version, cycle_date=cycle_date, risk_level=level,
                       memo_text=f"Memo {version} {cycle_date}", advice=advice, safety_passed=True)["memo_id"]


def test_none_before_first_memo(con):
    add_memo(con, "2025-03-31")
    assert current_memo(con, "feedback_A", "2025-03-28") is None


def test_applies_from_cycle_date(con):
    m1 = add_memo(con, "2025-03-31")
    assert current_memo(con, "feedback_A", "2025-03-31")["memo_id"] == m1


def test_latest_memo_wins_and_never_future(con):
    m1 = add_memo(con, "2025-03-31")
    m2 = add_memo(con, "2025-04-28", level=2)
    assert current_memo(con, "feedback_A", "2025-04-25")["memo_id"] == m1
    m = current_memo(con, "feedback_A", "2025-05-05")
    assert (m["memo_id"], m["risk_level"], m["memo_text"]) == (m2, 2, "Memo feedback_A 2025-04-28")


def test_failed_cycle_carries_previous_forward(con):
    m1 = add_memo(con, "2025-03-31")  # 04-28 cycle failed: no memo row
    assert current_memo(con, "feedback_A", "2025-05-05")["memo_id"] == m1


def test_other_setup_memo_ignored(con):
    add_memo(con, "2025-03-31", version="rejection_only_A")
    assert current_memo(con, "feedback_A", "2025-04-01") is None
