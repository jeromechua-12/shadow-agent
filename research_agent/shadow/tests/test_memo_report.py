import pytest

from shadow.data.db import init_db
from shadow.data.memo_report import fetch_memo, render
from shadow.data.writers import insert_memo


@pytest.fixture
def con(tmp_path):
    connection = init_db(tmp_path / "memo-report.duckdb")
    yield connection
    connection.close()

CATEGORIES = ("volatility", "event_risk", "valuation", "liquidity", "concentration")


def add_memo(con, cycle_date, advice="keep"):
    output = insert_memo(
        con,
        experiment_version="stats_dev",
        cycle_date=cycle_date,
        risk_level=3,
        risk_level_rationale="Balanced starting level.",
        memo_text=f"Memo for {cycle_date}",
        safety_passed=True,
        advice={
            category: {
                "advice": advice,
                "confidence": "medium",
                "reasoning": f"Evidence supports {advice}.",
                "conditions": ["sample remains stable"],
                "supporting_ids": [f"dec_{category}"],
                "contradicting_ids": [],
                "watch_next": ["next cycle sample"],
            }
            for category in CATEGORIES
        },
    )
    return output["memo_id"]


def test_fetch_by_id_includes_structured_advice(con):
    memo_id = add_memo(con, "2026-10-22", advice="tighten")

    report = fetch_memo(con, memo_id=memo_id)

    assert report["memo"]["memo_id"] == memo_id
    assert len(report["advice"]) == 5
    event = next(item for item in report["advice"] if item["category"] == "event_risk")
    assert event["advice"] == "tighten"
    assert event["conditions"] == ["sample remains stable"]
    assert event["supporting_ids"] == ["dec_event_risk"]
    assert "Memo for 2026-10-22" in render(report)


def test_fetch_latest_or_specific_cycle_for_version(con):
    older = add_memo(con, "2026-09-28")
    newer = add_memo(con, "2026-10-22", advice="tighten")

    assert fetch_memo(con, experiment_version="stats_dev")["memo"]["memo_id"] == newer
    assert fetch_memo(con, experiment_version="stats_dev", cycle_date="2026-09-28")["memo"]["memo_id"] == older
    assert fetch_memo(con, memo_id="missing") is None
