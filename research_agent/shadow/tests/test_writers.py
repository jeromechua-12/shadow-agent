"""Teammate insert helpers: valid records go in with derived fields; invalid ones list every problem."""

from datetime import date

import pandas as pd
import pytest

from shadow.data.checks import run_checks
from shadow.data.db import init_db
from shadow.data.loader import append_rows
from shadow.data.queries import CalibrationTools
from shadow.data.writers import (
    InvalidRecord,
    insert_book_snapshot,
    insert_decision,
    insert_memo,
    insert_outcome,
    label_for,
)

# Trading days around the test dates (Labor Day 2026-09-07 closed).
CAL = [d.date() for d in pd.bdate_range("2026-08-31", "2026-10-30") if d.date() != date(2026, 9, 7)]
RATINGS = {"volatility": 2, "event_risk": 1, "valuation": 1, "liquidity": 0, "concentration": 0}
ADVICE = {c: {"advice": "keep", "confidence": "low", "reasoning": "few trades", "cited_ids": []}
          for c in ("volatility", "event_risk", "valuation", "liquidity", "concentration")}


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "w.duckdb")
    for run_id, action, day in (("r_buy", "Buy", "2026-09-01"), ("r_hold", "Hold", "2026-09-02")):
        append_rows(con, "runs", [{"run_id": run_id, "ticker": "NVDA", "trade_date": day,
                                   "started_at": "2026-10-22T10:00:00+00:00", "cache_version": 1, "complete": True}])
        append_rows(con, "trader_proposals", [{"run_id": run_id, "ticker": "NVDA", "trade_date": day,
                                               "action": action, "direction": 1 if action == "Buy" else 0,
                                               "structured_ok": True}])
    yield con
    con.close()


def decision(con, **over):
    kw = {"run_id": "r_buy", "experiment_version": "llm_stats", "verdict": "reduced", "proposed_size": 1.0,
          "final_size": 0.5, "risk_ratings": RATINGS, "main_reason": "volatility", "explanation": "ATR is high"}
    return insert_decision(con, **{**kw, **over})


def test_reduced_decision_becomes_two_parts(con):
    ids = decision(con)
    rows = con.execute("SELECT part, part_size, ticker, trade_date, proposed_action, direction, risk_volatility "
                       "FROM decisions ORDER BY part").fetchall()
    assert len(ids) == 2
    assert [(r[0], r[1]) for r in rows] == [("approved_part", 0.5), ("rejected_part", 0.5)]
    assert rows[0][2:] == ("NVDA", rows[0][3], "Buy", 1, 2)


def test_decision_lists_every_problem(con):
    with pytest.raises(InvalidRecord) as e:
        decision(con, verdict="approved", risk_ratings={"volatility": 5}, main_reason="vibes")
    text = str(e.value)
    assert "does not match sizes 1.0 -> 0.5 (that is 'reduced')" in text
    assert "risk_ratings needs exactly these keys" in text
    assert "main_reason must be one of" in text
    assert con.execute("SELECT count(*) FROM decisions").fetchone() == (0,)


def test_decision_guards(con):
    with pytest.raises(InvalidRecord, match="not in runs"):
        decision(con, run_id="nope")
    with pytest.raises(InvalidRecord, match="Holds are not logged"):
        decision(con, run_id="r_hold")
    decision(con)
    with pytest.raises(InvalidRecord, match="already has a decision"):
        decision(con)


def test_outcome_computes_score_and_label(con):
    approved, rejected = decision(con)
    a = insert_outcome(con, decision_id=approved, entry_date="2026-09-02", resolved_on="2026-09-17",
                       stock_return=0.06, spy_return=0.01, calendar=CAL)
    r = insert_outcome(con, decision_id=rejected, stock_return=0.06, spy_return=0.01, calendar=CAL)
    assert (r["entry_date"], r["resolved_on"]) == ("2026-09-02", "2026-09-17")
    assert (a["label"], r["label"]) == ("correct_approval", "missed_gain")
    assert a["score"] == pytest.approx(0.05)
    assert not [f for f in run_checks(con) if f.count and f.severity == "error"]


def test_outcome_problems(con):
    (approved, _) = decision(con)
    with pytest.raises(InvalidRecord) as e:
        insert_outcome(con, decision_id=approved, entry_date="2026-09-01", resolved_on="2026-09-15",
                       stock_return="5%", spy_return=0.0, calendar=CAL)
    text = str(e.value)
    assert "must be the next trading day after 2026-09-01, i.e. 2026-09-02" in text
    assert "must be the 10th trading day after entry 2026-09-02, i.e. 2026-09-17" in text
    assert "stock_return must be a number" in text


@pytest.mark.parametrize("part, score, label", [
    ("approved_part", 0.01, "noise"), ("approved_part", -0.0101, "approval_loss"),
    ("rejected_part", 0.02, "missed_gain"), ("rejected_part", -0.02, "correct_rejection"),
])
def test_label_rule(part, score, label):
    assert label_for(part, score) == label


def test_book_snapshot_drawdown_and_order(con):
    kw = {"experiment_version": "llm_stats", "book": "decision", "cash": 0.0}
    assert insert_book_snapshot(con, snapshot_date="2026-09-01", value=100.0, **kw) == 0.0
    assert insert_book_snapshot(con, snapshot_date="2026-09-02", value=110.0, **kw) == 0.0
    assert insert_book_snapshot(con, snapshot_date="2026-09-03", value=99.0, **kw) == pytest.approx(-0.1)
    with pytest.raises(InvalidRecord, match="date order"):
        insert_book_snapshot(con, snapshot_date="2026-09-02", value=1.0, **kw)
    with pytest.raises(InvalidRecord, match="book must be"):
        insert_book_snapshot(con, snapshot_date="2026-09-04", value=1.0, **{**kw, "book": "real"})


def test_memo_with_hypothesis_and_later_check(con):
    first = insert_memo(con, experiment_version="llm_stats", cycle_date="2026-08-31", risk_level=3,
                        memo_text="Keep everything.", advice=ADVICE, safety_passed=True,
                        hypotheses=[{"statement": "macro rejections miss gains", "category": "event_risk",
                                     "subgroup": {"main_reason": "event_risk", "news_keyword": "fed"}}],
                        tool_calls=[{"tool": "cycle_summary", "args": {}, "result": {"n": 0}}])
    (hyp,) = first["hypothesis_ids"]
    insert_book_snapshot(con, experiment_version="llm_stats", book="decision", snapshot_date="2026-09-25",
                         cash=0.0, value=100.0)
    insert_memo(con, experiment_version="llm_stats", cycle_date="2026-09-28", risk_level=4, memo_text="Loosen.",
                advice=ADVICE, safety_passed=True,
                hypothesis_checks=[{"hypothesis_id": hyp, "status": "held", "evidence": {"n": 7}}])
    (h,) = CalibrationTools(con, "2026-10-01", "llm_stats").past_memos()["hypotheses"]
    assert h["status"] == "held"
    assert con.execute("SELECT previous_risk_level FROM memos WHERE risk_level = 4").fetchone() == (3,)


def test_memo_hard_limits(con):
    insert_memo(con, experiment_version="v", cycle_date="2026-08-31", risk_level=3, memo_text="x",
                advice=ADVICE, safety_passed=True)
    with pytest.raises(InvalidRecord, match="one step per cycle: 3 -> 5"):
        insert_memo(con, experiment_version="v", cycle_date="2026-09-28", risk_level=5, memo_text="x",
                    advice=ADVICE, safety_passed=True)
    for day, value in (("2026-09-01", 100.0), ("2026-09-25", 85.0)):
        insert_book_snapshot(con, experiment_version="v", book="decision", snapshot_date=day, cash=0.0, value=value)
    with pytest.raises(InvalidRecord, match="drawdown beyond 10%"):
        insert_memo(con, experiment_version="v", cycle_date="2026-09-28", risk_level=4, memo_text="x",
                    advice=ADVICE, safety_passed=True)
    # Stepping down is always allowed.
    insert_memo(con, experiment_version="v", cycle_date="2026-09-28", risk_level=2, memo_text="x",
                advice=ADVICE, safety_passed=True)


def test_memo_problems_listed(con):
    with pytest.raises(InvalidRecord) as e:
        insert_memo(con, experiment_version="v", cycle_date="soon", risk_level=0, memo_text="",
                    advice={"volatility": {"advice": "panic"}}, safety_passed=True,
                    tool_calls=[{"tool": "drop_table"}])
    assert len(e.value.problems) >= 6


def test_memo_dry_run_checks_but_writes_nothing(con):
    kw = {"experiment_version": "v", "memo_text": "x", "advice": ADVICE, "safety_passed": True}
    assert insert_memo(con, cycle_date="2026-08-31", risk_level=3, dry_run=True, **kw) == \
        {"memo_id": None, "hypothesis_ids": []}
    assert con.execute("SELECT count(*) FROM memos").fetchone() == (0,)
    with pytest.raises(InvalidRecord, match="must start at risk level 3"):
        insert_memo(con, cycle_date="2026-08-31", risk_level=4, dry_run=True, **kw)


def test_two_part_decision_is_all_or_nothing(con, monkeypatch):
    from shadow.data import writers

    monkeypatch.setattr(writers, "_new_id", lambda prefix: "dec_same")  # second part collides with the first
    with pytest.raises(Exception, match="(?i)constraint|duplicate"):
        decision(con)                                                     # reduced: two parts
    assert con.execute("SELECT count(*) FROM decisions").fetchone() == (0,)


def test_unparsed_pm_hold_is_stored_without_ratings(con):
    (d,) = decision(con, verdict="rejected", final_size=0.0, risk_ratings=None, main_reason="none",
                    pm_rating="Hold", explanation="PM output unparseable after one retry; recorded as Hold.")
    row = con.execute("SELECT part, risk_volatility, risk_event, pm_rating FROM decisions WHERE decision_id = ?",
                      [d]).fetchone()
    assert row == ("rejected_part", None, None, "Hold")
    with pytest.raises(InvalidRecord, match="risk_ratings may be omitted only"):
        decision(con, risk_ratings=None, experiment_version="v2")         # a real (reduced) decision needs ratings
