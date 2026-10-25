"""Evaluation: temporary DuckDB and fake SPY prices; nothing touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shadow.data import evaluation
from shadow.data.db import init_db
from shadow.data.loader import append_rows

DAYS = pd.bdate_range("2026-09-07", periods=40)   # Mon 2026-09-07, 8 full weeks

RUNS = {"rA": ("NVDA", "2026-09-08"), "rB": ("AAPL", "2026-09-08"),
        "rC": ("MSFT", "2026-09-09"), "rD": ("JPM", "2026-09-09")}

# decision_id, run, version, verdict, part, part_size, score, label
TRADES = [
    ("O1", "rA", "original", "approved", "approved_part", 1.0, 0.03, "correct_approval"),
    ("O2", "rB", "original", "rejected", "rejected_part", 1.0, -0.04, "correct_rejection"),
    ("O4", "rD", "original", "approved", "approved_part", 1.0, 0.005, "noise"),
    ("F1a", "rA", "feedback", "reduced", "approved_part", 0.5, 0.03, "correct_approval"),
    ("F1r", "rA", "feedback", "reduced", "rejected_part", 0.5, 0.03, "missed_gain"),
    ("F2", "rB", "feedback", "approved", "approved_part", 1.0, -0.04, "approval_loss"),
    ("F3", "rC", "feedback", "rejected", "rejected_part", 1.0, -0.05, "correct_rejection"),  # not shared
    ("F4", "rD", "feedback", "rejected", "rejected_part", 1.0, 0.005, "noise"),
]


def _spy(ticker, as_of):
    closes = 400 * np.cumprod(1 + np.random.default_rng(1).normal(0.001, 0.01, len(DAYS)))
    return pd.DataFrame({"Date": DAYS.strftime("%Y-%m-%d"), "Open": closes / 1.002, "Close": closes})


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "eval.duckdb")
    append_rows(con, "runs", [
        {"run_id": r, "ticker": t, "trade_date": d, "started_at": "2026-10-22T10:00:00+00:00",
         "cache_version": 1, "complete": True} for r, (t, d) in RUNS.items()])
    append_rows(con, "decisions", [
        {"decision_id": i, "run_id": r, "experiment_version": v, "ticker": RUNS[r][0], "trade_date": RUNS[r][1],
         "decided_at": f"{RUNS[r][1]}T20:00:00+00:00", "part": part, "verdict": verdict,
         "proposed_action": "Buy", "direction": 1, "proposed_size": 1.0, "final_size": 1.0, "part_size": size,
         "pm_rating": "Buy"} for i, r, v, verdict, part, size, _, _ in TRADES])
    append_rows(con, "outcomes", [
        {"decision_id": i, "entry_date": RUNS[r][1], "resolved_on": "2026-09-25", "stock_return": score,
         "spy_return": 0.0, "score": score, "label": label} for i, r, *_, score, label in TRADES])
    rng = np.random.default_rng(0)
    for v, drift in (("original", 0.0005), ("feedback", 0.002)):
        values = 100_000 * np.cumprod(1 + rng.normal(drift, 0.01, len(DAYS)))
        values[0] = 100_000
        append_rows(con, "book_snapshots", [
            {"snapshot_date": d.date().isoformat(), "book": book, "experiment_version": v, "cash": 0.0,
             "value": float(x), "drawdown": 0.0, "positions": {}}
            for d, x in zip(DAYS, values) for book in ("decision", "shadow")])
    return con


def test_weekly_returns_use_friday_closes_and_first_value():
    values = pd.Series([100.0, 105.0, 110.0, 99.0], index=pd.to_datetime(
        ["2026-09-07", "2026-09-11", "2026-09-17", "2026-09-18"]))   # Mon, Fri, Thu, Fri
    r = evaluation.weekly_returns(values)
    assert list(r.round(6)) == [0.05, round(99 / 105 - 1, 6)]


def test_weekly_returns_short_holiday_week_ends_on_its_last_day():
    values = pd.Series([100.0, 102.0, 104.0], index=pd.to_datetime(["2026-11-20", "2026-11-25", "2026-11-26"]))
    r = evaluation.weekly_returns(values)
    assert list(r.round(6)) == [0.0, 0.04]


def test_sharpe_is_annualised_and_none_when_flat():
    r = np.array([0.01, 0.03, -0.01, 0.01])
    assert evaluation.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(52))
    assert evaluation.sharpe(np.array([0.01, 0.01, 0.01])) is None
    assert evaluation.sharpe(np.array([0.01])) is None


def test_sharpe_ci_brackets_the_estimate_and_is_reproducible():
    r = np.random.default_rng(3).normal(0.004, 0.02, 26)
    lo, hi = evaluation.sharpe_ci(r, n_boot=2000)
    assert lo < evaluation.sharpe(r) < hi
    assert evaluation.sharpe_ci(r, n_boot=2000) == (lo, hi)
    assert evaluation.sharpe_ci(np.array([0.01] * 10), n_boot=200) is None


def test_spy_starts_in_cash_and_buys_at_next_open_with_cost():
    days = pd.DatetimeIndex(DAYS[:3])
    v = evaluation.spy_values(days, _spy)
    prices = _spy("SPY", "").set_index(pd.to_datetime(_spy("SPY", "")["Date"]))
    shares = 100_000 * 0.999 / prices.loc[days[1], "Open"]
    assert v.iloc[0] == 100_000
    assert list(v.iloc[1:]) == pytest.approx(list(shares * prices.loc[days[1:], "Close"]))


def test_spy_missing_price_raises():
    later = pd.DatetimeIndex([DAYS[-1], DAYS[-1] + pd.Timedelta(days=3)])
    with pytest.raises(ValueError, match="missing SPY price"):
        evaluation.spy_values(later, _spy)


def test_wilcoxon_reports_none_when_all_differences_are_zero():
    a = pd.Series([0.01, 0.02, -0.01])
    assert evaluation.wilcoxon_test(a, a)["p_value"] is None
    assert evaluation.wilcoxon_test(a + 0.01, a)["median_diff"] == pytest.approx(0.01)


def test_decision_quality_on_shared_stock_dates_without_noise(con):
    orig = evaluation.decision_quality(con, "original")
    assert orig == {"rejections_scored": 1, "rejection_precision": 1.0, "net_rejection_value": pytest.approx(0.04),
                    "approvals_scored": 1, "approval_loss_rate": 0.0}
    fb = evaluation.decision_quality(con, "feedback")     # F3 (MSFT, not shared) and F4 (noise) excluded
    assert fb == {"rejections_scored": 1, "rejection_precision": 0.0, "net_rejection_value": pytest.approx(-0.015),
                  "approvals_scored": 2, "approval_loss_rate": 0.5}


def test_evaluate_scores_all_three_setups(con):
    report = evaluation.evaluate(con, price_provider=_spy, n_boot=500)
    assert report["period"] == ["2026-09-07", DAYS[-1].date().isoformat()]
    assert set(report["setups"]) == {"original", "feedback", evaluation.SPY}
    for s in report["setups"].values():
        assert s["weeks"] == 8 and s["sharpe"] is not None and s["max_drawdown"] <= 0
    assert all(w["weeks"] == 8 and w["p_value"] is not None for w in report["wilcoxon"].values())
    text = evaluation.render(report)
    assert "spy_buy_and_hold" in text and "rejection_precision" in text


def test_evaluate_needs_shared_snapshots(tmp_path):
    with pytest.raises(ValueError, match="shared snapshot dates"):
        evaluation.evaluate(init_db(tmp_path / "empty.duckdb"), price_provider=_spy)
