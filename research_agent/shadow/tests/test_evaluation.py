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
    if ticker == evaluation.RISK_FREE:   # 4% then 6% a year, plus a 50% day before the period
        days = [DAYS[0] - pd.Timedelta(days=3), *DAYS]
        yields = [50.0] + [4.0] * 20 + [6.0] * 20
        return pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in days], "Close": yields})
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
         "pm_rating": {"approved": "Buy", "reduced": "Overweight", "rejected": "Hold"}[verdict]}
        for i, r, v, verdict, part, size, _, _ in TRADES])
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


def test_weekly_returns_use_friday_closes_and_start_value():
    values = pd.Series([101.0, 105.0, 110.0, 99.0], index=pd.to_datetime(
        ["2026-09-07", "2026-09-11", "2026-09-17", "2026-09-18"]))   # Mon, Fri, Thu, Fri
    r = evaluation.weekly_returns(values, start=100.0)
    assert list(r.round(6)) == [0.05, round(99 / 105 - 1, 6)]


def test_weekly_returns_short_holiday_week_ends_on_its_last_day():
    values = pd.Series([100.0, 102.0, 104.0], index=pd.to_datetime(["2026-11-20", "2026-11-25", "2026-11-26"]))
    r = evaluation.weekly_returns(values, start=100.0)
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


def test_spy_buys_at_first_open_with_cost():
    days = pd.DatetimeIndex(DAYS[:3])
    v = evaluation.spy_values(days, _spy)
    prices = _spy("SPY", "").set_index(pd.to_datetime(_spy("SPY", "")["Date"]))
    shares = 100_000 * 0.999 / prices.loc[days[0], "Open"]
    assert list(v) == pytest.approx(list(shares * prices.loc[days, "Close"]))


def test_spy_missing_price_raises():
    later = pd.DatetimeIndex([DAYS[-1], DAYS[-1] + pd.Timedelta(days=3)])
    with pytest.raises(ValueError, match="missing SPY price"):
        evaluation.spy_values(later, _spy)


def test_risk_free_averages_yields_inside_the_period_only():
    rf = evaluation.risk_free_weekly(DAYS, _spy)
    assert rf == pytest.approx(1.05 ** (1 / 52) - 1)


def test_risk_free_without_yields_raises():
    later = pd.DatetimeIndex([DAYS[-1] + pd.Timedelta(days=3)])
    with pytest.raises(ValueError, match="no \\^IRX yields"):
        evaluation.risk_free_weekly(later, _spy)


def test_wilcoxon_reports_none_when_all_differences_are_zero():
    a = pd.Series([0.01, 0.02, -0.01])
    assert evaluation.wilcoxon_test(a, a)["p_value"] is None
    assert evaluation.wilcoxon_test(a + 0.01, a)["median_diff"] == pytest.approx(0.01)


def test_decision_quality_on_shared_stock_dates_without_noise(con):
    orig = evaluation.decision_quality(con, "original", _spy)
    assert orig == {"rejections_scored": 1, "rejection_precision": 1.0, "net_rejection_value": pytest.approx(0.04),
                    "approvals_scored": 1, "approval_loss_rate": 0.0}
    fb = evaluation.decision_quality(con, "feedback", _spy)     # F3 (MSFT, not shared) and F4 (noise) excluded
    assert fb == {"rejections_scored": 1, "rejection_precision": 0.0, "net_rejection_value": pytest.approx(-0.015),
                  "approvals_scored": 2, "approval_loss_rate": 0.5}


def test_net_rejection_value_counts_only_what_the_rejection_changed_in_the_book(tmp_path):
    con = init_db(tmp_path / "book.duckdb")
    day, entry = "2026-09-08", "2026-09-09"
    # ticker: (action, pm_rating, verdict, [(part, part_size)], stock return vs SPY)
    cases = {"XOM": ("Sell", "Hold", "rejected", [("rejected_part", 1.0)], 0.03),          # nothing held: 0
             "JNJ": ("Sell", "Underweight", "reduced", [("approved_part", 0.5), ("rejected_part", 0.5)], 0.02),
             "TSLA": ("Buy", "Hold", "rejected", [("rejected_part", 1.0)], -0.05)}
    append_rows(con, "runs", [{"run_id": t, "ticker": t, "trade_date": day, "started_at": "2026-10-22T10:00:00+00:00",
                               "cache_version": 1, "complete": True} for t in cases])
    decisions, outcomes = [], []
    for v in ("original", "feedback"):
        for t, (action, rating, verdict, parts, excess) in cases.items():
            direction = 1 if action == "Buy" else -1
            for part, size in parts:
                i = f"{v}_{t}_{part}"
                decisions.append({"decision_id": i, "run_id": t, "experiment_version": v, "ticker": t,
                                  "trade_date": day, "decided_at": f"{day}T20:00:00+00:00", "part": part,
                                  "verdict": verdict, "proposed_action": action, "direction": direction,
                                  "proposed_size": 1.0, "final_size": 1.0 - parts[-1][1], "part_size": size,
                                  "pm_rating": rating})
                score = direction * excess
                label = (("correct_approval" if score > 0 else "approval_loss") if part == "approved_part"
                         else ("missed_gain" if score > 0 else "correct_rejection"))
                outcomes.append({"decision_id": i, "entry_date": entry, "resolved_on": "2026-09-23",
                                 "stock_return": excess, "spy_return": 0.0, "score": score, "label": label})
        # $10,000 per ticker at a $100 entry open: 80 JNJ shares = 0.8 of a position, 50 TSLA = 0.5.
        append_rows(con, "book_snapshots", [{"snapshot_date": day, "book": "decision", "experiment_version": v,
                                             "cash": 0.0, "value": 100_000.0, "drawdown": 0.0,
                                             "positions": {"JNJ": 80.0, "TSLA": 50.0}}])
    append_rows(con, "decisions", decisions)
    append_rows(con, "outcomes", outcomes)

    prices = lambda t, as_of: pd.DataFrame({"Date": [entry], "Open": [100.0]})   # noqa: E731
    # XOM 0; JNJ kept 0.4 that approving would have sold: 0.4 x 0.02; TSLA approving adds 0.5: 0.5 x 0.05.
    assert evaluation.decision_quality(con, "original", prices)["net_rejection_value"] == pytest.approx(0.033)
    con.close()


def test_evaluate_scores_all_three_setups(con):
    report = evaluation.evaluate(con, price_provider=_spy, n_boot=500)
    assert report["period"] == ["2026-09-07", DAYS[-1].date().isoformat()]
    assert report["risk_free_annual"] == pytest.approx(0.05)
    assert set(report["setups"]) == {"original", "feedback", evaluation.SPY}
    for s in report["setups"].values():
        assert s["weeks"] == 8 and s["sharpe"] is not None and s["max_drawdown"] <= 0
    assert all(w["weeks"] == 8 and w["p_value"] is not None for w in report["wilcoxon"].values())
    text = evaluation.render(report)
    assert "spy_buy_and_hold" in text and "rejection_precision" in text


def test_evaluate_needs_shared_snapshots(tmp_path):
    with pytest.raises(ValueError, match="shared snapshot dates"):
        evaluation.evaluate(init_db(tmp_path / "empty.duckdb"), price_provider=_spy)


def test_evaluate_refuses_while_resolvable_decisions_are_unscored(con):
    append_rows(con, "decisions", [{
        "decision_id": "late", "run_id": "rA", "experiment_version": "original", "ticker": "NVDA",
        "trade_date": "2026-09-08", "decided_at": "2026-09-08T20:00:00+00:00", "part": "approved_part",
        "verdict": "approved", "proposed_action": "Buy", "direction": 1, "proposed_size": 1.0,
        "final_size": 1.0, "part_size": 1.0, "pm_rating": "Buy"}])
    with pytest.raises(ValueError, match="1 decision.*not scored.*outcome_judge"):
        evaluation.evaluate(con, price_provider=_spy, n_boot=200)


def test_evaluate_allows_decisions_not_yet_resolvable(con):
    last = DAYS[-3].date().isoformat()     # its 10 trading days run past the price history
    append_rows(con, "runs", [{"run_id": "rZ", "ticker": "PG", "trade_date": last,
                               "started_at": "2026-10-22T10:00:00+00:00", "cache_version": 1, "complete": True}])
    append_rows(con, "decisions", [{
        "decision_id": "fresh", "run_id": "rZ", "experiment_version": "original", "ticker": "PG",
        "trade_date": last, "decided_at": f"{last}T20:00:00+00:00", "part": "approved_part",
        "verdict": "approved", "proposed_action": "Buy", "direction": 1, "proposed_size": 1.0,
        "final_size": 1.0, "part_size": 1.0, "pm_rating": "Buy"}])
    assert evaluation.evaluate(con, price_provider=_spy, n_boot=200)["setups"]
