"""The real Outcome judge: offline tests with an injected fake price provider.

No network, no yfinance: every test supplies its own price_provider callable,
so this exercises the real scoring logic (which decisions are due, entry/exit
dates, score and label via insert_outcome) without ever touching the network.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from shadow.data.db import init_db
from shadow.data.loader import append_rows
from shadow.data.outcome_judge import score_pending_decisions
from shadow.data.writers import insert_decision, outcome_dates

ZERO_RATINGS = {"volatility": 0, "event_risk": 0, "valuation": 0, "liquidity": 0, "concentration": 0}
TODAY = date(2026, 10, 22)


def _business_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


CALENDAR = _business_days(date(2026, 1, 1), TODAY)


def _insert_run(con, run_id: str, ticker: str, trade_date: date, action: str = "Buy") -> None:
    append_rows(con, "runs", [{"run_id": run_id, "ticker": ticker, "trade_date": trade_date,
                               "started_at": f"{trade_date}T10:00:00+00:00", "cache_version": 1, "complete": True}])
    append_rows(con, "trader_proposals", [{"run_id": run_id, "ticker": ticker, "trade_date": trade_date,
                                           "action": action, "direction": 1 if action == "Buy" else -1,
                                           "structured_ok": True}])


def _approve(con, run_id: str, version: str = "v1") -> str:
    (decision_id,) = insert_decision(con, run_id=run_id, experiment_version=version, verdict="approved",
                                     proposed_size=1.0, final_size=1.0, risk_ratings=ZERO_RATINGS, main_reason="none")
    return decision_id


def _reject(con, run_id: str, version: str = "v1") -> str:
    (decision_id,) = insert_decision(con, run_id=run_id, experiment_version=version, verdict="rejected",
                                     proposed_size=1.0, final_size=0.0, risk_ratings=ZERO_RATINGS,
                                     main_reason="volatility")
    return decision_id


def _price_df(values: dict[date, tuple]) -> pd.DataFrame:
    """values: {date: (open, close)}, either may be None (unused for that row)."""
    return pd.DataFrame([{"Date": d, "Open": o, "Close": c} for d, (o, c) in values.items()])


def _fake_provider(series: dict[str, pd.DataFrame]):
    def provider(ticker: str, as_of: str) -> pd.DataFrame:
        return series[ticker]
    return provider


@pytest.fixture
def con(tmp_path):
    return init_db(tmp_path / "db.duckdb")


def test_scores_a_resolved_approved_buy(con):
    _insert_run(con, "r1", "NVDA", date(2026, 1, 5), "Buy")
    decision_id = _approve(con, "r1")
    entry, resolved = outcome_dates(date(2026, 1, 5), CALENDAR)
    assert resolved <= TODAY  # sanity: must actually be resolvable by TODAY

    prices = {"NVDA": _price_df({entry: (100.0, None), resolved: (None, 110.0)}),
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 404.0)})}
    report = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=_fake_provider(prices))

    assert (report.scored, report.still_pending, report.errors, report.skipped_no_price) == (1, 0, [], [])
    stock_return, spy_return = 110.0 / 100.0 - 1, 404.0 / 400.0 - 1
    row = con.execute("SELECT stock_return, spy_return, score, label, resolved_on FROM outcomes "
                      "WHERE decision_id = ?", [decision_id]).fetchone()
    assert row[0] == pytest.approx(stock_return) and row[1] == pytest.approx(spy_return)
    assert row[2] == pytest.approx(stock_return - spy_return)  # direction +1 (Buy)
    assert row[3] == "correct_approval"  # approved_part, positive score
    assert row[4] == resolved


def test_scores_a_rejected_sell_as_missed_gain(con):
    # Sell rejected, but the stock rose more than SPY: direction -1 * positive gap < 0,
    # i.e. the stock went up (a Sell would have lost), so rejecting it was... let's
    # pick numbers that land on missed_gain: rejected_part succeeds when score < 0.
    _insert_run(con, "r2", "AAPL", date(2026, 1, 6), "Sell")
    decision_id = _reject(con, "r2")
    entry, resolved = outcome_dates(date(2026, 1, 6), CALENDAR)
    prices = {"AAPL": _price_df({entry: (100.0, None), resolved: (None, 90.0)}),  # stock fell 10%
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 400.0)})}  # SPY flat
    score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=_fake_provider(prices))
    label = con.execute("SELECT label FROM outcomes WHERE decision_id = ?", [decision_id]).fetchone()[0]
    # direction -1, stock_return -0.10, spy_return 0: score = -1 * (-0.10 - 0) = +0.10 > 0 -> missed_gain
    assert label == "missed_gain"


def test_noise_band(con):
    _insert_run(con, "r3", "JPM", date(2026, 1, 7), "Buy")
    decision_id = _approve(con, "r3")
    entry, resolved = outcome_dates(date(2026, 1, 7), CALENDAR)
    prices = {"JPM": _price_df({entry: (100.0, None), resolved: (None, 100.3)}),
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 400.0)})}
    score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=_fake_provider(prices))
    label = con.execute("SELECT label FROM outcomes WHERE decision_id = ?", [decision_id]).fetchone()[0]
    assert label == "noise"


def test_unelapsed_horizon_is_left_pending_not_guessed(con):
    # Trade date within the last ~2 weeks: the 10-trading-day horizon has not elapsed by TODAY.
    _insert_run(con, "r4", "NVDA", TODAY - timedelta(days=3), "Buy")
    _approve(con, "r4")
    report = score_pending_decisions(con, calendar=CALENDAR, today=TODAY,
                                     price_provider=_fake_provider({}))  # must never be called
    assert (report.scored, report.still_pending) == (0, 1)
    assert con.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 0


def test_missing_price_row_is_skipped_not_crashed(con):
    _insert_run(con, "r5", "GME", date(2026, 1, 8), "Buy")
    decision_id = _approve(con, "r5")
    entry, resolved = outcome_dates(date(2026, 1, 8), CALENDAR)
    # GME has no row at all for the exit date (e.g. a data gap); SPY is fine.
    prices = {"GME": _price_df({entry: (100.0, None)}),  # missing the resolved-date row
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 404.0)})}
    report = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=_fake_provider(prices))
    assert report.scored == 0 and len(report.skipped_no_price) == 1
    assert decision_id in report.skipped_no_price[0]
    assert con.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 0


def test_empty_price_cell_is_skipped_not_scored(con):
    _insert_run(con, "r9", "TSLA", date(2026, 1, 9), "Buy")
    decision_id = _approve(con, "r9")
    entry, resolved = outcome_dates(date(2026, 1, 9), CALENDAR)
    # The entry row exists but its open was never reported (fill_gaps=False leaves it NaN).
    prices = {"TSLA": _price_df({entry: (float("nan"), 101.0), resolved: (float("nan"), 110.0)}),
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 404.0)})}
    report = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=_fake_provider(prices))
    assert (report.scored, report.errors) == (0, []) and len(report.skipped_no_price) == 1
    assert decision_id in report.skipped_no_price[0]
    assert con.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 0


def test_dry_run_writes_nothing(con):
    _insert_run(con, "r6", "NVDA", date(2026, 1, 5), "Buy")
    _approve(con, "r6")
    entry, resolved = outcome_dates(date(2026, 1, 5), CALENDAR)
    prices = {"NVDA": _price_df({entry: (100.0, None), resolved: (None, 110.0)}),
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 404.0)})}
    report = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, dry_run=True,
                                     price_provider=_fake_provider(prices))
    assert report.scored == 1
    assert con.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 0


def test_experiment_version_filter(con):
    _insert_run(con, "r7", "NVDA", date(2026, 1, 5), "Buy")
    _insert_run(con, "r8", "AAPL", date(2026, 1, 5), "Buy")
    d_v1 = _approve(con, "r7", version="v1")
    _approve(con, "r8", version="v2")
    entry, resolved = outcome_dates(date(2026, 1, 5), CALENDAR)
    prices = {"NVDA": _price_df({entry: (100.0, None), resolved: (None, 110.0)}),
              "AAPL": _price_df({entry: (100.0, None), resolved: (None, 110.0)}),
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 404.0)})}
    report = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, experiment_version="v1",
                                     price_provider=_fake_provider(prices))
    assert report.scored == 1
    assert con.execute("SELECT decision_id FROM outcomes").fetchone()[0] == d_v1


def test_already_scored_decisions_are_not_touched_again(con):
    _insert_run(con, "r9", "NVDA", date(2026, 1, 5), "Buy")
    _approve(con, "r9")
    entry, resolved = outcome_dates(date(2026, 1, 5), CALENDAR)
    prices = {"NVDA": _price_df({entry: (100.0, None), resolved: (None, 110.0)}),
              "SPY": _price_df({entry: (400.0, None), resolved: (None, 404.0)})}
    provider = _fake_provider(prices)
    first = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=provider)
    second = score_pending_decisions(con, calendar=CALENDAR, today=TODAY, price_provider=provider)
    assert (first.scored, second.scored) == (1, 0)
    assert con.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 1
