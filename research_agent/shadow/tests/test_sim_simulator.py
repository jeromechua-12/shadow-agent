"""Advancing a book over the calendar: fills, ratings, resuming, stopping, and the two books' gap.

Offline: prices come from a fake provider, so nothing touches the network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from shadow.data.db import init_db
from shadow.data.loader import append_rows
from shadow.sim.config import SimConfig
from shadow.sim.simulator import advance

VERSION = "sim_test"
CONFIG = SimConfig(starting_cash=100_000.0, universe_size=10, transaction_cost=0.0)

# Five consecutive trading days (09-07 is Labor Day).
DAYS = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 8)]

# ticker -> {day: (open, close)}. AAPL rises steadily; MSFT is flat.
PRICES = {
    "AAPL": {DAYS[0]: (100.0, 100.0), DAYS[1]: (100.0, 110.0), DAYS[2]: (110.0, 120.0),
             DAYS[3]: (120.0, 130.0), DAYS[4]: (130.0, 140.0)},
    "MSFT": {d: (50.0, 50.0) for d in DAYS},
}


def fake_prices(ticker: str, as_of: str) -> pd.DataFrame:
    rows = PRICES[ticker]
    return pd.DataFrame([{"Date": d.isoformat(), "Open": o, "Close": c} for d, (o, c) in rows.items()])


@pytest.fixture
def con(tmp_path):
    con = init_db(tmp_path / "sim.duckdb")
    yield con
    con.close()


def add_decision(con, *, decision_id, ticker, trade_date, action, pm_rating, version=VERSION):
    """One decision set as the decision logger would write it: one row per part.

    The PM keeping the Trader's action is an approval; Hold is a rejection; any
    other rating is a reduction (an approved and a rejected part).
    """
    run_id = f"run_{decision_id}"
    append_rows(con, "runs", [{"run_id": run_id, "ticker": ticker, "trade_date": trade_date,
                               "started_at": "2026-10-22T10:00:00+00:00", "cache_version": 1, "complete": True}])
    verdict = "approved" if pm_rating == action else "rejected" if pm_rating == "Hold" else "reduced"
    final = {"approved": 1.0, "rejected": 0.0, "reduced": 0.5}[verdict]
    parts = {"approved": [("approved_part", 1.0)], "rejected": [("rejected_part", 1.0)],
             "reduced": [("approved_part", 0.5), ("rejected_part", 0.5)]}[verdict]
    append_rows(con, "decisions", [{
        "decision_id": f"{decision_id}_{part}", "run_id": run_id, "experiment_version": version,
        "ticker": ticker, "trade_date": trade_date, "decided_at": "2026-10-22T10:00:00+00:00",
        "part": part, "verdict": verdict, "proposed_action": action,
        "direction": {"Buy": 1, "Sell": -1}[action], "proposed_size": 1.0, "final_size": final,
        "part_size": size, "pm_rating": pm_rating, "main_reason": "volatility",
    } for part, size in parts])


def run(con, book, to_date=DAYS[-1], **kw):
    kw.setdefault("price_provider", fake_prices)
    return advance(con, experiment_version=VERSION, book=book, to_date=to_date,
                   calendar=DAYS, config=CONFIG, **kw)


def snapshots(con, book, version=VERSION):
    return con.execute("SELECT snapshot_date, cash, value, drawdown FROM book_snapshots "
                       "WHERE experiment_version = ? AND book = ? ORDER BY snapshot_date",
                       [version, book]).fetchall()


def last_positions(con, book):
    import json
    raw = con.execute("SELECT positions FROM book_snapshots WHERE experiment_version = ? AND book = ? "
                      "ORDER BY snapshot_date DESC LIMIT 1", [VERSION, book]).fetchone()[0]
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def buy_aapl_on_day_0(con):
    add_decision(con, decision_id="D1", ticker="AAPL", trade_date=DAYS[0], action="Buy", pm_rating="Buy")


# ---- the fill rule --------------------------------------------------------

def test_a_decision_fills_at_the_next_days_open_not_the_same_day(con):
    buy_aapl_on_day_0(con)
    run(con, "decision", decisions_through=DAYS[-1])
    rows = snapshots(con, "decision")
    # Day 0: decided, not yet filled, so the book is still all cash.
    assert rows[0][1] == pytest.approx(100_000.0)
    # Day 1: filled at that day's open (100.0), marked at its close (110.0).
    assert rows[1][1] == pytest.approx(90_000.0)
    assert rows[1][2] == pytest.approx(90_000.0 + 100 * 110.0)


def test_one_snapshot_per_trading_day(con):
    buy_aapl_on_day_0(con)
    report = run(con, "decision", decisions_through=DAYS[-1])
    assert report.days_written == len(DAYS)
    assert [r[0] for r in snapshots(con, "decision")] == DAYS


def test_drawdown_is_recorded_against_the_running_peak(con):
    buy_aapl_on_day_0(con)
    run(con, "decision", decisions_through=DAYS[-1])
    drawdowns = [r[3] for r in snapshots(con, "decision")]
    assert all(d <= 0 for d in drawdowns)
    assert drawdowns[-1] == pytest.approx(0.0)   # AAPL only rises, so the book peaks last


def test_a_decision_on_a_non_trading_day_fills_on_the_next_trading_day(con):
    buy_aapl_on_day_0(con)
    add_decision(con, decision_id="D2", ticker="MSFT", trade_date=date(2026, 9, 5), action="Buy", pm_rating="Buy")
    run(con, "decision", decisions_through=DAYS[-1])
    assert last_positions(con, "decision")["MSFT"] == pytest.approx(200.0)   # $10k at 50.0 on 09-08


# ---- ratings to positions -------------------------------------------------

def test_an_approved_sell_exits_and_does_not_go_short(con):
    buy_aapl_on_day_0(con)
    add_decision(con, decision_id="D2", ticker="AAPL", trade_date=DAYS[2], action="Sell", pm_rating="Sell")
    run(con, "decision", decisions_through=DAYS[-1])
    assert last_positions(con, "decision") == {}


def test_a_rejected_buy_keeps_what_the_book_already_holds(con):
    buy_aapl_on_day_0(con)
    add_decision(con, decision_id="D2", ticker="AAPL", trade_date=DAYS[2], action="Buy", pm_rating="Hold")
    run(con, "decision", decisions_through=DAYS[-1])
    assert last_positions(con, "decision")["AAPL"] == pytest.approx(100.0)


def test_a_rejected_sell_keeps_the_position_in_the_decision_book_only(con):
    buy_aapl_on_day_0(con)
    add_decision(con, decision_id="D2", ticker="AAPL", trade_date=DAYS[2], action="Sell", pm_rating="Hold")
    run(con, "decision", decisions_through=DAYS[-1])
    run(con, "shadow", decisions_through=DAYS[-1])
    assert last_positions(con, "decision")["AAPL"] == pytest.approx(100.0)
    assert last_positions(con, "shadow") == {}


def test_a_reduced_sell_halves_the_position(con):
    buy_aapl_on_day_0(con)
    add_decision(con, decision_id="D2", ticker="AAPL", trade_date=DAYS[2], action="Sell", pm_rating="Underweight")
    run(con, "decision", decisions_through=DAYS[-1])
    run(con, "shadow", decisions_through=DAYS[-1])
    assert last_positions(con, "decision")["AAPL"] == pytest.approx(50.0)
    assert last_positions(con, "shadow") == {}


def test_a_decision_without_a_pm_rating_is_an_error_for_the_decision_book(con):
    add_decision(con, decision_id="D1", ticker="AAPL", trade_date=DAYS[0], action="Buy", pm_rating=None)
    with pytest.raises(ValueError, match="no pm_rating"):
        run(con, "decision")
    assert run(con, "shadow").days_written > 0   # the shadow book does not need it


# ---- the two books --------------------------------------------------------

def test_the_two_books_differ_only_by_the_rating_they_follow(con):
    """A reduced trade: decision book takes half, shadow book takes the full proposal."""
    add_decision(con, decision_id="D1", ticker="AAPL", trade_date=DAYS[0], action="Buy", pm_rating="Overweight")
    run(con, "decision", decisions_through=DAYS[-1])
    run(con, "shadow", decisions_through=DAYS[-1])
    decision_end = snapshots(con, "decision")[-1][2]
    shadow_end = snapshots(con, "shadow")[-1][2]
    # AAPL rose, so the unfiltered book gained more: the risk team cost money here.
    assert shadow_end > decision_end
    assert (shadow_end - 100_000.0) == pytest.approx(2 * (decision_end - 100_000.0))


def test_an_approved_trade_leaves_the_books_identical(con):
    buy_aapl_on_day_0(con)
    run(con, "decision", decisions_through=DAYS[-1])
    run(con, "shadow", decisions_through=DAYS[-1])
    assert [r[2] for r in snapshots(con, "decision")] == pytest.approx([r[2] for r in snapshots(con, "shadow")])


def test_a_full_rejection_leaves_the_decision_book_flat(con):
    add_decision(con, decision_id="D1", ticker="AAPL", trade_date=DAYS[0], action="Buy", pm_rating="Hold")
    run(con, "decision", decisions_through=DAYS[-1])
    run(con, "shadow", decisions_through=DAYS[-1])
    assert all(r[2] == pytest.approx(100_000.0) for r in snapshots(con, "decision"))
    assert snapshots(con, "shadow")[-1][2] > 100_000.0


def test_a_reduced_trades_two_rows_produce_one_order(con):
    """Both parts carry the same ratings; the pair must not fill twice."""
    add_decision(con, decision_id="D1", ticker="AAPL", trade_date=DAYS[0], action="Buy", pm_rating="Overweight")
    run(con, "decision", decisions_through=DAYS[-1])
    assert snapshots(con, "decision")[1][1] == pytest.approx(95_000.0)   # one half-size buy, not two


# ---- never ahead of the decisions -------------------------------------------

def test_stops_on_the_first_trading_day_after_the_latest_decision(con):
    buy_aapl_on_day_0(con)
    report = run(con, "decision")
    assert [r[0] for r in snapshots(con, "decision")] == DAYS[:2]
    assert "only logged through" in report.stopped_reason
    assert not report.stopped_on_missing_price


def test_decisions_logged_later_are_applied_when_advancing_again(con):
    buy_aapl_on_day_0(con)
    run(con, "decision")                                    # stops at DAYS[1]
    add_decision(con, decision_id="D2", ticker="AAPL", trade_date=DAYS[2], action="Sell", pm_rating="Sell")
    run(con, "decision", decisions_through=DAYS[-1])
    assert [r[0] for r in snapshots(con, "decision")] == DAYS
    assert last_positions(con, "decision") == {}


def test_decisions_through_lets_the_book_cross_days_with_no_decisions(con):
    buy_aapl_on_day_0(con)
    report = run(con, "decision", decisions_through=DAYS[3])
    assert report.last_date == DAYS[4]
    assert report.stopped_reason is None


# ---- missing prices ---------------------------------------------------------

def test_a_missing_price_stops_the_run_instead_of_skipping_the_day(con):
    buy_aapl_on_day_0(con)

    def gappy(ticker, as_of):
        df = fake_prices(ticker, as_of)
        return df[df["Date"] != DAYS[2].isoformat()] if ticker == "AAPL" else df

    report = run(con, "decision", decisions_through=DAYS[-1], price_provider=gappy)
    assert report.stopped_on_missing_price
    assert "no close for AAPL" in report.stopped_reason
    assert [r[0] for r in snapshots(con, "decision")] == DAYS[:2]   # nothing after the gap


def test_a_missing_price_for_an_unheld_ticker_does_not_stop_the_run(con):
    buy_aapl_on_day_0(con)

    def no_msft(ticker, as_of):
        if ticker == "MSFT":
            raise FileNotFoundError("no MSFT prices")
        return fake_prices(ticker, as_of)

    report = run(con, "decision", decisions_through=DAYS[-1], price_provider=no_msft)
    assert report.days_written == len(DAYS)


# ---- resuming and advancing in chunks -------------------------------------

def test_advancing_in_chunks_matches_one_long_run(con, tmp_path):
    """Two calls to a cycle date must give the same series as one call to the end."""
    buy_aapl_on_day_0(con)
    run(con, "decision", to_date=DAYS[2], decisions_through=DAYS[-1])
    run(con, "decision", to_date=DAYS[-1], decisions_through=DAYS[-1])
    chunked = snapshots(con, "decision")

    other = init_db(tmp_path / "once.duckdb")
    buy_aapl_on_day_0(other)
    advance(other, experiment_version=VERSION, book="decision", to_date=DAYS[-1], decisions_through=DAYS[-1],
            calendar=DAYS, config=CONFIG, price_provider=fake_prices)
    at_once = snapshots(other, "decision")
    other.close()

    assert [r[0] for r in chunked] == [r[0] for r in at_once]
    assert [r[2] for r in chunked] == pytest.approx([r[2] for r in at_once])


def test_resuming_rebuilds_cash_and_positions_from_the_last_snapshot(con):
    buy_aapl_on_day_0(con)
    run(con, "decision", to_date=DAYS[1])
    report = run(con, "decision", decisions_through=DAYS[-1])
    assert report.first_date == DAYS[2]      # continued, did not restart
    assert len(snapshots(con, "decision")) == len(DAYS)
    assert last_positions(con, "decision")["AAPL"] == pytest.approx(100.0)


def test_advancing_to_a_date_already_written_writes_nothing(con):
    buy_aapl_on_day_0(con)
    run(con, "decision", to_date=DAYS[2], decisions_through=DAYS[-1])
    before = snapshots(con, "decision")
    assert run(con, "decision", to_date=DAYS[2], decisions_through=DAYS[-1]).days_written == 0
    assert snapshots(con, "decision") == before


def test_dry_run_writes_nothing(con):
    buy_aapl_on_day_0(con)
    report = run(con, "decision", decisions_through=DAYS[-1], dry_run=True)
    assert report.days_written == len(DAYS)
    assert snapshots(con, "decision") == []


# ---- scoping and edges ----------------------------------------------------

def test_only_this_experiment_versions_decisions_are_used(con):
    buy_aapl_on_day_0(con)
    add_decision(con, decision_id="D2", ticker="MSFT", trade_date=DAYS[0], action="Buy", pm_rating="Buy",
                 version="other_version")
    run(con, "decision", decisions_through=DAYS[-1])
    assert "MSFT" not in last_positions(con, "decision")


def test_no_decisions_is_not_an_error(con):
    report = run(con, "decision")
    assert report.days_written == 0
    assert snapshots(con, "decision") == []


def test_an_unknown_book_is_rejected(con):
    with pytest.raises(ValueError, match="decision.*shadow|shadow.*decision"):
        advance(con, experiment_version=VERSION, book="real", to_date=DAYS[-1],
                calendar=DAYS, config=CONFIG, price_provider=fake_prices)
