"""Advance a book through the calendar and write one snapshot per trading day.

The simulator is a *step to a date*, not a batch over a finished experiment.
Within an experiment version the loop is circular: decisions feed the books,
and ``writers.insert_memo`` reads the decision book's latest snapshot before
it will allow a risk-level increase, so the memo that shapes the next four
weeks of decisions cannot be written until the book has caught up to the
cycle date.

``advance`` therefore runs forward from wherever the book already is to a
date the caller names, but never past the decisions that exist: the last day
it writes is the first trading day after the latest decision (or after
``decisions_through``, when the caller knows decisions are complete to a later
date, e.g. days on which every proposal was Hold). Snapshots are append-only,
so a day written before its decisions were logged could never be corrected.

Both books read the same rows of ``decisions``: the decision book follows the
Portfolio Manager's ``pm_rating``, the shadow book the Trader's
``proposed_action``. Ratings map to positions as in the proposal (see
``config``). A rejected trade is one the PM turned into Hold, so it places no
order and the decision book keeps what it held.

Resuming is automatic: a book's state is rebuilt from its most recent
snapshot, whose ``cash`` and ``positions`` are the full state. An interrupted
run is continued simply by calling ``advance`` again. A day with a missing
price stops the run there (nothing is guessed, and later days would build on
a wrong state); fix the price data and call ``advance`` again.

    python -m shadow.sim.simulator --experiment-version llm_stats --book decision --to 2026-09-28
    python -m shadow.sim.simulator --experiment-version llm_stats --book shadow --to 2026-09-28
    python -m shadow.sim.simulator --experiment-version llm_stats --book decision --to 2026-09-28 --dry-run
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import duckdb

from shadow.data.writers import insert_book_snapshot
from shadow.sim.book import Book, Order, order_for_rating
from shadow.sim.config import DEFAULT, SimConfig

logger = logging.getLogger(__name__)

PriceProvider = Callable[[str, str], "pandas.DataFrame"]  # noqa: F821 (ticker, as_of) -> OHLCV

#: Which rating column each book follows. This single mapping is the entire
#: difference between the two books.
RATING_COLUMN = {"decision": "pm_rating", "shadow": "proposed_action"}


@dataclass
class AdvanceReport:
    book: str
    experiment_version: str
    days_written: int = 0
    first_date: date | None = None
    last_date: date | None = None
    fills: int = 0
    #: Why the book stopped before ``to_date``, if it did.
    stopped_reason: str | None = None
    #: True when it stopped on missing prices (an error), not on missing decisions.
    stopped_on_missing_price: bool = False


# -- prices -----------------------------------------------------------------

class PriceBook:
    """Open and close prices per ticker and day, loaded once per ticker."""

    def __init__(self, as_of: str, price_provider: PriceProvider | None = None):
        self._as_of = as_of
        self._provider = price_provider or _default_price_provider
        self._frames: dict[str, object] = {}

    def _frame(self, ticker: str):
        if ticker not in self._frames:
            df = self._provider(ticker, self._as_of)
            self._frames[ticker] = df.set_index(df["Date"].astype(str).str[:10])
        return self._frames[ticker]

    def _get(self, ticker: str, day: date, column: str) -> float | None:
        try:
            price = float(self._frame(ticker).loc[day.isoformat(), column])
        except (KeyError, FileNotFoundError, ValueError, TypeError):
            return None
        return price if math.isfinite(price) and price > 0 else None

    def open(self, ticker: str, day: date) -> float | None:
        return self._get(ticker, day, "Open")

    def close(self, ticker: str, day: date) -> float | None:
        return self._get(ticker, day, "Close")


def _default_price_provider(ticker: str, as_of: str):
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
    # Prices as reported: the default fill_gaps=True would carry yesterday's price
    # into a missing cell, so a gap would fill at a stale price instead of stopping.
    return load_ohlcv(ticker, as_of, fill_gaps=False)


# -- reading state and decisions --------------------------------------------

def _resume_point(con: duckdb.DuckDBPyConnection, experiment_version: str, book: str,
                  config: SimConfig) -> tuple[Book, date | None]:
    """Rebuild the book from its last snapshot; return it and that date."""
    row = con.execute(
        "SELECT snapshot_date, cash, positions FROM book_snapshots "
        "WHERE experiment_version = ? AND book = ? ORDER BY snapshot_date DESC LIMIT 1",
        [experiment_version, book],
    ).fetchone()
    if row is None:
        return Book(config=config), None
    snapshot_date, cash, positions = row
    held = json.loads(positions) if isinstance(positions, str) else (positions or {})
    return Book(config=config, cash=float(cash), positions={t: float(s) for t, s in held.items()}), snapshot_date


def _ratings(con: duckdb.DuckDBPyConnection, experiment_version: str, book: str) -> dict[tuple[str, date], str]:
    """The rating this book follows for every (ticker, trade_date) with a decision.

    A reduced trade is two rows (approved and rejected part) carrying the same
    ratings, so rows collapse to one per (ticker, trade_date).
    """
    column = RATING_COLUMN[book]
    rows = con.execute(
        f"SELECT DISTINCT ticker, trade_date, {column} FROM decisions "
        "WHERE experiment_version = ? ORDER BY trade_date, ticker",
        [experiment_version],
    ).fetchall()
    ratings: dict[tuple[str, date], str] = {}
    missing, conflicting = [], []
    for ticker, trade_date, rating in rows:
        key = (ticker, trade_date)
        if rating is None:
            missing.append(f"{ticker} {trade_date}")
        elif key in ratings and ratings[key] != rating:
            conflicting.append(f"{ticker} {trade_date}")
        ratings[key] = rating
    if missing:
        raise ValueError(f"{len(missing)} {experiment_version} decision(s) have no {column}, which the "
                         f"{book} book needs (e.g. {', '.join(missing[:5])}); the decision logger must fill it")
    if conflicting:
        raise ValueError(f"{experiment_version} decision parts disagree on {column} for "
                         f"{', '.join(conflicting[:5])}")
    return ratings


def _orders_by_fill_day(ratings: dict[tuple[str, date], str], days: list[date]) -> dict[date, list[Order]]:
    """Orders keyed by the day they fill: the first trading day after the trade date.

    If two decisions for one ticker would fill on the same day (one dated on a
    non-trading day), the later decision wins. Hold places no order.
    """
    latest: dict[tuple[date, str], tuple[date, str]] = {}
    for (ticker, trade_date), rating in sorted(ratings.items(), key=lambda kv: kv[0][1]):
        i = bisect.bisect_right(days, trade_date)
        if i == len(days):
            continue  # the calendar does not reach its fill day yet
        latest[(days[i], ticker)] = (trade_date, rating)
    orders: dict[date, list[Order]] = {}
    for (fill_day, ticker), (_, rating) in sorted(latest.items()):
        order = order_for_rating(ticker, rating)
        if order is not None:
            orders.setdefault(fill_day, []).append(order)
    return orders


# -- the loop ---------------------------------------------------------------

def advance(
    con: duckdb.DuckDBPyConnection, *, experiment_version: str, book: str, to_date,
    decisions_through=None, calendar: list[date] | None = None, config: SimConfig = DEFAULT,
    price_provider: PriceProvider | None = None, dry_run: bool = False,
) -> AdvanceReport:
    """Run ``book`` forward to ``to_date``, writing one snapshot per trading day.

    Starts from the day after the book's last snapshot, or from the first
    trading day on or after its first decision if it has none. Stops early at
    the first trading day after ``decisions_through`` (default: the latest
    decision's trade date), or at a day whose prices cannot be loaded.
    """
    if book not in RATING_COLUMN:
        raise ValueError(f"book must be 'decision' or 'shadow', got {book!r}")
    to_date = date.fromisoformat(to_date) if isinstance(to_date, str) else to_date
    if isinstance(decisions_through, str):
        decisions_through = date.fromisoformat(decisions_through)

    if calendar is None:
        from shadow.data.conditions import trading_calendar
        calendar = trading_calendar()
    days = sorted(calendar)

    report = AdvanceReport(book=book, experiment_version=experiment_version)
    ratings = _ratings(con, experiment_version, book)
    if not ratings:
        logger.warning("no decisions for experiment version %r; nothing to simulate", experiment_version)
        return report
    orders_by_day = _orders_by_fill_day(ratings, days)

    # Never write a day whose fills could still change: the last safe day is
    # the first trading day after the date decisions are known to be complete.
    through = max(decisions_through or date.min, max(d for _, d in ratings))
    i = bisect.bisect_right(days, through)
    last_safe = days[i] if i < len(days) else days[-1]
    end = min(to_date, last_safe)
    if end < to_date:
        report.stopped_reason = (f"decisions are only logged through {through}, so the book stops at {end}; "
                                 f"log later decisions (or pass decisions_through) and advance again")

    state, last_snapshot = _resume_point(con, experiment_version, book, config)
    start = min(d for _, d in ratings)
    window = [d for d in days if start <= d <= end and (last_snapshot is None or d > last_snapshot)]
    if not window:
        return report

    prices = PriceBook(as_of=end.isoformat(), price_provider=price_provider)
    for day in window:
        due = orders_by_day.get(day, [])
        open_prices, missing_open = _price_map(prices, [o.ticker for o in due], day, "open")
        to_mark = sorted(set(state.positions) | {o.ticker for o in due})
        close_prices, missing_close = _price_map(prices, to_mark, day, "close")
        if missing_open or missing_close:
            parts = ([f"no open for {', '.join(missing_open)}"] if missing_open else []) + \
                    ([f"no close for {', '.join(missing_close)}"] if missing_close else [])
            report.stopped_reason = f"{day}: {'; '.join(parts)}; stopped before writing this day"
            report.stopped_on_missing_price = True
            break

        fills = state.apply_fills(due, open_prices)
        value = state.value(close_prices)
        report.fills += len(fills)

        if not dry_run:
            insert_book_snapshot(con, experiment_version=experiment_version, book=book,
                                 snapshot_date=day, cash=state.cash, value=value,
                                 positions=dict(state.positions))
        report.days_written += 1
        report.first_date = report.first_date or day
        report.last_date = day
    return report


def _price_map(prices: PriceBook, tickers, day: date, which: str) -> tuple[dict[str, float], list[str]]:
    out: dict[str, float] = {}
    missing: list[str] = []
    for ticker in tickers:
        price = prices.open(ticker, day) if which == "open" else prices.close(ticker, day)
        if price is None:
            missing.append(ticker)
        else:
            out[ticker] = price
    return out, sorted(missing)


def main(argv: list[str] | None = None) -> int:
    from shadow.data.db import check_schema_version, connect, resolve_db

    parser = argparse.ArgumentParser(description="Advance a book and write its daily snapshots.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true", help="use warehouse/dev.duckdb")
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--book", required=True, choices=sorted(RATING_COLUMN))
    parser.add_argument("--to", required=True, help="advance through this date (YYYY-MM-DD)")
    parser.add_argument("--decisions-through",
                        help="decisions are complete through this date (default: the latest decision's date)")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    path = resolve_db(args.db, args.dev)
    con = connect(path, read_only=args.dry_run)
    check_schema_version(con, path)
    report = advance(con, experiment_version=args.experiment_version, book=args.book,
                     to_date=args.to, decisions_through=args.decisions_through, dry_run=args.dry_run)
    print(f"database: {path}")
    print(f"{report.book} book / {report.experiment_version}")
    print(f"days written: {report.days_written}"
          f"{' (dry run, nothing written)' if args.dry_run else ''}"
          f"{f' [{report.first_date}..{report.last_date}]' if report.days_written else ''}")
    print(f"fills: {report.fills}")
    if report.stopped_reason:
        print(f"stopped early: {report.stopped_reason}")
    return 1 if report.stopped_on_missing_price else 0


if __name__ == "__main__":
    raise SystemExit(main())
