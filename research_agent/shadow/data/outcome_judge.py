"""The real Outcome judge: scores every resolved decision 10 trading days later.

For each decision without an outcome yet, once its 10-trading-day horizon has
elapsed (the next trading day after the trade date is the entry / next-day
open; the 10th trading day after entry is the exit / close): fetches real
prices and hands them to ``insert_outcome``, which computes score and label
(score = direction x (stock_return - spy_return), noise band +-1%, same rule
everywhere in this project). A decision whose horizon has not elapsed yet is
left pending, never guessed at; one whose price data is missing is skipped
and reported, not silently dropped.

Usage:
    python -m shadow.data.outcome_judge                        # every pending decision, main DB
    python -m shadow.data.outcome_judge --dev                   # dev DB
    python -m shadow.data.outcome_judge --experiment-version llm_stats
    python -m shadow.data.outcome_judge --dry-run                # report only; writes nothing
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import duckdb

from shadow.data.writers import InvalidRecord, insert_outcome, outcome_dates

logger = logging.getLogger(__name__)

PriceProvider = Callable[[str, str], "pandas.DataFrame"]  # noqa: F821 (ticker, as_of) -> OHLCV rows up to as_of


@dataclass
class ScoringReport:
    scored: int = 0
    still_pending: int = 0
    skipped_no_price: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _default_price_provider(ticker: str, as_of: str):
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
    return load_ohlcv(ticker, as_of)


def _pending_decisions(con: duckdb.DuckDBPyConnection, experiment_version: str | None):
    """decision_id, ticker, trade_date for every decision that has no outcome yet."""
    sql = ("SELECT d.decision_id, d.ticker, d.trade_date FROM decisions d "
           "LEFT JOIN outcomes o ON o.decision_id = d.decision_id WHERE o.decision_id IS NULL")
    params: list = []
    if experiment_version is not None:
        sql += " AND d.experiment_version = ?"
        params.append(experiment_version)
    sql += " ORDER BY d.trade_date, d.decision_id"
    return con.execute(sql, params).fetchall()


def _price_return(df, entry: date, exit_: date) -> float:
    e, x = entry.isoformat(), exit_.isoformat()
    return float(df.loc[x, "Close"] / df.loc[e, "Open"] - 1)


def score_pending_decisions(
    con: duckdb.DuckDBPyConnection, *, experiment_version: str | None = None,
    calendar: list[date] | None = None, today: date | None = None, dry_run: bool = False,
    price_provider: PriceProvider | None = None,
) -> ScoringReport:
    """Score every decision whose 10-trading-day horizon has elapsed; leave the rest pending.

    ``calendar`` defaults to SPY's real trading days; pass your own only for
    tests. ``today`` defaults to the real date. ``price_provider`` defaults to
    ``load_ohlcv``; inject a fake for tests so nothing touches the network.
    """
    if calendar is None:
        from shadow.data.conditions import trading_calendar
        calendar = trading_calendar()
    days = sorted(calendar)
    today = today or date.today()
    price_provider = price_provider or _default_price_provider

    report = ScoringReport()
    series_cache: dict[str, object] = {}

    def series(ticker: str):
        if ticker not in series_cache:
            df = price_provider(ticker, today.isoformat())
            series_cache[ticker] = df.set_index(df["Date"].astype(str).str[:10])
        return series_cache[ticker]

    for decision_id, ticker, trade_date in _pending_decisions(con, experiment_version):
        try:
            entry, resolved = outcome_dates(trade_date, days)
        except InvalidRecord as e:
            report.errors.append(f"{decision_id}: {e}")
            continue
        if resolved is None or resolved > today:
            report.still_pending += 1
            continue
        try:
            stock_return = _price_return(series(ticker), entry, resolved)
            spy_return = _price_return(series("SPY"), entry, resolved)
        except (KeyError, FileNotFoundError, ValueError) as e:
            report.skipped_no_price.append(f"{decision_id} ({ticker}, {entry}..{resolved}): {e}")
            continue
        if dry_run:
            report.scored += 1
            continue
        try:
            insert_outcome(con, decision_id=decision_id, stock_return=stock_return,
                           spy_return=spy_return, calendar=days)
            report.scored += 1
        except InvalidRecord as e:
            report.errors.append(f"{decision_id}: {e}")
    return report


def main(argv: list[str] | None = None) -> int:
    from shadow.data.db import check_schema_version, connect, resolve_db

    parser = argparse.ArgumentParser(description="Score every resolved decision 10 trading days later.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true", help="use warehouse/dev.duckdb")
    parser.add_argument("--experiment-version", help="score only this experiment version (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be scored; write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    path = resolve_db(args.db, args.dev)
    con = connect(path)
    check_schema_version(con, path)
    report = score_pending_decisions(con, experiment_version=args.experiment_version, dry_run=args.dry_run)
    print(f"database: {path}")
    print(f"scored: {report.scored}{' (dry run, nothing written)' if args.dry_run else ''}")
    print(f"still pending (horizon not reached): {report.still_pending}")
    if report.skipped_no_price:
        print(f"skipped (no price data): {len(report.skipped_no_price)}")
        for s in report.skipped_no_price[:5]:
            print(f"  {s}")
    if report.errors:
        print(f"errors: {len(report.errors)}")
        for e in report.errors[:5]:
            print(f"  {e}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
