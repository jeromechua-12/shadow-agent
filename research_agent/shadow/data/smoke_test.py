"""Stand-ins for teammates' components, to exercise the data layer end to end on real runs.

Writes only to the dev database and only under experiment_version "smoke_test":
- a rule-based "risk team": reject when ATR% >= 3.5%, halve when >= 2.5%, else approve;
- an "outcome judge": 10-trading-day return vs SPY from real prices (next-day open to
  the 10th trading day's close), through insert_outcome;
- one template memo dated before the sample, so past_memos has something to read.

None of this is the real risk team or outcome judge; it only feeds the query
functions realistic shapes. It refuses to run against the main database.

Usage:
    python -m shadow.data.smoke_test --as-of 2026-09-28
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import duckdb

from shadow.data.checks import overview, run_checks
from shadow.data.conditions import market_conditions
from shadow.data.db import MAIN_DB_NAME, dev_db_path, init_db
from shadow.data.loader import load_all
from shadow.data.paths import sample_cache_dir
from shadow.data.queries import CalibrationTools
from shadow.data.writers import (
    CATEGORIES,
    insert_decision,
    insert_memo,
    insert_outcome,
    outcome_dates,
)

VERSION = "smoke_test"


def stand_in_decisions(con: duckdb.DuckDBPyConnection) -> int:
    """One rule-based decision per current run with a Buy or Sell proposal."""
    rows = con.execute(
        "SELECT r.run_id, p.action, c.atr_pct, c.vix FROM current_runs r "
        "JOIN trader_proposals p ON p.run_id = r.run_id LEFT JOIN latest_conditions c ON c.run_id = r.run_id "
        "WHERE p.action IN ('Buy', 'Sell') AND NOT EXISTS (SELECT 1 FROM decisions d "
        "WHERE d.run_id = r.run_id AND d.experiment_version = ?) ORDER BY r.trade_date, r.ticker",
        [VERSION]).fetchall()
    for run_id, _action, atr, vix in rows:
        atr = atr or 0.0
        final, reason = (0.0, "volatility") if atr >= 0.035 else (0.5, "volatility") if atr >= 0.025 else (1.0, "none")
        verdict = {1.0: "approved", 0.5: "reduced", 0.0: "rejected"}[final]
        insert_decision(
            con, run_id=run_id, experiment_version=VERSION, verdict=verdict, proposed_size=1.0, final_size=final,
            main_reason=reason, explanation=f"smoke_test stand-in rule: ATR% {atr:.2%}, VIX {vix}",
            risk_ratings={"volatility": 0 if atr < 0.02 else 1 if atr < 0.025 else 2 if atr < 0.035 else 3,
                          "event_risk": 2 if (vix or 0) >= 20 else 1, "valuation": 1, "liquidity": 0,
                          "concentration": 0})
    return len(rows)


def stand_in_outcomes(con: duckdb.DuckDBPyConnection) -> int:
    """10-trading-day result vs SPY for every decision part without one, if already known."""
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    today = date.today().isoformat()
    todo = con.execute("SELECT d.decision_id, d.ticker, d.trade_date FROM decisions d LEFT JOIN outcomes o "
                       "ON o.decision_id = d.decision_id WHERE d.experiment_version = ? AND o.decision_id IS NULL "
                       "ORDER BY d.trade_date", [VERSION]).fetchall()
    prices: dict = {}

    def series(sym):
        if sym not in prices:
            df = load_ohlcv(sym, today)
            prices[sym] = df.set_index(df["Date"].astype(str).str[:10])
        return prices[sym]

    def ret(df, entry, exit_):
        return float(df.loc[exit_, "Close"] / df.loc[entry, "Open"] - 1)

    spy = series("SPY")
    calendar = [date.fromisoformat(d) for d in spy.index]
    written = 0
    for decision_id, ticker, trade_date in todo:
        # Same rule as insert_outcome: next trading day, then the 10th trading day after it.
        entry, resolved = outcome_dates(trade_date, calendar)
        if resolved is None:
            continue  # not resolved yet
        e, x = entry.isoformat(), resolved.isoformat()
        insert_outcome(con, decision_id=decision_id, stock_return=ret(series(ticker), e, x),
                       spy_return=ret(spy, e, x), calendar=calendar)
        written += 1
    return written


def stand_in_memo(con: duckdb.DuckDBPyConnection, cycle_date: str) -> None:
    if con.execute("SELECT 1 FROM memos WHERE experiment_version = ?", [VERSION]).fetchone():
        return
    insert_memo(con, experiment_version=VERSION, cycle_date=cycle_date, risk_level=3,
                memo_text="smoke_test template memo: keep every category until there is evidence.",
                advice={c: {"advice": "keep", "confidence": "low", "reasoning": "no resolved trades yet",
                            "cited_ids": []} for c in CATEGORIES},
                hypotheses=[{"statement": "Halving trades when ATR% >= 2.5% avoids losses",
                             "category": "volatility", "subgroup": {"main_reason": "volatility"}}],
                safety_passed=True, model="template")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the sample into the dev DB and exercise the query functions.")
    parser.add_argument("--db", default=None, help="dev DuckDB file (default warehouse/dev.duckdb)")
    parser.add_argument("--as-of", default="2026-09-28", help="calibration date for the query demo")
    parser.add_argument("--memo-date", default="2026-08-31", help="date of the template memo")
    parser.add_argument("--out", help="also write the full query results to this JSON file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    path = Path(args.db) if args.db else dev_db_path()
    if path.name == MAIN_DB_NAME:
        raise SystemExit(f"refusing to write smoke_test rows into the main database {path}")
    con = init_db(path)
    # The dev database is built from the sample cache only, never the real one.
    loaded = load_all(con, cache_dir=sample_cache_dir(), market=market_conditions)
    stand_in_memo(con, args.memo_date)
    made = stand_in_decisions(con)
    resolved = stand_in_outcomes(con)
    print(f"database: {path}")
    print(f"loaded: {loaded}")
    print(f"stand-ins: {made} decisions, {resolved} outcomes")

    findings = run_checks(con)
    for f in findings:
        if f.count:
            print(f"{'FAIL' if f.severity == 'error' else 'WARN'} {f.check}: {f.count}  e.g. {f.examples[:2]}")
    print(f"checks: {sum(1 for f in findings if not f.count)}/{len(findings)} clean")

    tools = CalibrationTools(con, args.as_of, VERSION)
    worst = tools.search_trades(order="worst", limit=3)
    results = {
        "overview": overview(con),
        "cycle_summary": tools.cycle_summary(),
        "search_trades_worst": worst,
        "subgroup_by_verdict": tools.subgroup_stats(split_by="verdict"),
        "subgroup_fed_news": tools.subgroup_stats({"news_keyword": "fed"}),
        "read_trade": tools.read_trade(worst["trades"][0]["decision_id"]) if worst["trades"] else None,
        "past_memos": tools.past_memos(),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return 1 if any(f.count and f.severity == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
