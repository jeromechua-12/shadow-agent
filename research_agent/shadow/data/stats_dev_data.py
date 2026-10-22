"""Synthetic, test-only data for developing the Statistics/Scoring engine.

Everything in here is made up: fake runs, a fake rule-based "risk team" and a
fake two-book series. None of it is aresh's decision logger / two-book
simulator / outcome judge, and none of it is meant to be. Its only job is to
give the statistics engine (cusum_trend in queries.py, drawdown, decision vs
shadow returns) a realistic, schema-valid dataset to run against before that
real component exists, the same way smoke_test.py fakes a risk team to
exercise the query functions.

Writes only to the dev database, under experiment_version "stats_dev". Two
deliberate patterns are built in, so the statistics have something to find:
  - event_risk rejections start mostly correct (the market dips after they're
    rejected) and flip to mostly wrong (missed_gain) half way through the
    period: cusum_trend on the event_risk category should flag a downward
    trend. The other four categories keep a stable ~55% success rate and
    should not flag.
  - the decision book takes a deliberate ~12% drawdown in the middle of the
    period and lags the shadow book overall, so book-level drawdown and
    decision-vs-shadow return comparisons have something non-trivial to show.

Usage:
    python -m shadow.data.stats_dev_data --out stats_dev.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from shadow.data.checks import overview, run_checks
from shadow.data.db import MAIN_DB_NAME, dev_db_path, init_db
from shadow.data.loader import append_rows
from shadow.data.queries import CATEGORIES, CalibrationTools
from shadow.data.writers import insert_book_snapshot, insert_decision, insert_outcome

VERSION = "stats_dev"
SEED = 20261022
TICKERS = ("NVDA", "AAPL", "JPM")
START = date(2026, 3, 2)              # ~6.5 months of history: enough per-category volume for CUSUM to have power
LAST_TRADE_DATE = date(2026, 9, 18)   # leaves room for every outcome to resolve by CALENDAR_END
CALENDAR_END = date(2026, 10, 22)
NOISE_BAND = 0.012                    # just above writers.NOISE_BAND, so "noise" draws are unambiguous


def business_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _insert_run(con, rng: random.Random, run_id: str, ticker: str, trade_date: date) -> str:
    action = "Buy" if rng.random() < 0.7 else "Sell"
    append_rows(con, "runs", [{"run_id": run_id, "ticker": ticker, "trade_date": trade_date,
                               "started_at": f"{trade_date}T10:00:00+00:00", "cache_version": 1, "complete": True}])
    append_rows(con, "trader_proposals", [{"run_id": run_id, "ticker": ticker, "trade_date": trade_date,
                                           "action": action, "direction": 1 if action == "Buy" else -1,
                                           "entry_price": 100.0, "stop_loss": 95.0 if action == "Buy" else 105.0,
                                           "position_sizing": "full", "reasoning": "stats_dev stand-in",
                                           "structured_ok": True}])
    return action


def _half(trade_date: date) -> str:
    return "first" if trade_date <= START + (LAST_TRADE_DATE - START) / 2 else "second"


def _event_risk_wants_success(trade_date: date) -> bool:
    """First half: always a correct rejection. Second half: always a missed gain.

    Deterministic (unlike every other category, which is a noisy 55% rate) so the
    CUSUM demo shows an unambiguous, clean signal rather than depending on getting
    a large-enough sample for a noisy split to clear the decision threshold.
    """
    return _half(trade_date) == "first"


STABLE_SUCCESS_RATE = 0.55  # every category except event_risk holds steady at this rate


def build_decisions_and_outcomes(con, rng: random.Random, calendar: list[date]) -> int:
    made = 0
    for trade_date in business_days(START, LAST_TRADE_DATE):
        for ticker in TICKERS:
            run_id = f"synthetic_{ticker}_{trade_date.isoformat()}"
            action = _insert_run(con, rng, run_id, ticker, trade_date)
            direction = 1 if action == "Buy" else -1

            reject = rng.random() < 0.4
            main_reason = rng.choice(CATEGORIES) if reject else "none"
            ratings = {c: (2 if c == main_reason else 1) for c in CATEGORIES}
            verdict = "rejected" if reject else "approved"
            final_size = 0.0 if reject else 1.0
            ids = insert_decision(
                con, run_id=run_id, experiment_version=VERSION, verdict=verdict,
                proposed_size=1.0, final_size=final_size, risk_ratings=ratings, main_reason=main_reason,
                explanation=f"stats_dev stand-in rule: main_reason={main_reason}",
            )
            made += len(ids)

            want_success = (_event_risk_wants_success(trade_date) if main_reason == "event_risk"
                           else rng.random() < STABLE_SUCCESS_RATE)
            # label_for: approved_part succeeds (correct_approval) on score > 0;
            # rejected_part succeeds (correct_rejection) on score < 0.
            score_sign = (1 if want_success else -1) if not reject else (-1 if want_success else 1)
            if rng.random() < 0.08:
                score = rng.uniform(-NOISE_BAND, NOISE_BAND)  # an occasional noise trade
            else:
                score = score_sign * rng.uniform(0.02, 0.08)
            spy_return = 0.01
            stock_return = spy_return + score / direction
            insert_outcome(con, decision_id=ids[0], stock_return=stock_return, spy_return=spy_return,
                           calendar=calendar)
    return made


def build_book_snapshots(con, rng: random.Random, calendar: list[date]) -> int:
    # Extends through CALENDAR_END (past LAST_TRADE_DATE), so the default --as-of
    # (after every outcome has resolved) still falls inside the books' date range
    # and cycle_summary's "last 28 days" window has snapshots to compute a return from.
    days = [d for d in calendar if START <= d <= CALENDAR_END]
    drawdown_start = days[len(days) // 3]
    drawdown_end = days[len(days) // 3 + 10]
    decision_value, shadow_value = 1_000_000.0, 1_000_000.0
    written = 0
    for d in days:
        if drawdown_start <= d <= drawdown_end:
            decision_value *= 1 - rng.uniform(0.005, 0.02)   # the deliberate ~12% drawdown
        else:
            decision_value *= 1 + rng.uniform(-0.004, 0.006)
        shadow_value *= 1 + rng.uniform(-0.003, 0.008)       # modestly better drift throughout
        insert_book_snapshot(con, experiment_version=VERSION, book="decision", snapshot_date=d,
                             cash=0.0, value=decision_value, positions={})
        insert_book_snapshot(con, experiment_version=VERSION, book="shadow", snapshot_date=d,
                             cash=0.0, value=shadow_value, positions={})
        written += 2
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic dataset for statistics-engine development.")
    parser.add_argument("--db", default=None, help="dev DuckDB file (default warehouse/dev.duckdb)")
    parser.add_argument("--as-of", default=CALENDAR_END.isoformat(),
                        help="calibration date for the query demo (default: after every outcome has resolved)")
    parser.add_argument("--out", help="also write the full query results to this JSON file")
    args = parser.parse_args(argv)

    path = Path(args.db) if args.db else dev_db_path()
    if path.name == MAIN_DB_NAME:
        raise SystemExit(f"refusing to write stats_dev rows into the main database {path}")
    con = init_db(path)
    if con.execute("SELECT 1 FROM decisions WHERE experiment_version = ?", [VERSION]).fetchone():
        raise SystemExit(f"{path} already has {VERSION!r} rows; delete the dev database and rerun to regenerate")

    rng = random.Random(SEED)
    calendar = business_days(START, CALENDAR_END)
    made = build_decisions_and_outcomes(con, rng, calendar)
    snaps = build_book_snapshots(con, rng, calendar)
    print(f"database: {path}")
    print(f"stand-ins: {made} decisions (with outcomes), {snaps} book snapshots")

    findings = run_checks(con, calendar=calendar)
    for f in findings:
        if f.count:
            print(f"{'FAIL' if f.severity == 'error' else 'WARN'} {f.check}: {f.count}  e.g. {f.examples[:2]}")
    print(f"checks: {sum(1 for f in findings if not f.count)}/{len(findings)} clean")

    tools = CalibrationTools(con, args.as_of, VERSION)
    results = {
        "overview": overview(con),
        "cycle_summary": tools.cycle_summary(),
        "subgroup_by_verdict": tools.subgroup_stats(split_by="verdict"),
        "event_risk_trend": tools.subgroup_stats({"main_reason": "event_risk"}),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if any(f.count and f.severity == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
