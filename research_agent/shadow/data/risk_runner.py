"""Replay the cached Trader proposals through the risk debate and Portfolio Manager, day by day.

For each cached trading date, in order:
1. On a cycle date (every ``CYCLE_DAYS`` cached dates): score due outcomes, bring
   both ``feedback`` books up to the date, and run one calibration cycle. This
   comes before the day's trades, since a memo applies from its own cycle date.
2. Per setup: advance the decision book, then run the risk debate + PM on every
   Buy/Sell proposal and log the decision. Holds are skipped. A PM whose output
   cannot be parsed is run once more; if it fails again the trade is logged as Hold.
At the end both books of both setups are advanced to the last date, and every
outcome whose 10 trading days have passed by today is scored.

Upstream's graph is reused from the Trader onward (``update_state(as_node="Trader")``)
with its SQLite checkpointer, so a stopped run resumes where it was. The memory
log is off. A (setup, ticker, date) that already has a decision is skipped.

The run refuses to start if a proposal still to decide is dated before a book's last
snapshot (e.g. a second ``--tickers`` pass): its fill could never be applied. It stops
when a missing price halts a book, and logs outcomes it could not score and cycles
that wrote no memo.

    python -m shadow.data.risk_runner --dev --tickers NVDA --through 2026-09-01 --dry-run
    python -m shadow.data.risk_runner --dev --tickers NVDA --through 2026-09-01 --max-pairs 2
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import re
from datetime import date
from pathlib import Path

from shadow.data import evidence
from shadow.data.cache_runner import REPORT_KEYS, UsageCounter, openrouter_spend, read_cache, spend_since
from shadow.data.calibration_runner import run_cycle
from shadow.data.db import check_schema_version, connect, resolve_db
from shadow.data.outcome_judge import score_pending_decisions
from shadow.data.paths import warehouse_dir
from shadow.data.risk_memo import current_memo
from shadow.data.writers import CATEGORIES, insert_decision
from shadow.sim.config import DEFAULT
from shadow.sim.simulator import advance
from tradingagents.graph.checkpointer import get_checkpointer, thread_id
from tradingagents.portfolio import PortfolioContext, Position

logger = logging.getLogger(__name__)

VERSIONS = ("original", "feedback")
FEEDBACK = "feedback"
CYCLE_DAYS = 20
RETRY_STATUSES = ("api_error", "not_configured")   # calibration attempts a rerun must redo
PM_ATTEMPTS = 2   # a PM output that cannot be parsed is retried once, then logged as Hold
UNPARSED_NOTE = "PM output unparseable after one retry; recorded as Hold."
#: Final size per Trader action and PM rating (proposed size is always 1.0); any other rating rejects.
FINAL_SIZE = {"Buy": {"Buy": 1.0, "Overweight": 0.5}, "Sell": {"Sell": 1.0, "Underweight": 0.5}}
#: The head of ``render_pm_decision``'s text; a free-text PM never writes the risk-ratings line.
RENDERED_PM = re.compile(
    r"\*\*Rating\*\*: (Buy|Overweight|Hold|Underweight|Sell)\s*\n\s*\*\*Risk Ratings \(0-3\)\*\*: "
    r"volatility ([0-3]), event risk ([0-3]), valuation ([0-3]), liquidity ([0-3]), concentration ([0-3]); "
    r"main reason: (volatility|event_risk|valuation|liquidity|concentration|none)\s*$", re.MULTILINE)


def schedule(con, through: date | None = None, cycle_days: int = CYCLE_DAYS) -> tuple[list[date], set[date]]:
    """(cached dates up to ``through``, cycle dates among them); cycles count from the first cached date."""
    dates = [r[0] for r in con.execute("SELECT DISTINCT trade_date FROM current_runs ORDER BY trade_date").fetchall()]
    cycles = set(dates[cycle_days::cycle_days])
    dates = [d for d in dates if through is None or d <= through]
    return dates, cycles & set(dates)


def calibrated(con, day: date) -> bool:
    """True if the feedback setup already ran a calibration cycle on ``day``; API or key failures do not count."""
    return con.execute("SELECT 1 FROM calibration_attempts WHERE experiment_version = ? AND cycle_date = ? "
                       f"AND status NOT IN {RETRY_STATUSES}", [FEEDBACK, day]).fetchone() is not None


def pending_pairs(con, version: str, day: date, tickers=None) -> list[tuple[str, str, str]]:
    """(run_id, ticker, action) for the day's Buy/Sell proposals that ``version`` has not decided yet."""
    rows = con.execute(
        "SELECT r.run_id, r.ticker, p.action FROM current_runs r JOIN trader_proposals p ON p.run_id = r.run_id "
        "WHERE r.trade_date = ? AND p.action IN ('Buy', 'Sell') AND NOT EXISTS (SELECT 1 FROM decisions d "
        "WHERE d.experiment_version = ? AND d.ticker = r.ticker AND d.trade_date = r.trade_date) "
        "ORDER BY r.ticker", [day, version]).fetchall()
    return [r for r in rows if tickers is None or r[1] in tickers]


def late_pairs(con, dates: list[date], tickers=None) -> list[tuple[str, str, date]]:
    """(version, ticker, date) still to decide on a date a book has already passed; its fill would be lost."""
    late = []
    for version in VERSIONS:
        last = con.execute("SELECT max(snapshot_date) FROM book_snapshots WHERE experiment_version = ?",
                           [version]).fetchone()[0]
        if last is None:
            continue
        late += [(version, ticker, day) for day in dates if day < last
                 for _, ticker, _ in pending_pairs(con, version, day, tickers)]
    return late


def grid_gaps(con, dates: list[date], calendar: list[date], tickers=None) -> list[tuple[str, date]]:
    """(ticker, date) with no complete run on a trading day between the first and last cached date."""
    if not dates:
        return []
    have = set(con.execute("SELECT ticker, trade_date FROM current_runs").fetchall())
    names = tickers or sorted({t for t, _ in have})
    return [(t, d) for d in calendar if dates[0] <= d <= dates[-1] for t in names if (t, d) not in have]


def sized_decision(action: str, pm_rating: str) -> tuple[float, str, str]:
    """(final_size, verdict, rating to log) for a Trader action; a rejected trade is logged as Hold."""
    final = FINAL_SIZE[action].get(pm_rating, 0.0)
    verdict = "approved" if final == 1.0 else "reduced" if final > 0 else "rejected"
    return final, verdict, pm_rating if final > 0 else "Hold"


def portfolio_block(con, version: str, ticker: str, day: date) -> str:
    """The decision book on ``day`` as the risk team sees it: holdings, cash and drawdown."""
    row = con.execute("SELECT cash, positions, drawdown FROM book_snapshots WHERE experiment_version = ? "
                      "AND book = 'decision' AND snapshot_date <= ? ORDER BY snapshot_date DESC LIMIT 1",
                      [version, day]).fetchone()
    cash, held, drawdown = row if row else (DEFAULT.starting_cash, {}, 0.0)
    held = json.loads(held) if isinstance(held, str) else (held or {})
    book = PortfolioContext(cash=cash, currency="USD",
                            positions=[Position(ticker=t, quantity=q) for t, q in held.items() if q])
    return f"{book.render(ticker)}\n- Drawdown from peak: {drawdown:.1%}"


def load_cache(con, run_id: str, ticker: str, day: date) -> dict:
    """The run's upstream cache file: the path the database recorded, else the real cache folder."""
    row = con.execute("SELECT cache_file FROM runs WHERE run_id = ?", [run_id]).fetchone()
    path = Path(row[0]) if row and row[0] else None
    cache = (json.loads(path.read_text(encoding="utf-8")) if path and path.exists()
             else read_cache(ticker, day.isoformat()))
    if cache is None or cache["run_id"] != run_id:
        raise RuntimeError(f"cache file for {ticker} {day} is missing or not current run {run_id}")
    return cache


def run_pair(graph, checkpoint_dir, cache: dict, ticker: str, day: date, version: str,
             portfolio: str, memo: dict | None, attempt: int = 1) -> tuple[dict | None, dict]:
    """Risk debate + PM on one cached proposal; returns (PM fields, final state).

    PM fields are None when the PM fell back to free text. A thread that already finished (the
    run stopped before logging it) is not re-run; its fields are read back from the rendered PM text.
    Each attempt has its own checkpoint thread, so a retry runs the risk stage afresh.
    """
    outputs = cache["outputs"]
    state = graph.propagator.create_initial_state(
        ticker, day.isoformat(), instrument_context=outputs["instrument_context"], portfolio_context=portfolio,
        risk_memo=memo["memo_text"] if memo else "", risk_memo_id=memo["memo_id"] if memo else None)
    state.update({k: outputs[k] for k in (*REPORT_KEYS, "investment_plan", "trader_investment_plan")})
    state["investment_debate_state"].update(outputs["investment_debate"])

    args = graph.propagator.get_graph_args()
    signature = f"risk|{version}|memo={memo['memo_id'] if memo else 'none'}"
    if attempt > 1:
        signature += f"|attempt={attempt}"
    args["config"]["configurable"] = {"thread_id": thread_id(ticker, day.isoformat(), signature)}
    with get_checkpointer(checkpoint_dir, ticker) as saver, evidence.capture() as rec:
        app = graph.workflow.compile(checkpointer=saver)
        snap = app.get_state(args["config"])
        if not snap.values:
            app.update_state(args["config"], state, as_node="Trader")
        elif not snap.next:
            found = RENDERED_PM.match(snap.values.get("final_trade_decision") or "")
            if found is None:
                return None, snap.values
            logger.info("%s %s %s: PM decision recovered from finished checkpoint", version, ticker, day)
            return {"rating": found[1], **{c: int(v) for c, v in zip(CATEGORIES, found.groups()[1:6])},
                    "main_reason": found[7]}, snap.values
        for _ in app.stream(None, **args):
            pass
        final = app.get_state(args["config"]).values
    return rec.structured.get("pm_decision"), final


def advance_book(con, **kwargs) -> None:
    """Advance one book; raise when a missing price stops it (later days would see a stale book)."""
    report = advance(con, **kwargs)
    if report.stopped_on_missing_price:
        raise RuntimeError(f"{report.experiment_version} {report.book} book: {report.stopped_reason}; "
                           "fix the price data and rerun (the run resumes)")


def score_due(con, **kwargs) -> int:
    """Score due outcomes, log any that were skipped; returns the number scored."""
    report = score_pending_decisions(con, **kwargs)
    for problem in report.skipped_no_price + report.errors:
        logger.warning("outcome not scored: %s", problem)
    return report.scored


def run(db_path, graph, *, tickers=None, through: date | None = None, cycle_days: int = CYCLE_DAYS,
        checkpoint_dir=None, calendar: list[date] | None = None, price_provider=None,
        calibrate=run_cycle) -> dict:
    """Replay every cached date up to ``through`` for both setups; returns counts of what was done."""
    if calendar is None:
        from shadow.data.conditions import trading_calendar
        calendar = trading_calendar()
    days = sorted(calendar)
    checkpoint_dir = checkpoint_dir or warehouse_dir() / "risk_runner"
    sim = {"calendar": days, "price_provider": price_provider}
    evidence.install()
    done = {"decisions": 0, "pm_retries": 0, "pm_fallback_holds": 0, "cycles": 0, "outcomes_scored": 0}

    con = connect(db_path)
    try:
        check_schema_version(con, db_path)
        dates, cycles = schedule(con, through, cycle_days)
        if late := late_pairs(con, dates, tickers):
            raise RuntimeError(f"{len(late)} proposal(s) are dated before their books' last snapshot, so their "
                               f"fills would be lost (e.g. {[(v, t, d.isoformat()) for v, t, d in late[:3]]}); "
                               "run all tickers in one pass on a fresh database")
        if gaps := grid_gaps(con, dates, days, tickers):
            raise RuntimeError(f"{len(gaps)} ticker-date pair(s) have no complete upstream run "
                               f"(e.g. {[(t, d.isoformat()) for t, d in gaps[:3]]}), which would shift cycle "
                               "dates; cache them first (cache_runner --rerun-incomplete)")
        for day in dates:
            i = bisect.bisect_left(days, day)
            before = days[i - 1] if i else None
            if day in cycles and not calibrated(con, day):
                done["outcomes_scored"] += score_due(con, today=day, calendar=days, price_provider=price_provider)
                for book in ("decision", "shadow"):
                    advance_book(con, experiment_version=FEEDBACK, book=book, to_date=day,
                                 decisions_through=before, **sim)
                # The calibration runner opens its own connections to the same file.
                con.close()
                result = calibrate(as_of=day.isoformat(), experiment_version=FEEDBACK, db_path=db_path) or {}
                con = connect(db_path)
                if not calibrated(con, day):
                    # Trading on without this cycle's memo could not be undone, so stop; a rerun redoes the cycle.
                    status, error = con.execute(
                        "SELECT status, error FROM calibration_attempts WHERE experiment_version = ? AND cycle_date = ? "
                        "ORDER BY created_at DESC LIMIT 1", [FEEDBACK, day]).fetchone() or ("no attempt", None)
                    raise RuntimeError(f"calibration {day} did not run ({status}: {error}); "
                                       "check the API key or credit and rerun, this cycle will be redone")
                done["cycles"] += 1
                if result.get("status", "memo_written") == "memo_written":
                    logger.info("calibration %s: memo %s written", day, result.get("memo_id"))
                else:
                    logger.warning("calibration %s: %s, the previous memo stays current (%s)",
                                   day, result["status"], result.get("error") or "see calibration_attempts")
            for version in VERSIONS:
                advance_book(con, experiment_version=version, book="decision", to_date=day,
                             decisions_through=before, **sim)
                memo = current_memo(con, day, FEEDBACK) if version == FEEDBACK else None
                for run_id, ticker, action in pending_pairs(con, version, day, tickers):
                    portfolio = portfolio_block(con, version, ticker, day)
                    cache = load_cache(con, run_id, ticker, day)
                    for attempt in range(1, PM_ATTEMPTS + 1):
                        fields, final = run_pair(graph, checkpoint_dir, cache, ticker, day, version,
                                                 portfolio, memo, attempt)
                        if fields is not None:
                            break
                        if attempt < PM_ATTEMPTS:
                            logger.warning("%s %s %s: no structured PM decision; retrying once", version, ticker, day)
                            done["pm_retries"] += 1
                    if fields is None:
                        # The protocol: retry once, then record Hold. Ratings stay NULL (none were given).
                        logger.warning("%s %s %s: PM still unparseable; recording Hold", version, ticker, day)
                        text = final.get("final_trade_decision") or ""
                        insert_decision(con, run_id=run_id, experiment_version=version, verdict="rejected",
                                        proposed_size=1.0, final_size=0.0, risk_ratings=None, main_reason="none",
                                        explanation=f"{UNPARSED_NOTE} {text}".strip(),
                                        pm_rating="Hold", memo_id=final.get("risk_memo_id"))
                        done["pm_fallback_holds"] += 1
                        done["decisions"] += 1
                        continue
                    size, verdict, rating = sized_decision(action, fields["rating"])
                    insert_decision(con, run_id=run_id, experiment_version=version, verdict=verdict,
                                    proposed_size=1.0, final_size=size,
                                    risk_ratings={c: fields[c] for c in CATEGORIES},
                                    main_reason=fields["main_reason"], explanation=final["final_trade_decision"],
                                    pm_rating=rating, memo_id=final["risk_memo_id"])
                    done["decisions"] += 1
        if dates:
            for version in VERSIONS:
                for book in ("decision", "shadow"):
                    advance_book(con, experiment_version=version, book=book, to_date=dates[-1],
                                 decisions_through=dates[-1], **sim)
            # Outcomes are otherwise scored only on cycle dates; score all that are due by
            # today so the evaluation sees the last ~10 trading days too.
            done["outcomes_scored"] += score_due(con, calendar=days, price_provider=price_provider)
    finally:
        con.close()
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay cached proposals through the risk debate and PM.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true", help="use warehouse/dev.duckdb")
    parser.add_argument("--tickers", nargs="+", help="only these tickers (default: all cached)")
    parser.add_argument("--through", type=date.fromisoformat, help="stop after this date (YYYY-MM-DD)")
    parser.add_argument("--cycle-days", type=int, default=CYCLE_DAYS, help="cached dates per calibration cycle")
    parser.add_argument("--max-pairs", type=int, default=1,
                        help="refuse to start more risk-stage LLM runs than this (default 1)")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="refuse to start more calibration cycles than this (default 0)")
    parser.add_argument("--dry-run", action="store_true", help="only report what would run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tickers = [t.upper() for t in args.tickers] if args.tickers else None

    from shadow.data.conditions import trading_calendar
    calendar = trading_calendar()
    path = resolve_db(args.db, args.dev)
    con = connect(path, read_only=True)
    try:
        check_schema_version(con, path)
        dates, cycles = schedule(con, args.through, args.cycle_days)
        pairs = sum(len(pending_pairs(con, v, d, tickers)) for d in dates for v in VERSIONS)
        due = sorted(d for d in cycles if not calibrated(con, d))
        late = late_pairs(con, dates, tickers)
        gaps = grid_gaps(con, dates, calendar, tickers)
    finally:
        con.close()
    print(f"database: {path}")
    print(f"dates: {len(dates)} ({dates[0] if dates else '-'}..{dates[-1] if dates else '-'})")
    print(f"risk-stage runs to do: {pairs}; calibration cycles to do: {len(due)} {[d.isoformat() for d in due]}")
    if late:
        print(f"refusing: {len(late)} proposal(s) are dated before their books' last snapshot, so their fills "
              f"would be lost (e.g. {[(v, t, d.isoformat()) for v, t, d in late[:3]]}); "
              "run all tickers in one pass on a fresh database")
        return 2
    if gaps:
        print(f"refusing: {len(gaps)} ticker-date pair(s) have no complete upstream run "
              f"(e.g. {[(t, d.isoformat()) for t, d in gaps[:3]]}), which would shift cycle dates; "
              "cache them first (cache_runner --rerun-incomplete)")
        return 2
    if args.dry_run:
        return 0
    if pairs > args.max_pairs or len(due) > args.max_cycles:
        print(f"refusing: limits are --max-pairs {args.max_pairs} and --max-cycles {args.max_cycles}")
        return 2

    from shadow.data.settings import ANALYSTS, shadow_config
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    config = shadow_config()
    if config["llm_provider"] != "openrouter":
        raise RuntimeError(f"llm_provider is {config['llm_provider']!r}; expected openrouter (check research_agent/.env)")
    usage = UsageCounter()
    graph = TradingAgentsGraph(selected_analysts=ANALYSTS, config=config, callbacks=[usage])
    spend_before = openrouter_spend()
    done = run(path, graph, tickers=tickers, through=args.through, cycle_days=args.cycle_days, calendar=calendar)
    spent = spend_since(spend_before)
    print(f"done: {done}; risk-stage usage {usage.as_dict()}")
    print(f"OpenRouter spend: {f'${spent:.4f}' if spent is not None else 'unknown'} (key total, includes calibration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
