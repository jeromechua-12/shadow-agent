"""Named insert helpers for teammates' components. Each validates, derives what it can, and appends.

    insert_decision       decision logger        one PM decision -> its approved / rejected parts
    insert_outcome        outcome judge          10-day result; score and label are computed here
    insert_book_snapshot  two-book simulator     one day of one book; drawdown computed here
    insert_memo           Calibration Agent      memo + advice + hypotheses + checks + tool calls

Every problem with a record is reported at once in an ``InvalidRecord`` error,
and nothing is written unless the whole record is valid. Rows are only ever
added (see the README for why).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import duckdb

from shadow.data.loader import append_rows

CATEGORIES = ("volatility", "event_risk", "valuation", "liquidity", "concentration")
RATING_COLUMNS = {c: ("risk_event" if c == "event_risk" else f"risk_{c}") for c in CATEGORIES}
DIRECTION = {"Buy": 1, "Hold": 0, "Sell": -1}
NOISE_BAND = 0.01
HORIZON_DAYS = 10
MAX_DRAWDOWN_FOR_MORE_RISK = -0.10
DEFAULT_RISK_LEVEL = 3          # every experiment version's first memo
RECENT_SNAPSHOT_DAYS = 7        # a risk increase needs a decision-book snapshot this fresh


class InvalidRecord(ValueError):
    """A record failed validation; ``problems`` lists every reason."""

    def __init__(self, what: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"invalid {what}:\n  - " + "\n  - ".join(problems))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _as_date(value, name: str, problems: list[str]) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        problems.append(f"{name} must be a date like '2026-09-01', got {value!r}")
        return None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def label_for(part: str, score: float) -> str:
    """The outcome label: noise within ±1%, else by part and the sign of the score."""
    if abs(score) <= NOISE_BAND:
        return "noise"
    if part == "approved_part":
        return "correct_approval" if score > 0 else "approval_loss"
    return "missed_gain" if score > 0 else "correct_rejection"


# ---- decisions -------------------------------------------------------------------

def insert_decision(
    con: duckdb.DuckDBPyConnection, *, run_id: str, experiment_version: str, verdict: str,
    proposed_size: float, final_size: float, risk_ratings: dict, main_reason: str,
    explanation: str = "", pm_rating: str | None = None, memo_id: str | None = None,
    decided_at: str | None = None,
) -> list[str]:
    """Log one Portfolio Manager decision on an upstream run's Trader proposal.

    Writes one row per part: an approved part of ``final_size`` and/or a rejected
    part of ``proposed_size - final_size``. Ticker, trade date, proposed action
    and direction come from the run itself. Returns the new decision_ids.

    risk_ratings: {"volatility": 0-3, "event_risk": 0-3, "valuation": 0-3,
                   "liquidity": 0-3, "concentration": 0-3}
    """
    problems: list[str] = []
    run = con.execute("SELECT r.ticker, r.trade_date, p.action FROM runs r "
                      "LEFT JOIN trader_proposals p ON p.run_id = r.run_id WHERE r.run_id = ?", [run_id]).fetchone()
    if run is None:
        raise InvalidRecord("decision", [f"run_id {run_id!r} is not in runs (load the upstream cache first)"])
    ticker, trade_date, action = run
    current = con.execute("SELECT run_id FROM current_runs WHERE ticker = ? AND trade_date = ?",
                          [ticker, trade_date]).fetchone()
    if current is None or current[0] != run_id:
        raise InvalidRecord("decision", [
            f"run {run_id} is not the current run for {ticker} {trade_date} "
            f"(current: {current[0] if current else 'none complete'}); decide only on current_runs"])
    existing = con.execute("SELECT DISTINCT run_id FROM decisions WHERE experiment_version = ? AND ticker = ? "
                           "AND trade_date = ?", [experiment_version, ticker, trade_date]).fetchall()
    if existing:
        raise InvalidRecord("decision", [
            f"{experiment_version!r} already has a decision set for {ticker} {trade_date} "
            f"(run {existing[0][0]}); one per version, ticker and date"])
    if action not in DIRECTION:
        problems.append(f"run {run_id} has no structured Trader action ({action!r}); nothing to decide on")
    elif action == "Hold":
        problems.append("the Trader proposed Hold; Holds are not logged as decisions (open question for the group)")
    if not isinstance(experiment_version, str) or not experiment_version:
        problems.append("experiment_version is required")
    if verdict not in ("approved", "reduced", "rejected"):
        problems.append(f"verdict must be approved, reduced or rejected, got {verdict!r}")
    if not _is_number(proposed_size) or proposed_size <= 0:
        problems.append(f"proposed_size must be a number > 0 (fraction of the per-ticker allocation), got {proposed_size!r}")
    if not _is_number(final_size) or final_size < 0:
        problems.append(f"final_size must be a number >= 0, got {final_size!r}")
    if not problems:
        eps = 1e-9
        expected = ("approved" if abs(final_size - proposed_size) <= eps else
                    "rejected" if final_size <= eps else
                    "reduced" if final_size < proposed_size else None)
        if expected is None:
            problems.append(f"final_size {final_size} is larger than proposed_size {proposed_size}")
        elif expected != verdict:
            problems.append(f"verdict {verdict!r} does not match sizes {proposed_size} -> {final_size} "
                            f"(that is {expected!r})")
    ratings = risk_ratings if isinstance(risk_ratings, dict) else {}
    if set(ratings) != set(CATEGORIES):
        problems.append(f"risk_ratings needs exactly these keys: {list(CATEGORIES)}; got {sorted(ratings)}")
    for cat, v in ratings.items():
        if cat in CATEGORIES and not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 3):
            problems.append(f"risk_ratings[{cat!r}] must be an integer 0-3, got {v!r}")
    if main_reason not in (*CATEGORIES, "none"):
        problems.append(f"main_reason must be one of {[*CATEGORIES, 'none']}, got {main_reason!r}")
    if pm_rating is not None and pm_rating not in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
        problems.append(f"pm_rating must be a 5-tier rating, got {pm_rating!r}")
    if memo_id is not None:
        memo = con.execute("SELECT cycle_date, experiment_version FROM memos WHERE memo_id = ?", [memo_id]).fetchone()
        if memo is None:
            problems.append(f"memo_id {memo_id!r} is not in memos")
        elif memo[0] > trade_date or memo[1] != experiment_version:
            problems.append(f"memo {memo_id} (cycle {memo[0]}, {memo[1]}) cannot be in force for a "
                            f"{experiment_version} trade on {trade_date}")
    if problems:
        raise InvalidRecord("decision", problems)

    base = {"run_id": run_id, "experiment_version": experiment_version, "memo_id": memo_id, "ticker": ticker,
            "trade_date": trade_date, "decided_at": decided_at or _now(), "verdict": verdict,
            "proposed_action": action, "direction": DIRECTION[action], "proposed_size": float(proposed_size),
            "final_size": float(final_size), "pm_rating": pm_rating, "main_reason": main_reason,
            "explanation": explanation, **{RATING_COLUMNS[c]: v for c, v in ratings.items()}}
    parts = []
    if final_size > 1e-9:
        parts.append({**base, "decision_id": _new_id("dec"), "part": "approved_part", "part_size": float(final_size)})
    if proposed_size - final_size > 1e-9:
        parts.append({**base, "decision_id": _new_id("dec"), "part": "rejected_part",
                      "part_size": float(proposed_size - final_size)})
    append_rows(con, "decisions", parts)
    return [p["decision_id"] for p in parts]


# ---- outcomes --------------------------------------------------------------------

def outcome_dates(trade_date: date, calendar) -> tuple[date, date | None]:
    """(entry_date, resolved_on) on the trading calendar; resolved_on is None if not reached yet.

    entry_date is the first trading day after the trade date (next-day open fill);
    resolved_on is the HORIZON_DAYS-th trading day after the entry.
    """
    days = sorted(calendar)
    later = [d for d in days if d > trade_date]
    if not later:
        raise InvalidRecord("outcome", [f"the trading calendar has no day after {trade_date}"])
    entry = later[0]
    i = days.index(entry) + HORIZON_DAYS
    return entry, (days[i] if i < len(days) else None)


def insert_outcome(
    con: duckdb.DuckDBPyConnection, *, decision_id: str, stock_return: float, spy_return: float,
    entry_date=None, resolved_on=None, horizon_days: int = HORIZON_DAYS, calendar=None,
) -> dict:
    """Record a decision part's 10-day result. Dates, score and label are fixed here.

    entry_date:  the first trading day after the trade date (next-day open fill)
    resolved_on: the 10th trading day after the entry, when the result is known
    Both come from the trading calendar (SPY's trading days by default, or the
    ``calendar`` given); pass them only to have them checked. Returns
    {"entry_date", "resolved_on", "score", "label"}.
    """
    problems: list[str] = []
    dec = con.execute("SELECT trade_date, direction, part FROM decisions WHERE decision_id = ?", [decision_id]).fetchone()
    if dec is None:
        raise InvalidRecord("outcome", [f"decision_id {decision_id!r} is not in decisions"])
    trade_date, direction, part = dec
    if horizon_days != HORIZON_DAYS:
        problems.append(f"horizon_days must be {HORIZON_DAYS} (the project's outcome horizon), got {horizon_days!r}")
    for name, v in (("stock_return", stock_return), ("spy_return", spy_return)):
        if not _is_number(v) or v != v:
            problems.append(f"{name} must be a number (e.g. 0.031 for +3.1%), got {v!r}")
    if calendar is None:
        from shadow.data.conditions import trading_calendar
        calendar = trading_calendar()
    entry, resolved = outcome_dates(trade_date, [_as_date(d, "calendar day", problems) for d in calendar])
    if resolved is None:
        problems.append(f"not resolved yet: the calendar does not reach {HORIZON_DAYS} trading days after {entry}")
    if entry_date is not None and _as_date(entry_date, "entry_date", problems) not in (None, entry):
        problems.append(f"entry_date {entry_date} must be the next trading day after {trade_date}, i.e. {entry}")
    if resolved_on is not None and resolved is not None \
            and _as_date(resolved_on, "resolved_on", problems) not in (None, resolved):
        problems.append(f"resolved_on {resolved_on} must be the {HORIZON_DAYS}th trading day after entry "
                        f"{entry}, i.e. {resolved}")
    if resolved is not None and resolved > date.today():
        problems.append(f"resolved_on {resolved} is in the future")
    if problems:
        raise InvalidRecord("outcome", problems)
    score = direction * (stock_return - spy_return)
    label = label_for(part, score)
    append_rows(con, "outcomes", [{"decision_id": decision_id, "horizon_days": HORIZON_DAYS, "entry_date": entry,
                                   "resolved_on": resolved, "stock_return": float(stock_return),
                                   "spy_return": float(spy_return), "score": score, "label": label}])
    return {"entry_date": entry.isoformat(), "resolved_on": resolved.isoformat(), "score": score, "label": label}


# ---- book snapshots ----------------------------------------------------------------

def insert_book_snapshot(
    con: duckdb.DuckDBPyConnection, *, experiment_version: str, book: str, snapshot_date,
    cash: float, value: float, positions: dict | None = None,
) -> float:
    """Record one day of one book. Drawdown is computed from the book's running peak; returned."""
    problems: list[str] = []
    day = _as_date(snapshot_date, "snapshot_date", problems)
    if book not in ("decision", "shadow"):
        problems.append(f"book must be 'decision' or 'shadow', got {book!r}")
    if not isinstance(experiment_version, str) or not experiment_version:
        problems.append("experiment_version is required")
    for name, v in (("cash", cash), ("value", value)):
        if not _is_number(v) or v < 0:
            problems.append(f"{name} must be a number >= 0, got {v!r}")
    if positions is not None and not isinstance(positions, dict):
        problems.append("positions must be a dict {ticker: shares}")
    if problems:
        raise InvalidRecord("book snapshot", problems)
    later = con.execute("SELECT max(snapshot_date) FROM book_snapshots WHERE experiment_version = ? AND book = ? "
                        "AND snapshot_date >= ?", [experiment_version, book, day]).fetchone()[0]
    if later is not None:
        raise InvalidRecord("book snapshot", [f"{book} book already has a snapshot on or after {day} ({later}); "
                                              "snapshots must be added in date order"])
    peak = con.execute("SELECT max(value) FROM book_snapshots WHERE experiment_version = ? AND book = ?",
                       [experiment_version, book]).fetchone()[0]
    peak = max(peak or 0.0, value)
    drawdown = value / peak - 1 if peak > 0 else 0.0
    append_rows(con, "book_snapshots", [{"snapshot_date": day, "book": book, "experiment_version": experiment_version,
                                         "cash": float(cash), "value": float(value), "drawdown": drawdown,
                                         "positions": positions or {}}])
    return drawdown


# ---- memos -------------------------------------------------------------------------

def insert_memo(
    con: duckdb.DuckDBPyConnection, *, experiment_version: str, cycle_date, risk_level: int, memo_text: str,
    advice: dict, safety_passed: bool, safety_flags: dict | None = None, model: str | None = None,
    hypotheses: list | None = None, hypothesis_checks: list | None = None, tool_calls: list | None = None,
) -> dict:
    """Record a Calibration Agent memo and everything behind it, in one transaction.

    advice:            {category: {"advice": tighten|loosen|keep, "confidence": low|medium|high,
                                   "reasoning": str, "cited_ids": [str, ...]}}  (all five categories)
    hypotheses:        [{"statement": str, "category": str|None, "subgroup": {filters given to subgroup_stats}}]
    hypothesis_checks: [{"hypothesis_id": str, "status": pending|held|failed, "evidence": {...}}]
    tool_calls:        [{"tool": str, "args": {...}, "result": {...}, "called_at": iso timestamp}]

    Hard limits are enforced: risk level 1-5, at most one step from the previous
    memo, and no step up while the decision book is more than 10% below its peak.
    Returns {"memo_id": ..., "hypothesis_ids": [...]}.
    """
    problems: list[str] = []
    cycle = _as_date(cycle_date, "cycle_date", problems)
    if not isinstance(experiment_version, str) or not experiment_version:
        problems.append("experiment_version is required")
    if not (isinstance(risk_level, int) and not isinstance(risk_level, bool) and 1 <= risk_level <= 5):
        problems.append(f"risk_level must be an integer 1-5, got {risk_level!r}")
    if not isinstance(memo_text, str) or not memo_text.strip():
        problems.append("memo_text is required (it goes into the risk team's prompts)")
    advice = advice if isinstance(advice, dict) else {}
    if set(advice) != set(CATEGORIES):
        problems.append(f"advice needs exactly these categories: {list(CATEGORIES)}; got {sorted(advice)}")
    for cat, a in advice.items():
        if not isinstance(a, dict):
            problems.append(f"advice[{cat!r}] must be a dict")
            continue
        if a.get("advice") not in ("tighten", "loosen", "keep"):
            problems.append(f"advice[{cat!r}]['advice'] must be tighten, loosen or keep")
        if a.get("confidence") not in ("low", "medium", "high"):
            problems.append(f"advice[{cat!r}]['confidence'] must be low, medium or high")
        if not str(a.get("reasoning") or "").strip():
            problems.append(f"advice[{cat!r}]['reasoning'] is required")
    for i, h in enumerate(hypotheses or []):
        if not str(h.get("statement") or "").strip() or not isinstance(h.get("subgroup"), dict):
            problems.append(f"hypotheses[{i}] needs a 'statement' and a 'subgroup' dict")
    for i, c in enumerate(hypothesis_checks or []):
        if c.get("status") not in ("pending", "held", "failed"):
            problems.append(f"hypothesis_checks[{i}]['status'] must be pending, held or failed")
        elif not con.execute("SELECT 1 FROM memo_hypotheses WHERE hypothesis_id = ?", [c.get("hypothesis_id")]).fetchone():
            problems.append(f"hypothesis_checks[{i}]: unknown hypothesis_id {c.get('hypothesis_id')!r}")
    for i, t in enumerate(tool_calls or []):
        if t.get("tool") not in ("cycle_summary", "search_trades", "read_trade", "subgroup_stats", "past_memos"):
            problems.append(f"tool_calls[{i}]['tool'] must be one of the five calibration tools")
    if problems:
        raise InvalidRecord("memo", problems)

    prev = con.execute("SELECT risk_level, cycle_date FROM memos WHERE experiment_version = ? AND cycle_date < ? "
                       "ORDER BY cycle_date DESC LIMIT 1", [experiment_version, cycle]).fetchone()
    previous_level = prev[0] if prev else None
    if con.execute("SELECT 1 FROM memos WHERE experiment_version = ? AND cycle_date >= ?",
                   [experiment_version, cycle]).fetchone():
        problems.append(f"{experiment_version} already has a memo on or after {cycle}; memos are added in date order")
    if previous_level is None and risk_level != DEFAULT_RISK_LEVEL:
        problems.append(f"the first memo of {experiment_version!r} must start at risk level {DEFAULT_RISK_LEVEL}, "
                        f"got {risk_level}")
    if previous_level is not None and abs(risk_level - previous_level) > 1:
        problems.append(f"risk level may move one step per cycle: {previous_level} -> {risk_level}")
    if previous_level is not None and risk_level > previous_level:
        snap = con.execute("SELECT snapshot_date, drawdown FROM book_snapshots WHERE experiment_version = ? "
                           "AND book = 'decision' AND snapshot_date <= ? ORDER BY snapshot_date DESC LIMIT 1",
                           [experiment_version, cycle]).fetchone()
        if snap is None or (cycle - snap[0]).days > RECENT_SNAPSHOT_DAYS:
            # Without a fresh book value the 10% drawdown limit cannot be checked, so no increase.
            problems.append(f"no decision-book snapshot in the {RECENT_SNAPSHOT_DAYS} days before {cycle} "
                            f"(latest: {snap[0] if snap else 'none'}); cannot raise risk")
        elif snap[1] < MAX_DRAWDOWN_FOR_MORE_RISK:
            problems.append(f"no extra risk after a drawdown beyond 10% (decision book at {snap[1]:.1%})")
    for i, c in enumerate(hypothesis_checks or []):
        origin = con.execute("SELECT m.experiment_version, m.cycle_date FROM memo_hypotheses h "
                             "JOIN memos m ON m.memo_id = h.memo_id WHERE h.hypothesis_id = ?",
                             [c["hypothesis_id"]]).fetchone()
        if origin[0] != experiment_version:
            problems.append(f"hypothesis_checks[{i}]: {c['hypothesis_id']} belongs to experiment version "
                            f"{origin[0]!r}, not {experiment_version!r}")
        elif origin[1] >= cycle:
            problems.append(f"hypothesis_checks[{i}]: {c['hypothesis_id']} was proposed on {origin[1]}; "
                            f"a check must come from a later cycle than that")
    if problems:
        raise InvalidRecord("memo", problems)

    memo_id = _new_id("memo")
    hyp_rows = [{"hypothesis_id": _new_id("hyp"), "memo_id": memo_id, "category": h.get("category"),
                 "statement": h["statement"], "subgroup": h["subgroup"]} for h in hypotheses or []]
    con.begin()
    try:
        append_rows(con, "memos", [{"memo_id": memo_id, "experiment_version": experiment_version, "cycle_date": cycle,
                                    "created_at": _now(), "risk_level": risk_level,
                                    "previous_risk_level": previous_level, "memo_text": memo_text, "model": model,
                                    "safety_passed": bool(safety_passed), "safety_flags": safety_flags or {}}])
        append_rows(con, "memo_advice", [{"memo_id": memo_id, "category": cat, "advice": a["advice"],
                                          "confidence": a["confidence"], "reasoning": a["reasoning"],
                                          "cited_ids": [str(x) for x in a.get("cited_ids") or []]}
                                         for cat, a in advice.items()])
        append_rows(con, "memo_hypotheses", hyp_rows)
        append_rows(con, "hypothesis_checks", [{"hypothesis_id": c["hypothesis_id"], "memo_id": memo_id,
                                                "status": c["status"], "evidence": c.get("evidence") or {}}
                                               for c in hypothesis_checks or []])
        append_rows(con, "memo_tool_calls", [{"memo_id": memo_id, "seq": i, "tool": t["tool"],
                                              "args": t.get("args") or {}, "result": t.get("result"),
                                              "called_at": t.get("called_at") or _now()}
                                             for i, t in enumerate(tool_calls or [])])
        con.commit()
    except Exception:
        con.rollback()
        raise
    return {"memo_id": memo_id, "hypothesis_ids": [h["hypothesis_id"] for h in hyp_rows]}
