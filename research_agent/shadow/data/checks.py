"""Data quality checks over the evidence database.

Each check returns how many rows break a rule and a few example IDs. Severity:
``error`` means the data is wrong or leaks the future and must be fixed before
an experiment uses it; ``warning`` means a run or record is weaker than it
should be (missing source, empty reply) and may need a rerun.

Usage:
    python -m shadow.data.checks           # prints a table, exits 1 if any error is found
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

from shadow.data.paths import warehouse_dir

NOISE_BAND = 0.01
TOLERANCE = 1e-9


@dataclass
class Finding:
    check: str
    severity: str  # error | warning
    description: str
    count: int
    examples: list = field(default_factory=list)


# name -> (severity, description, SQL returning one example id per offending row)
SQL_CHECKS = {
    "incomplete_runs": ("warning", "upstream runs marked incomplete (empty report or fixable data failure)",
        "SELECT run_id FROM runs WHERE NOT complete"),
    "retryable_data_failures": ("warning", "data calls that failed for a reason a rerun could fix",
        "SELECT method || ' ' || status || ' ' || run_id FROM evidence_calls WHERE status <> 'ok' AND retryable"),
    "llm_hit_token_cap": ("warning", "LLM replies cut off by the token cap (finish_reason = length)",
        "SELECT run_id || ' ' || node FROM llm_calls WHERE finish_reason = 'length'"),
    "news_after_trade_date": ("error", "a run was shown a news item created after its trade date (look-ahead)",
        """SELECT rti.run_id || ' ' || rti.item_id FROM run_text_items rti
           JOIN text_items ti ON ti.item_id = rti.item_id JOIN runs r ON r.run_id = rti.run_id
           WHERE ti.created_at >= CAST(r.trade_date + 1 AS TIMESTAMPTZ)"""),
    "evidence_date_mismatch": ("error", "a data call's as_of differs from its run's trade date",
        "SELECT e.call_id FROM evidence_calls e JOIN runs r ON r.run_id = e.run_id WHERE e.as_of <> r.trade_date"),
    "evidence_fetched_before_trade_date": ("warning", "a data call ran before the date it analyses (clock or loader bug)",
        "SELECT e.call_id FROM evidence_calls e WHERE CAST(e.fetched_at AS DATE) < e.as_of"),
    "trader_numbers_missing": ("warning", "Trader fell back to free text, or a Buy/Sell has no entry price",
        "SELECT run_id FROM trader_proposals WHERE NOT structured_ok OR (action IN ('Buy','Sell') AND entry_price IS NULL)"),
    "stop_on_wrong_side": ("warning", "stop-loss above the entry for a Buy, or below it for a Sell",
        """SELECT run_id FROM trader_proposals WHERE stop_loss IS NOT NULL AND entry_price IS NOT NULL
           AND ((action = 'Buy' AND stop_loss >= entry_price) OR (action = 'Sell' AND stop_loss <= entry_price))"""),
    "decision_without_run": ("error", "decision ticker or date differs from the upstream run it judged",
        """SELECT d.decision_id FROM decisions d JOIN runs r ON r.run_id = d.run_id
           WHERE d.ticker <> r.ticker OR d.trade_date <> r.trade_date"""),
    "duplicate_decision_parts": ("error", "more than one row for the same run, experiment and part",
        """SELECT run_id || ' ' || experiment_version || ' ' || part FROM decisions
           GROUP BY run_id, experiment_version, part HAVING count(*) > 1"""),
    "part_sizes_do_not_add_up": ("error", "approved part is not final_size, or parts do not sum to proposed_size",
        """SELECT run_id || ' ' || experiment_version FROM decisions GROUP BY run_id, experiment_version
           HAVING abs(sum(part_size) - any_value(proposed_size)) > 1e-9
               OR abs(coalesce(sum(part_size) FILTER (WHERE part = 'approved_part'), 0) - any_value(final_size)) > 1e-9
               OR count(DISTINCT proposed_size) > 1 OR count(DISTINCT final_size) > 1"""),
    "verdict_inconsistent_with_sizes": ("error", "approved needs final = proposed, rejected final = 0, reduced in between",
        """SELECT decision_id FROM decisions WHERE
              (verdict = 'approved' AND abs(final_size - proposed_size) > 1e-9)
           OR (verdict = 'rejected' AND abs(final_size) > 1e-9)
           OR (verdict = 'reduced' AND NOT (final_size > 1e-9 AND final_size < proposed_size - 1e-9))"""),
    "outcome_not_after_trade": ("error", "outcome resolved or entered on or before the trade date",
        """SELECT o.decision_id FROM outcomes o JOIN decisions d ON d.decision_id = o.decision_id
           WHERE o.resolved_on <= d.trade_date OR o.entry_date <= d.trade_date OR o.resolved_on < o.entry_date"""),
    "outcome_score_mismatch": ("error", "score is not direction x (stock return - SPY return)",
        f"""SELECT o.decision_id FROM outcomes o JOIN decisions d ON d.decision_id = o.decision_id
            WHERE abs(o.score - d.direction * (o.stock_return - o.spy_return)) > {TOLERANCE}"""),
    "outcome_label_mismatch": ("error", f"label does not follow from part and score (noise band ±{NOISE_BAND:.0%})",
        f"""SELECT o.decision_id FROM outcomes o JOIN decisions d ON d.decision_id = o.decision_id
            WHERE o.label <> CASE
              WHEN abs(o.score) <= {NOISE_BAND} THEN 'noise'
              WHEN d.part = 'approved_part' AND o.score > 0 THEN 'correct_approval'
              WHEN d.part = 'approved_part' THEN 'approval_loss'
              WHEN o.score > 0 THEN 'missed_gain'
              ELSE 'correct_rejection' END"""),
    "drawdown_above_zero": ("error", "book drawdown must be <= 0",
        "SELECT experiment_version || ' ' || book || ' ' || snapshot_date FROM book_snapshots WHERE drawdown > 1e-12"),
    "risk_level_jump": ("error", "risk level moved more than one step in a cycle",
        "SELECT memo_id FROM memos WHERE previous_risk_level IS NOT NULL AND abs(risk_level - previous_risk_level) > 1"),
    "decision_memo_from_future": ("error", "a decision used a memo whose cycle date is after the trade date",
        """SELECT d.decision_id FROM decisions d JOIN memos m ON m.memo_id = d.memo_id
           WHERE m.cycle_date > d.trade_date OR m.experiment_version <> d.experiment_version"""),
}

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _check_args_after_trade_date(con) -> Finding:
    """Any date argument a data call received that is after its run's trade date."""
    bad = []
    for call_id, args, as_of in con.execute("SELECT call_id, CAST(args AS VARCHAR), as_of FROM evidence_calls").fetchall():
        if any(d > as_of.isoformat() for d in _ISO_DATE.findall(args or "")):
            bad.append(call_id)
    return Finding("data_call_asked_for_future", "error",
                   "a data call was made with a date argument after the run's trade date", len(bad), bad[:5])


def _check_payloads(con, warehouse: Path) -> Finding:
    """Every saved payload exists and matches its SHA-256."""
    bad = []
    rows = con.execute("SELECT DISTINCT payload_path, payload_sha256 FROM evidence_calls "
                       "WHERE payload_path IS NOT NULL").fetchall()
    for rel, sha in rows:
        path = warehouse / "raw" / rel
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            bad.append(rel)
    return Finding("payload_missing_or_altered", "error",
                   "a saved data payload is missing or no longer matches its hash", len(bad), bad[:5])


def run_checks(con: duckdb.DuckDBPyConnection, warehouse: Path | None = None) -> list[Finding]:
    findings = []
    for name, (severity, description, sql) in SQL_CHECKS.items():
        ids = [r[0] for r in con.execute(sql).fetchall()]
        findings.append(Finding(name, severity, description, len(ids), ids[:5]))
    findings.append(_check_args_after_trade_date(con))
    findings.append(_check_payloads(con, Path(warehouse or warehouse_dir())))
    return findings


def overview(con: duckdb.DuckDBPyConnection) -> dict:
    """Row counts and the evidence status mix, for a quick look at what is loaded."""
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                        "WHERE table_type = 'BASE TABLE' ORDER BY table_name").fetchall()]
    # Table names come from the catalogue, not from input.
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    status = con.execute("SELECT method, status, count(*) FROM evidence_calls GROUP BY ALL ORDER BY ALL").fetchall()
    return {"rows": counts, "evidence_status": [{"method": m, "status": s, "calls": n} for m, s, n in status]}


def main(argv: list[str] | None = None) -> int:
    from shadow.data.db import connect

    parser = argparse.ArgumentParser(description="Run data quality checks.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    con = connect(args.db, read_only=True)
    findings = run_checks(con)
    if args.json:
        print(json.dumps({"overview": overview(con), "findings": [asdict(f) for f in findings]}, indent=1, default=str))
    else:
        for f in findings:
            mark = "OK  " if f.count == 0 else ("FAIL" if f.severity == "error" else "WARN")
            print(f"{mark} {f.check:36} {f.count:>5}  {f.description}")
            for ex in f.examples:
                print(f"       e.g. {ex}")
    return 1 if any(f.count and f.severity == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
