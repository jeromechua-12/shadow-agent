"""Display a saved Calibration Agent memo and its structured category advice."""

from __future__ import annotations

import argparse
import json


def _decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def fetch_memo(con, *, memo_id: str | None = None,
               experiment_version: str | None = None, cycle_date: str | None = None) -> dict | None:
    """Fetch one memo by ID or the latest (optionally date-specific) memo for a version."""
    if memo_id:
        row = con.execute(
            "SELECT memo_id, experiment_version, cycle_date, risk_level, previous_risk_level, "
            "risk_level_rationale, memo_text, model, safety_passed, safety_flags "
            "FROM memos WHERE memo_id = ?", [memo_id]
        ).fetchone()
    elif experiment_version and cycle_date:
        row = con.execute(
            "SELECT memo_id, experiment_version, cycle_date, risk_level, previous_risk_level, "
            "risk_level_rationale, memo_text, model, safety_passed, safety_flags "
            "FROM memos WHERE experiment_version = ? AND cycle_date = ? "
            "ORDER BY created_at DESC LIMIT 1", [experiment_version, cycle_date]
        ).fetchone()
    elif experiment_version:
        row = con.execute(
            "SELECT memo_id, experiment_version, cycle_date, risk_level, previous_risk_level, "
            "risk_level_rationale, memo_text, model, safety_passed, safety_flags "
            "FROM memos WHERE experiment_version = ? "
            "ORDER BY cycle_date DESC, created_at DESC LIMIT 1", [experiment_version]
        ).fetchone()
    else:
        raise ValueError("provide memo_id or experiment_version")

    if row is None:
        return None
    columns = ("memo_id", "experiment_version", "cycle_date", "risk_level", "previous_risk_level",
               "risk_level_rationale", "memo_text", "model", "safety_passed", "safety_flags")
    memo = dict(zip(columns, row))
    memo["safety_flags"] = _decode(memo["safety_flags"]) or {}
    advice = []
    for advice_row in con.execute(
        "SELECT category, advice, confidence, reasoning, cited_ids, conditions, supporting_ids, "
        "contradicting_ids, watch_next FROM memo_advice WHERE memo_id = ? ORDER BY category",
        [memo["memo_id"]],
    ).fetchall():
        advice_columns = ("category", "advice", "confidence", "reasoning", "cited_ids", "conditions",
                          "supporting_ids", "contradicting_ids", "watch_next")
        item = dict(zip(advice_columns, advice_row))
        for key in ("conditions", "watch_next"):
            item[key] = _decode(item[key]) or []
        for key in ("cited_ids", "supporting_ids", "contradicting_ids"):
            item[key] = list(item[key] or [])
        advice.append(item)
    return {"memo": memo, "advice": advice}


def render(report: dict) -> str:
    memo = report["memo"]
    lines = [
        f"Calibration memo {memo['memo_id']}",
        f"Experiment: {memo['experiment_version']}  Cycle: {memo['cycle_date']}",
        f"Risk level: {memo['risk_level']}  Previous: {memo['previous_risk_level'] or 'none'}",
        f"Safety passed: {memo['safety_passed']}",
    ]
    if memo.get("risk_level_rationale"):
        lines.extend([f"Risk rationale: {memo['risk_level_rationale']}", ""])
    lines.extend(["Memo:", memo["memo_text"], "", "Structured category advice:"])
    for item in report["advice"]:
        lines.extend([
            f"\n{item['category'].replace('_', ' ').title()}: {item['advice']} "
            f"({item['confidence']} confidence)",
            f"Reasoning: {item['reasoning']}",
        ])
        for label, key in (("Conditions", "conditions"), ("Supporting IDs", "supporting_ids"),
                           ("Contradicting IDs", "contradicting_ids"), ("Watch next", "watch_next")):
            if item[key]:
                lines.append(f"{label}: " + "; ".join(str(value) for value in item[key]))
    flags = memo.get("safety_flags") or {}
    review_flags = flags.get("flags", []) if isinstance(flags, dict) else []
    if review_flags:
        lines.extend(["", "Review flags:"])
        lines.extend(f"- {flag.get('code', 'review')}: {flag.get('message', '')}" for flag in review_flags)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from shadow.data.db import check_schema_version, connect, resolve_db

    parser = argparse.ArgumentParser(description="Display a saved Calibration Agent memo and category advice.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--memo-id", help="display this exact memo")
    selector.add_argument("--experiment-version", help="display the latest memo for this experiment")
    parser.add_argument("--cycle-date", help="select one cycle date (requires --experiment-version)")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true", help="use warehouse/dev.duckdb")
    parser.add_argument("--json", action="store_true", help="print the memo and advice as JSON")
    args = parser.parse_args(argv)
    if args.cycle_date and not args.experiment_version:
        parser.error("--cycle-date requires --experiment-version")

    path = resolve_db(args.db, args.dev)
    con = connect(path, read_only=True)
    try:
        check_schema_version(con, path)
        report = fetch_memo(con, memo_id=args.memo_id,
                            experiment_version=args.experiment_version, cycle_date=args.cycle_date)
    finally:
        con.close()
    if report is None:
        selector_text = (f"memo_id={args.memo_id}" if args.memo_id else
                         f"experiment_version={args.experiment_version}, cycle_date={args.cycle_date or 'latest'}")
        print(f"No memo found for {selector_text} in {path}")
        return 1
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
