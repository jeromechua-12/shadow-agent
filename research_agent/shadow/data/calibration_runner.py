"""Run a three-attempt, auditable Calibration Agent cycle."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone

from shadow.data.calibration_agent import (
    AttemptResult,
    CALIBRATION_MODEL,
    build_calibration_llm,
    run_attempt,
)
from shadow.data.calibration_models import CATEGORIES
from shadow.data.calibration_safety import SafetyResult, json_pointer_value, validate_output
from shadow.data.db import check_schema_version, connect, init_db, resolve_db
from shadow.data.queries import CalibrationTools
from shadow.data.writers import insert_calibration_attempts, insert_memo

ATTEMPT_COUNT = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _previous_state(con, as_of: date, experiment_version: str) -> tuple[int | None, float | None, date | None]:
    row = con.execute(
        "SELECT risk_level FROM memos WHERE experiment_version = ? AND cycle_date < ? "
        "ORDER BY cycle_date DESC LIMIT 1", [experiment_version, as_of]
    ).fetchone()
    previous_level = int(row[0]) if row else None
    snap = con.execute(
        "SELECT snapshot_date, drawdown FROM book_snapshots WHERE experiment_version = ? "
        "AND book = 'decision' AND snapshot_date <= ? ORDER BY snapshot_date DESC LIMIT 1",
        [experiment_version, as_of],
    ).fetchone()
    if snap is None:
        return previous_level, None, None
    return previous_level, float(snap[1]), snap[0]


def _output_with_accepted_level(output, safety: SafetyResult):
    accepted = safety.accepted_risk_level
    if accepted is None or accepted == output.risk_level:
        return output
    rationale = (output.risk_level_rationale.rstrip() +
                 " The proposed increase was held at the prior level because a recent "
                 "decision-book snapshot was unavailable.")
    return output.model_copy(update={"risk_level": accepted, "risk_level_rationale": rationale})


def render_memo(output, safety: SafetyResult, as_of: date, experiment_version: str) -> str:
    lines = [f"Calibration memo for {experiment_version} as of {as_of.isoformat()}",
             output.summary,
             f"Overall risk level: {output.risk_level} — {output.risk_level_rationale}", ""]
    for category in CATEGORIES:
        advice = output.advice[category]
        lines.append(f"{category.replace('_', ' ').title()}: {advice.advice} ({advice.confidence} confidence).")
        lines.append(f"Reasoning: {advice.reasoning}")
        if advice.conditions:
            lines.append("Conditions: " + "; ".join(advice.conditions))
        if advice.supporting_ids:
            lines.append("Supporting records: " + ", ".join(advice.supporting_ids))
        if advice.contradicting_ids:
            lines.append("Contradicting records: " + ", ".join(advice.contradicting_ids))
        if advice.watch_next:
            lines.append("Watch next cycle: " + "; ".join(advice.watch_next))
        lines.append("")
    if output.numeric_claims:
        lines.append("Verified figures:")
        for claim in output.numeric_claims:
            suffix = f" {claim.unit}" if claim.unit else ""
            lines.append(f"- {claim.statement}: {claim.value:g}{suffix} (tool call {claim.tool_call_seq})")
        lines.append("")
    if output.hypotheses:
        lines.append("Hypotheses to watch:")
        lines.extend(f"- {h.category}: {h.statement}" for h in output.hypotheses)
        lines.append("")
    if output.hypothesis_checks:
        lines.append("Prior hypothesis checks:")
        lines.extend(f"- {h.hypothesis_id}: {h.status}" for h in output.hypothesis_checks)
        lines.append("")
    if safety.flags:
        lines.append("Review flags:")
        lines.extend(f"- {flag['code']}: {flag.get('message', flag.get('category', 'review'))}"
                     for flag in safety.flags)
    return "\n".join(lines).strip()


def _attempt_record(result, safety: SafetyResult | None, attempt_no: int,
                    calibration_run_id: str) -> dict:
    output = result.output
    return {
        "attempt_id": f"attempt_{uuid.uuid4().hex[:12]}",
        "calibration_run_id": calibration_run_id,
        "attempt_no": attempt_no,
        "status": "invalid" if output is not None and safety and not safety.passed else result.status,
        "model": CALIBRATION_MODEL,
        "proposed_risk_level": output.risk_level if output else None,
        "accepted_risk_level": safety.accepted_risk_level if safety else None,
        "output": output.as_json_dict() if output else {},
        "error": result.error,
        "safety_passed": bool(safety and safety.passed),
        "safety_blockers": safety.blockers if safety else [],
        "safety_flags": safety.flags if safety else [],
        "evidence_score": list(safety.evidence_score) if safety else [0, 0, 0],
        "tool_calls": result.tool_calls,
        "created_at": _now(),
    }


def _hypothesis_checks(output, tool_calls: list[dict]) -> list[dict]:
    by_seq = {int(call["seq"]): call for call in tool_calls}
    rows = []
    for check in output.hypothesis_checks:
        call = by_seq[check.evidence_tool_call_seq]
        evidence = json_pointer_value(call.get("result"), check.evidence_json_pointer)
        rows.append({"hypothesis_id": check.hypothesis_id, "status": check.status,
                     "evidence": {"tool_call_seq": check.evidence_tool_call_seq,
                                  "json_pointer": check.evidence_json_pointer,
                                  "result": evidence,
                                  "record_ids": check.evidence_record_ids}})
    return rows


def run_cycle(*, as_of: str, experiment_version: str, db_path=None,
              llm=None) -> dict:
    """Run three serialized model attempts, then persist attempts and at most one memo."""
    cycle_date = date.fromisoformat(as_of)
    path = resolve_db(db_path, dev=False) if db_path else resolve_db(None, dev=False)
    # Apply the additive schema migration before opening a read-only connection.
    migration_con = init_db(path)
    migration_con.close()
    read_con = connect(path, read_only=True)
    check_schema_version(read_con, path)
    try:
        if read_con.execute("SELECT 1 FROM memos WHERE experiment_version = ? AND cycle_date = ?",
                            [experiment_version, cycle_date]).fetchone():
            raise ValueError(f"a memo already exists for {experiment_version} on {as_of}")
        previous_level, drawdown, snapshot_date = _previous_state(read_con, cycle_date, experiment_version)
        query_tools = CalibrationTools(read_con, cycle_date, experiment_version)
        try:
            model = llm if llm is not None else build_calibration_llm()
        except Exception as exc:
            results = [AttemptResult(output=None, tool_calls=[],
                                     error=f"{type(exc).__name__}: {str(exc)[:2000]}",
                                     status="not_configured") for _ in range(ATTEMPT_COUNT)]
        else:
            results = [run_attempt(model, query_tools, as_of=as_of,
                                   experiment_version=experiment_version,
                                   previous_risk_level=previous_level, attempt_no=i + 1)
                       for i in range(ATTEMPT_COUNT)]
        checked = []
        for result in results:
            safety = (validate_output(result.output, result.tool_calls,
                                      previous_risk_level=previous_level,
                                      decision_drawdown=drawdown,
                                      snapshot_date=snapshot_date, as_of=cycle_date)
                      if result.output else None)
            checked.append((result, safety))
    finally:
        read_con.close()

    calibration_run_id = f"calrun_{uuid.uuid4().hex[:12]}"
    records = [_attempt_record(result, safety, i + 1, calibration_run_id)
               for i, (result, safety) in enumerate(checked)]
    passing = [(result, safety, records[i]) for i, (result, safety) in enumerate(checked)
               if result.output is not None and safety is not None and safety.passed]

    # Persist every attempt before writing a final memo; failed runs remain auditable.
    write_con = connect(path)
    check_schema_version(write_con, path)
    try:
        insert_calibration_attempts(write_con, experiment_version=experiment_version,
                                    cycle_date=cycle_date, calibration_run_id=calibration_run_id,
                                    attempts=records)
        if not passing:
            return {"calibration_run_id": calibration_run_id, "status": "no_valid_memo",
                    "attempts": len(records), "memo_id": None}

        selected_result, selected_safety, selected_record = max(
            passing, key=lambda x: (x[1].evidence_score, -x[2]["attempt_no"]))
        selected_output = _output_with_accepted_level(selected_result.output, selected_safety)
        advice = {}
        for category, value in selected_output.advice.items():
            advice[category] = {
                "advice": value.advice,
                "confidence": value.confidence,
                "reasoning": value.reasoning,
                "cited_ids": list(dict.fromkeys(value.supporting_ids + value.contradicting_ids)),
                "supporting_ids": value.supporting_ids,
                "contradicting_ids": value.contradicting_ids,
                "conditions": value.conditions,
                "watch_next": value.watch_next,
            }
        output = insert_memo(
            write_con, experiment_version=experiment_version, cycle_date=cycle_date,
            risk_level=selected_output.risk_level, risk_level_rationale=selected_output.risk_level_rationale,
            memo_text=render_memo(selected_output, selected_safety, cycle_date, experiment_version),
            advice=advice, safety_passed=True,
            safety_flags={"flags": selected_safety.flags,
                          "attempt_spread": {"risk_levels": [r.output.risk_level for r, _ in checked if r.output],
                                             "advice": {cat: [r.output.advice[cat].advice
                                                              for r, _ in checked if r.output]
                                                        for cat in CATEGORIES}},
                          "selected_attempt_id": selected_record["attempt_id"],
                          "proposed_risk_level": selected_result.output.risk_level},
            model=CALIBRATION_MODEL,
            hypotheses=[h.model_dump(mode="json") for h in selected_output.hypotheses],
            hypothesis_checks=_hypothesis_checks(selected_output, selected_result.tool_calls),
            tool_calls=selected_result.tool_calls,
        )
        return {"calibration_run_id": calibration_run_id, "status": "memo_written",
                "attempts": len(records), "selected_attempt_id": selected_record["attempt_id"],
                **output}
    finally:
        write_con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one three-attempt Calibration Agent cycle.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--db", help="DuckDB path; defaults to the main calibration database")
    parser.add_argument("--dev", action="store_true", help="use the development database")
    args = parser.parse_args(argv)
    path = resolve_db(args.db, args.dev)
    result = run_cycle(as_of=args.as_of, experiment_version=args.experiment_version, db_path=path)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "memo_written" else 2


if __name__ == "__main__":
    raise SystemExit(main())

