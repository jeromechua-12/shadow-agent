"""A readable digest of the statistics engine's output for one calibration date.

Wraps ``CalibrationTools.cycle_summary`` (success rate, Beta-Binomial posterior,
CUSUM trend, drawdown, decision-vs-shadow return) with plain-text formatting
instead of raw JSON, for a quick look before handing numbers to the
Calibration Agent. No new calculations: everything here already lives in
``queries.py``.

Usage:
    python -m shadow.data.stats_report --as-of 2026-10-22 --experiment-version stats_dev --dev
    python -m shadow.data.stats_report --as-of 2026-10-22 --experiment-version llm_stats --json
"""

from __future__ import annotations

import argparse
import json

from shadow.data.queries import CATEGORIES, CalibrationTools


def _fmt_rate(group: dict) -> str:
    if group["n_scored"] == 0:
        return "no scored trades"
    thin = " (thin evidence)" if group["thin_evidence"] else ""
    beta = group["beta"]
    return (f"{group['success_rate']:.0%} over {group['n_scored']} scored "
           f"[90% CI {beta['ci90'][0]:.0%}-{beta['ci90'][1]:.0%}]{thin}")


def _fmt_trend(trend: dict) -> str:
    if not trend["flagged"]:
        return "no trend flagged"
    at = trend["flagged_at"]
    return f"FLAGGED {trend['direction']} since {at['decision_id']} ({at['resolved_on']})"


def render(summary: dict) -> str:
    lines = [f"as of {summary['as_of']}  version={summary['experiment_version']}",
            f"this cycle (last {summary['window']['to']} vs "
            f"{summary['window']['from_exclusive']}): {_fmt_rate(summary['this_cycle'])}",
            f"all history: {_fmt_rate(summary['all_history'])}",
            f"trend (all history): {_fmt_trend(summary['all_history']['trend'])}",
            f"decisions awaiting outcome: {summary['decisions_awaiting_outcome']}",
            ""]
    if summary["current_memo"]:
        m = summary["current_memo"]
        lines.append(f"current memo: {m['memo_id']} (cycle {m['cycle_date']}, risk level {m['risk_level']})")
    else:
        lines.append("current memo: none yet")
    lines.append("")
    lines.append("by category:")
    for cat in (*CATEGORIES, "none"):
        h = summary["by_category"][cat]["all_history"]
        lines.append(f"  {cat:14} {_fmt_rate(h):55} {_fmt_trend(h['trend'])}")
    lines.append("")
    lines.append("books:")
    for book, b in summary["books"].items():
        if book.startswith("decision_minus"):
            continue
        max_dd = b["max_drawdown_this_cycle"]
        lines.append(f"  {book:9} value={b['value']:,.0f}  drawdown={b['drawdown']:.1%}  "
                     f"return_this_cycle={b['return_this_cycle']:.1%}  "
                     f"max_drawdown_this_cycle={'n/a' if max_dd is None else f'{max_dd:.1%}'}")
    if "decision_minus_shadow_return_this_cycle" in summary["books"]:
        lines.append(f"  decision - shadow return this cycle: "
                     f"{summary['books']['decision_minus_shadow_return_this_cycle']:.1%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from shadow.data.db import check_schema_version, connect, resolve_db

    parser = argparse.ArgumentParser(description="A readable digest of the statistics engine's output.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true", help="use warehouse/dev.duckdb")
    parser.add_argument("--as-of", required=True, help="calibration date")
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--json", action="store_true", help="print the full cycle_summary as JSON instead")
    args = parser.parse_args(argv)

    path = resolve_db(args.db, args.dev)
    con = connect(path, read_only=True)
    check_schema_version(con, path)
    summary = CalibrationTools(con, args.as_of, args.experiment_version).cycle_summary()
    print(json.dumps(summary, indent=1, default=str) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
