"""Find the current risk memo for a trading day, for the risk analysts' and PM's prompts."""

from __future__ import annotations


def current_memo(con, trade_date, experiment_version: str) -> dict | None:
    """Latest memo of the version written before ``trade_date``; None when there is none."""
    row = con.execute(
        "SELECT memo_id, memo_text FROM memos WHERE experiment_version = ? AND cycle_date < ? "
        "ORDER BY cycle_date DESC LIMIT 1", [experiment_version, trade_date]
    ).fetchone()
    return {"memo_id": row[0], "memo_text": row[1]} if row else None
