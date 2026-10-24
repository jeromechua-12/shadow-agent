"""Find the current memo the risk team sees on a trading day."""

from __future__ import annotations

from datetime import date

import duckdb


def current_memo(con: duckdb.DuckDBPyConnection, memo_version: str, trade_date: str | date) -> dict | None:
    """Latest safety-passed memo of ``memo_version`` with cycle_date <= trade_date, or None.

    If all 3 calibration attempts fail, no memo is written, so the previous one carries forward.
    """
    row = con.execute(
        "SELECT memo_id, cycle_date, risk_level, memo_text FROM memos "
        "WHERE experiment_version = ? AND cycle_date <= ? AND safety_passed "
        "ORDER BY cycle_date DESC LIMIT 1", [memo_version, trade_date]).fetchone()
    if row is None:
        return None
    return dict(zip(("memo_id", "cycle_date", "risk_level", "memo_text"), row))
