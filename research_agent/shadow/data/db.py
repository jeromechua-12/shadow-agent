"""The DuckDB evidence database: location, connection, schema creation."""

from __future__ import annotations

from pathlib import Path

import duckdb

from shadow.data.paths import warehouse_dir

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"


def default_db_path() -> Path:
    return warehouse_dir() / "calibration.duckdb"


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path) if path is not None else default_db_path()
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    # Timestamps are stored and compared in UTC (a run dated D sees items
    # created before midnight UTC after D), whatever the laptop's timezone.
    con.execute("SET TimeZone = 'UTC'")
    return con


def init_db(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Create every table and view if missing (idempotent) and return a connection."""
    con = connect(path)
    con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    con.execute("INSERT OR IGNORE INTO schema_info VALUES ('schema_version', ?)", [SCHEMA_VERSION])
    return con
