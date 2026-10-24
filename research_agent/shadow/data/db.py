"""The DuckDB evidence database: location, connection, schema creation."""

from __future__ import annotations

from pathlib import Path

import duckdb

from shadow.data.paths import warehouse_dir

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "3"


MAIN_DB_NAME = "calibration.duckdb"
DEV_DB_NAME = "dev.duckdb"


def default_db_path() -> Path:
    """The main database: real upstream runs and teammates' real rows."""
    return warehouse_dir() / MAIN_DB_NAME


def dev_db_path() -> Path:
    """A scratch database for samples and smoke_test stand-ins; never mixed with the main one."""
    return warehouse_dir() / DEV_DB_NAME


def resolve_db(path: str | Path | None, dev: bool = False) -> Path:
    """The path a command line --db / --dev pair selects."""
    if path and dev:
        raise ValueError("give either --db or --dev, not both")
    return Path(path) if path else (dev_db_path() if dev else default_db_path())


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path) if path is not None else default_db_path()
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    # Timestamps are stored and compared in UTC (a run dated D sees items
    # created before midnight UTC after D), whatever the laptop's timezone.
    con.execute("SET TimeZone = 'UTC'")
    return con


def check_schema_version(con: duckdb.DuckDBPyConnection, path=None, allow_empty: bool = False) -> None:
    """Raise a clear error if the database was built by a different schema version."""
    has_info = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'schema_info'").fetchone()[0]
    found = con.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone() if has_info else None
    if found is None and allow_empty:
        return
    if found is None or found[0] != SCHEMA_VERSION:
        raise RuntimeError(f"{path or default_db_path()} has schema version {found[0] if found else 'unknown'}, "
                           f"this code needs {SCHEMA_VERSION}. No rows were removed. "
                           "Run the data loader to apply a supported additive migration.")


def init_db(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Create/upgrade the schema without discarding existing evidence rows."""
    con = connect(path)
    try:
        has_info = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'schema_info'"
        ).fetchone()[0]
        if has_info:
            found = con.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
            version = found[0] if found else None
            if version == "2":
                # Add only nullable/new fields and tables; preserve every existing row.
                con.execute("ALTER TABLE memos ADD COLUMN IF NOT EXISTS risk_level_rationale VARCHAR")
                con.execute("ALTER TABLE memo_advice ADD COLUMN IF NOT EXISTS conditions JSON")
                con.execute("ALTER TABLE memo_advice ADD COLUMN IF NOT EXISTS supporting_ids VARCHAR[]")
                con.execute("ALTER TABLE memo_advice ADD COLUMN IF NOT EXISTS contradicting_ids VARCHAR[]")
                con.execute("ALTER TABLE memo_advice ADD COLUMN IF NOT EXISTS watch_next JSON")
            elif version not in (None, SCHEMA_VERSION):
                raise RuntimeError(f"{path or default_db_path()} has schema version {version}; "
                                   f"this code needs {SCHEMA_VERSION}. No rows were removed.")
        con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        con.execute("INSERT OR REPLACE INTO schema_info VALUES ('schema_version', ?)", [SCHEMA_VERSION])
        return con
    except Exception:
        con.close()
        raise
