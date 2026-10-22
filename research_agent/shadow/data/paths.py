"""Filesystem locations for the data layer, independent of the working directory.

Everything resolves from this file's location so commands behave the same from
the repo root, from ``research_agent/`` or from anywhere else. Each location can
be moved with an environment variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

RESEARCH_AGENT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = RESEARCH_AGENT_DIR.parent


def load_env() -> None:
    """Load ``research_agent/.env`` without overriding variables already set."""
    load_dotenv(RESEARCH_AGENT_DIR / ".env", override=False)


def warehouse_dir() -> Path:
    """Root of the evidence warehouse (raw downloads and the DuckDB file)."""
    return Path(os.environ.get("SHADOW_WAREHOUSE_DIR") or REPO_ROOT / "warehouse")


def upstream_cache_dir() -> Path:
    """Root of the cached upstream (analysts → Trader) outputs."""
    return Path(os.environ.get("SHADOW_UPSTREAM_CACHE_DIR") or REPO_ROOT / "upstream_cache")
