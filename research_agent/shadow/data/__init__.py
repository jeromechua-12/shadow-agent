"""Data layer: upstream cache, raw evidence capture, DuckDB evidence database."""

from shadow.data.paths import load_env

# tradingagents reads .env from the working directory and folds TRADINGAGENTS_*
# into DEFAULT_CONFIG at import time. Load research_agent/.env first so the
# settings are the same wherever a command is run from.
load_env()
