# Shadow Trading Agent

Does feeding back how the risk team's past decisions turned out, including trades it
rejected, improve its risk decisions and risk-adjusted returns? Built on TradingAgents.
Design: [docs/architecture.md](docs/architecture.md). Data layer:
[research_agent/shadow/data/README.md](research_agent/shadow/data/README.md).

## Setup

From `research_agent/`: create the venv, run `pip install -e ".[dev]"`, and put
`OPENROUTER_API_KEY`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `SEC_EDGAR_USER_AGENT`
and `FRED_API_KEY` in `research_agent/.env`. Run everything on one machine: the database
allows one writer, and `warehouse/` holds the news, payloads and checkpoints.

## Run order

All commands run from `research_agent/`. Each LLM step has a limit flag that defaults to a
single run; raise it on purpose. Add `--dev` to use the sample database instead.

```powershell
# 1. Upstream cache: analysts -> Trader, once per stock and trading day
python -m shadow.data.cache_runner --tickers AAPL NVDA --dates 2026-09-01 --dry-run
python -m shadow.data.cache_runner --tickers AAPL NVDA --dates 2026-09-01 --max-runs 2 --workers 2

# 2. Load the cache into DuckDB and check the data
python -m shadow.data.loader
python -m shadow.data.checks

# 3. Risk stage for both setups, calibration every 20 trading days, books and outcomes
python -m shadow.data.risk_runner --dry-run
python -m shadow.data.risk_runner --max-pairs 50 --max-cycles 1

# 4. Score any outcome still due (the risk runner does this at the end too)
python -m shadow.data.outcome_judge

# 5. Evaluate: SPY vs original vs feedback
python -m shadow.data.evaluation
```

Tests: `python -m pytest shadow/tests -q` from `research_agent/` (offline, no LLM calls).

## Acknowledgments

The research layer of this project uses
[TradingAgents](https://github.com/TauricResearch/TradingAgents) by
TauricResearch (Yijia Xiao, Edward Sun, Di Luo, Wei Wang), licensed under the
Apache License 2.0. This project is independent and is not affiliated with or
endorsed by TauricResearch.
