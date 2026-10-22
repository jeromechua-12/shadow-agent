# Shadow-agent project context

DSA4213 (NLP for Data Science) group project, NUS, Sem 1 AY2026/27.
We build on TradingAgents (TauricResearch, Apache-2.0), copied into `research_agent/`.
Research question: does feeding back how the risk team's past decisions turned out,
including trades it rejected, improve its risk decisions and risk-adjusted returns?

## How the system works

Daily trading loop (per stock, per trading day):
1. Upstream agents (existing, reused unchanged): market, sentiment, news and
   fundamentals analysts → bull vs bear researchers → research manager → Trader proposal.
   Run once per stock per day and cached, so every experiment version sees identical proposals.
2. Risk debate (aggressive, neutral, conservative) + Portfolio Manager (existing, extended
   by a teammate): prompts get the latest risk memo; PM also outputs a verdict
   (approved / reduced / rejected), a 0-3 rating for five risk categories
   (volatility, event risk, valuation, liquidity, concentration) and the main reason.
3. Decision logger (new): compares the Trader's proposal with the final decision for every
   trade; a reduced trade is split into an approved part and a rejected part.
4. Two-book simulator (new): decision book follows final decisions; shadow book follows the
   Trader's proposals in full. Same prices and costs, next-day open fills, long-only.
   Ratings map to positions: Buy 100% of per-ticker allocation, Overweight 50%,
   Hold unchanged, Underweight halve, Sell exit.
5. Outcome judge (new): 10 trading days later, score = direction × (stock return − SPY return).
   Labels: correct rejection, missed gain, approval loss, correct approval; scores within ±1% = noise.
6. Evidence database (new, DuckDB): stores everything with dates.

Calibration loop (every 4 simulated weeks):
7. Statistics engine: Beta-Binomial success rates per category, CUSUM trend, drawdown,
   real vs shadow returns. These are EVIDENCE, not verdicts. They no longer decide advice.
8. Calibration Agent (LLM): reads the statistics, uses read-only tools to search trades,
   read the reports and news behind them, compute statistics on subgroups it defines,
   and check whether last cycle's hypotheses held. Decides tighten / loosen / keep per
   category, with confidence, reasoning and cited record IDs. New subgroup patterns are
   hypotheses until confirmed on the next cycle's trades.
9. Safety checker: every number in the memo must match a tool result. Only hard limits block:
   risk level 1 to 5, at most one step per cycle, no extra risk after a >10% drawdown.
   The 85% Beta-Binomial threshold and CUSUM are NOT blocking rules; advice that goes against
   strong statistics without a reason is flagged.
10. Risk memo goes into the risk team's prompts for the next 4 weeks.

Experiment: three versions on the same cache — rules with a template memo; LLM with
statistics only; LLM with statistics plus past cases. Baselines: SPY buy-and-hold,
original TradingAgents.

## My role (Avery): data layer

- Upstream cache: run the graph up to the Trader once per stock per trading day; save outputs.
- Raw evidence capture: save every data tool call's inputs, outputs, vendor, status and fetch time.
  News, social posts and prediction markets cannot be re-fetched for past dates.
- DuckDB evidence database: schema, loaders, data quality checks.
- Read-only query functions for the Calibration Agent: cycle_summary, search_trades,
  read_trade, subgroup_stats, past_memos.

## Schema notes (base: the Calibration Data Plan in docs/)

Keep runs, evidence_calls, text_items, run_text_items, agent_outputs, conditions.
Changes for this project:
- trader_proposals: action, direction (+1/0/-1), entry_price, stop_loss, position_sizing, reasoning.
- decisions: one row per trade part; proposed vs final size, verdict, part
  (approved_part / rejected_part), five risk ratings 0-3, main_reason, explanation,
  experiment_version.
- outcomes: 10-day horizon, resolved_on, stock_return, spy_return, score, label.
- book_snapshots: date, book (decision / shadow), experiment_version, cash, value, drawdown.
- memos (replaces lessons): cycle date, experiment_version, advice per category, confidence,
  cited record IDs, hypotheses with status (pending / held / failed), tool calls.

## Rules

- Do NOT edit anything under research_agent/tradingagents/ or research_agent/cli/.
  Another teammate owns changes there. Hook in from my own module instead
  (e.g. compile ta.workflow with interrupt_after=["Trader"]; wrap route_to_vendor in
  tradingagents.agents.tools, which imports it directly).
- My code lives in research_agent/shadow/data/.
- Never commit data: upstream_cache/, warehouse/, *.duckdb are gitignored.
- Point-in-time: every record keeps the analysis date, fetch time, and (for outcomes) the
  date the outcome became known. Queries for a calibration date only return outcomes
  resolved on or before it. Enforce this inside the query functions.
- Append-only: never overwrite or delete records; a rerun gets a new run_id.
- Numbers as numeric columns, not inside text. No SQL built from raw agent-supplied strings.
- Python only. LLM provider is OpenRouter (TRADINGAGENTS_LLM_PROVIDER=openrouter); the key is
  OPENROUTER_API_KEY in research_agent/.env. Model IDs use OpenRouter names (e.g. openai/...).
  Respect OpenRouter rate limits: retry with backoff on 429, and keep parallel runs small.
- Upstream agents must not see portfolio state or the memory log, so cached outputs depend
  only on (ticker, date). The Trader reads portfolio context, so run with no portfolio.
- Never start a run that calls the LLM on more than 1 stock and 1 date without asking me.

## Environment and data status (as of 2026-10-22)

- Working branch: `yt`. Windows, PowerShell, venv at research_agent/.venv.
- research_agent/.env holds OPENROUTER_API_KEY, ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY,
  SEC_EDGAR_USER_AGENT and FRED_API_KEY (FRED available since 2026-10-22). Code must still run
  without a key, record the source as not_configured, and allow rerunning with --rerun-incomplete.
- LLM: DeepSeek V4 Flash via OpenRouter, temperature 0.0, max tokens 16384. (8192 was too low:
  hidden reasoning used up the cap and agents returned empty text, finish_reason=length.)
  OpenRouter budget is limited (about $15); one full TradingAgents run cost about $0.02.
- Test runs showed, for past dates:
  - prices, technical indicators, financial statements: work
  - Yahoo news (company and global): empty, Yahoo only serves recent items
  - StockTwits: HTTP 403 for any date
  - Reddit: only works for recent dates
  - Polymarket and valuation metrics: withheld for past dates on purpose (look-ahead guard)
  - FRED macro indicators: work (vintage pinned to the analysis date)
- Financial statements use tool_vendors "sec_edgar,yfinance" (EDGAR by filing date first).
- News source for past dates: Alpaca news API (historical Benzinga news, free account, tested OK).
  GET https://data.alpaca.markets/v1beta1/news with params symbols, start, end, limit (max 50),
  page_token; headers APCA-API-KEY-ID and APCA-API-SECRET-KEY. Use created_at (not updated_at)
  for point-in-time filtering. Plug it in from my own module by adding an "alpaca" vendor to
  tradingagents.dataflows.router.VENDOR_METHODS and setting data_vendors["news_data"] = "alpaca".
  Download once into local storage, then serve from there, so runs are reproducible.
- Record why each data call failed with a clear status: ok, no_data, unavailable, withheld,
  blocked, not_configured.

## Universe and settings

Stocks: AAPL, GOOGL, MSFT, NVDA, AMZN, JPM, XOM, JNJ, TSLA, PG. Benchmark: SPY.
Test period about 6 months (about 126 trading days), daily decisions.
One research debate round, one risk round. One pinned model for all runs.
