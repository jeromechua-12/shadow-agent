# Components
- **Upstream (reused, cached)**
  - Data tools: prices, filings, news, macro, social
  - Analysts: market, sentiment, news, fundamentals
  - Bull and bear researchers, research manager, Trader
  - Upstream cache: all outputs saved per stock and day
  - No upstream agent sees portfolio state or the memo
- **Risk stage (existing, extended)**
  - Risk debate: three analysts, each reads the latest risk memo
  - Portfolio Manager: verdict plus 0–3 rating for five risk categories
    - Verdict: approved / reduced / rejected
    - Categories: volatility, event risk, valuation, liquidity, concentration
    - Also outputs the main reason
  - Unparseable output: retry once, then record Hold
- **Logging and scoring (new)**
  - Decision logger: proposal vs final decision, reduced trades split in two
  - Two-book simulator: decision book vs shadow book, same prices and costs
  - Outcome judge: direction × (stock return − SPY return) after 10 days
  - Labels: correct rejection, missed gain, approval loss, correct approval
  - ±1% noise band; resolved date recorded to prevent leakage
- **Calibration (new)**
  - Evidence database: one DuckDB file, dated records, plain-text reports
  - Statistics engine: Beta-Binomial rates, CUSUM, drawdown, real vs shadow
  - Output is evidence with uncertainty, not advice
  - Calibration Agent: five read-only tools, date filter enforced in tools
  - Decides tighten, loosen or keep per risk category
  - Cites supporting and contradicting record IDs
  - New subgroup patterns stay hypotheses until confirmed next cycle
  - Safety checker: blocks hard-limit breaches only
    - Every number in the memo must match a tool result
    - Hard limits: risk level 1–5, at most one step per cycle,
      no extra risk after a >10% drawdown
    - 85% Beta-Binomial threshold and CUSUM are not blocking rules
  - Flags advice against strong statistics without a reason
  - Risk memo: inserted into risk debate and PM prompts for 4 weeks

  # Loops
- **Daily trading loop (per stock, per trading day)**
  - Upstream (cached) → risk debate + PM → decision logger
    → two-book simulator → outcome judge (10 trading days later) → evidence database
- **Calibration loop (every 4 simulated weeks)**
  - Statistics engine → Calibration Agent → safety checker
    → risk memo into risk team prompts for the next 4 weeks

# Evaluation Protocol
- **Same inputs**
  - Same stocks, test period, prices and costs
  - Upstream outputs cached; only the risk team and feedback differ
  - Gain/loss thresholds and evidence levels fixed across setups
- **Test period**
  - 6 months of past data, after the LLM training cutoff where possible
  - Reduces look-ahead bias
- **Trading frequency**
  - Fixed schedule (e.g. daily), same position limits
  - Number of trades not capped
- **Sample size**
  - ~126 trading days × 10 stocks = upper bound of ~1,260 decisions
  - ~6 calibration cycles, ~5 receive feedback
  - Hold proposals are not trades
  - Scores inside the noise band are excluded
- **Variability**
  - Fixed temperature, 0.2
  - Calibration runs once per point
- **Statistical testing**
  - Bootstrap CIs on Sharpe ratio from weekly results
  - Wilcoxon signed-rank test on weekly return differences
- **Fair comparison**
  - All setups scored on the same stock-and-date decisions

# Evaluation Metrics
- Rejection precision = correct rejections / (correct rejections + missed gains)
- Net value of rejections (avoided loss - missed gain) 
- Approval loss rate = approval losses / (approval losses + correct approvals)

# Evaluation Setups
| Setup | Description | What it shows |
|---|---|---|
| SPY buy-and-hold | Buy SPY at start, hold to end | Whether agents beat the market |
| Original TradingAgents | Pipeline, no calibration | Baseline without feedback |
| TradingAgents with feedback | Memo added every 4 weeks | Whether feedback improves decisions |

# Stock Tickers
AAPL, GOOGL, MSFT, NVDA, AMZN, JPM, XOM, JNJ, TSLA, PG. Benchmark: SPY.
