-- Shadow-agent evidence database (DuckDB). Base: the Calibration Data Plan,
-- with the project changes in CLAUDE.md.
--
-- Rules every table follows:
--   * Append-only. Loaders INSERT; nothing is UPDATEd or DELETEd. A rerun gets a
--     new run_id; "current" rows come from views. A hypothesis's status changes
--     by adding a check row, not by editing the hypothesis.
--   * Point-in-time. Every row carries the date it is about (trade_date /
--     as_of / cycle_date) and when it was recorded or became known
--     (fetched_at / decided_at / resolved_on). The query functions only return
--     outcomes with resolved_on <= the calibration date.
--   * Numbers live in numeric columns, never only inside text.

CREATE TABLE IF NOT EXISTS schema_info (
  key   VARCHAR PRIMARY KEY,
  value VARCHAR
);

-- ---------------------------------------------------------------------------
-- Runs: one upstream run (analysts -> Trader) per row
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
  run_id             VARCHAR PRIMARY KEY,
  ticker             VARCHAR NOT NULL,
  trade_date         DATE NOT NULL,           -- the analysis date
  started_at         TIMESTAMPTZ NOT NULL,
  duration_s         DOUBLE,
  cache_version      INTEGER NOT NULL,
  complete           BOOLEAN NOT NULL,
  incomplete_reasons VARCHAR[],
  llm_provider       VARCHAR,
  deep_model         VARCHAR,
  quick_model        VARCHAR,
  temperature        DOUBLE,
  max_tokens         INTEGER,
  debate_rounds      INTEGER,
  analysts           VARCHAR[],
  data_vendors       JSON,
  tool_vendors       JSON,
  code_commit        VARCHAR,
  code_dirty         BOOLEAN,
  llm_calls          INTEGER,
  input_tokens       BIGINT,
  output_tokens      BIGINT,
  cost_usd           DOUBLE,
  cache_file         VARCHAR                   -- upstream_cache/<TICKER>/<date>.json at load time
);

-- Every LLM call of a run (diagnostics: empty replies, token-cap hits, cost).
CREATE TABLE IF NOT EXISTS llm_calls (
  run_id            VARCHAR NOT NULL REFERENCES runs(run_id),
  seq               INTEGER NOT NULL,
  node              VARCHAR,
  model             VARCHAR,
  finish_reason     VARCHAR,
  input_tokens      INTEGER,
  output_tokens     INTEGER,
  reasoning_tokens  INTEGER,
  content_chars     INTEGER,
  tool_calls        INTEGER,
  cost_usd          DOUBLE,
  PRIMARY KEY (run_id, seq)
);

-- ---------------------------------------------------------------------------
-- Evidence: what the agents saw
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_calls (
  call_id        VARCHAR PRIMARY KEY,
  run_id         VARCHAR NOT NULL REFERENCES runs(run_id),
  node           VARCHAR,                 -- LangGraph node, e.g. tools_market, Sentiment Analyst
  agent          VARCHAR,                 -- market | sentiment | news | fundamentals
  kind           VARCHAR NOT NULL,        -- router | social
  method         VARCHAR NOT NULL,        -- get_stock_data, get_news, fetch_reddit_posts, ...
  args           JSON,
  as_of          DATE NOT NULL,           -- the run's trade_date
  fetched_at     TIMESTAMPTZ NOT NULL,
  vendor         VARCHAR,                 -- vendor that served it; NULL if none did
  vendors_tried  VARCHAR[],
  status         VARCHAR NOT NULL CHECK (status IN
                   ('ok','no_data','unavailable','withheld','blocked','not_configured','error')),
  retryable      BOOLEAN NOT NULL,        -- a rerun could fix it (missing key, throttling)
  detail         VARCHAR,                 -- why it failed
  seconds        DOUBLE,
  output_chars   INTEGER,
  payload_path   VARCHAR,                 -- warehouse/raw/payloads/<sha[:2]>/<sha>.txt
  payload_sha256 VARCHAR
);

-- Headlines and posts, stored once however many runs saw them.
CREATE TABLE IF NOT EXISTS text_items (
  item_id       VARCHAR PRIMARY KEY,      -- '<source>:<original id>', e.g. 'alpaca_news:61507903'
  source        VARCHAR NOT NULL,         -- alpaca_news | stocktwits | reddit | polymarket
  source_id     VARCHAR NOT NULL,
  store_key     VARCHAR,                  -- the query it was downloaded for: ticker or _global
  symbols       VARCHAR[],
  created_at    TIMESTAMPTZ NOT NULL,     -- point-in-time filter uses this, never updated_at
  updated_at    TIMESTAMPTZ,
  first_seen_at TIMESTAMPTZ NOT NULL,     -- when we downloaded it
  title         VARCHAR,
  body          VARCHAR,
  author        VARCHAR,
  publisher     VARCHAR,
  url           VARCHAR,
  user_tag      VARCHAR,                  -- StockTwits Bullish / Bearish
  extra         JSON
);

-- Which items each data call put in front of an agent, in the order shown.
CREATE TABLE IF NOT EXISTS run_text_items (
  run_id  VARCHAR NOT NULL REFERENCES runs(run_id),
  call_id VARCHAR NOT NULL REFERENCES evidence_calls(call_id),
  item_id VARCHAR NOT NULL REFERENCES text_items(item_id),
  rank    INTEGER NOT NULL,
  PRIMARY KEY (call_id, item_id)
);

-- ---------------------------------------------------------------------------
-- What the upstream agents said
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_outputs (
  run_id     VARCHAR NOT NULL REFERENCES runs(run_id),
  agent      VARCHAR NOT NULL,   -- market | sentiment | news | fundamentals | bull | bear
                                 -- | research_manager | trader
  round      INTEGER NOT NULL,   -- 1 for reports; debate round for bull/bear
  text       VARCHAR NOT NULL,
  structured JSON,               -- the agent's structured fields, if it produced them
  PRIMARY KEY (run_id, agent, round)
);

CREATE TABLE IF NOT EXISTS trader_proposals (
  run_id          VARCHAR PRIMARY KEY REFERENCES runs(run_id),
  ticker          VARCHAR NOT NULL,
  trade_date      DATE NOT NULL,
  action          VARCHAR CHECK (action IN ('Buy','Hold','Sell')),
  direction       INTEGER CHECK (direction IN (-1, 0, 1)),   -- +1 Buy, 0 Hold, -1 Sell
  entry_price     DOUBLE,
  stop_loss       DOUBLE,
  position_sizing VARCHAR,
  reasoning       VARCHAR,
  structured_ok   BOOLEAN NOT NULL  -- false: Trader fell back to free text, fields are NULL
);

-- Market numbers at decision time, computed by plain code (no LLM).
CREATE TABLE IF NOT EXISTS conditions (
  run_id               VARCHAR PRIMARY KEY REFERENCES runs(run_id),
  as_of                DATE NOT NULL,
  close                DOUBLE,
  atr_pct              DOUBLE,   -- 14-day ATR / close
  volatility_20d       DOUBLE,   -- stdev of daily returns, annualised
  rsi_14               DOUBLE,
  pct_vs_sma200        DOUBLE,
  vix                  DOUBLE,
  yield_curve          DOUBLE,   -- 10y minus 2y
  sentiment_score      DOUBLE,   -- 0-10, from the Sentiment Analyst's structured output
  sentiment_band       VARCHAR,
  sentiment_confidence VARCHAR,
  news_items_seen      INTEGER,
  missing_sources      VARCHAR[] -- methods with no usable data (status <> 'ok')
);

-- ---------------------------------------------------------------------------
-- Risk team decisions (written by the decision logger; one row per trade part)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
  decision_id          VARCHAR PRIMARY KEY,
  run_id               VARCHAR NOT NULL REFERENCES runs(run_id),  -- upstream run it judged
  experiment_version   VARCHAR NOT NULL,   -- e.g. rules_template | llm_stats | llm_stats_cases
  memo_id              VARCHAR,            -- risk memo in force (NULL before the first memo)
  ticker               VARCHAR NOT NULL,
  trade_date           DATE NOT NULL,
  decided_at           TIMESTAMPTZ NOT NULL,
  part                 VARCHAR NOT NULL CHECK (part IN ('approved_part','rejected_part')),
  verdict              VARCHAR NOT NULL CHECK (verdict IN ('approved','reduced','rejected')),
  proposed_action      VARCHAR NOT NULL CHECK (proposed_action IN ('Buy','Hold','Sell')),
  direction            INTEGER NOT NULL CHECK (direction IN (-1, 0, 1)),
  proposed_size        DOUBLE NOT NULL,    -- fraction of the per-ticker allocation
  final_size           DOUBLE NOT NULL,
  part_size            DOUBLE NOT NULL,    -- size of this part (approved or rejected share)
  pm_rating            VARCHAR,            -- Buy / Overweight / Hold / Underweight / Sell
  risk_volatility      TINYINT CHECK (risk_volatility BETWEEN 0 AND 3),
  risk_event           TINYINT CHECK (risk_event BETWEEN 0 AND 3),
  risk_valuation       TINYINT CHECK (risk_valuation BETWEEN 0 AND 3),
  risk_liquidity       TINYINT CHECK (risk_liquidity BETWEEN 0 AND 3),
  risk_concentration   TINYINT CHECK (risk_concentration BETWEEN 0 AND 3),
  main_reason          VARCHAR CHECK (main_reason IN
                         ('volatility','event_risk','valuation','liquidity','concentration','none')),
  explanation          VARCHAR
);

-- 10 trading days later: score = direction x (stock return - SPY return).
CREATE TABLE IF NOT EXISTS outcomes (
  decision_id   VARCHAR PRIMARY KEY REFERENCES decisions(decision_id),
  horizon_days  INTEGER NOT NULL DEFAULT 10,
  entry_date    DATE NOT NULL,      -- next-day open fill
  resolved_on   DATE NOT NULL,      -- the date this outcome became known
  stock_return  DOUBLE NOT NULL,
  spy_return    DOUBLE NOT NULL,
  score         DOUBLE NOT NULL,
  label         VARCHAR NOT NULL CHECK (label IN
                  ('correct_rejection','missed_gain','approval_loss','correct_approval','noise'))
);

-- Daily value of each book under each experiment version.
CREATE TABLE IF NOT EXISTS book_snapshots (
  snapshot_date      DATE NOT NULL,
  book               VARCHAR NOT NULL CHECK (book IN ('decision','shadow')),
  experiment_version VARCHAR NOT NULL,
  cash               DOUBLE NOT NULL,
  value              DOUBLE NOT NULL,
  drawdown           DOUBLE NOT NULL,   -- (value - running peak) / running peak, <= 0
  positions          JSON,              -- {ticker: shares}
  PRIMARY KEY (experiment_version, book, snapshot_date)
);

-- ---------------------------------------------------------------------------
-- Risk memos from the Calibration Agent (replace the Data Plan's lessons)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memos (
  memo_id             VARCHAR PRIMARY KEY,
  experiment_version  VARCHAR NOT NULL,
  cycle_date          DATE NOT NULL,     -- calibration date; may only use outcomes resolved by it
  created_at          TIMESTAMPTZ NOT NULL,
  risk_level          INTEGER NOT NULL CHECK (risk_level BETWEEN 1 AND 5),
  previous_risk_level INTEGER CHECK (previous_risk_level BETWEEN 1 AND 5),
  memo_text           VARCHAR NOT NULL,  -- what goes into the risk team's prompts
  model               VARCHAR,
  safety_passed       BOOLEAN NOT NULL,
  safety_flags        JSON,              -- e.g. advice against strong statistics without a reason
  UNIQUE (experiment_version, cycle_date)
);

CREATE TABLE IF NOT EXISTS memo_advice (
  memo_id    VARCHAR NOT NULL REFERENCES memos(memo_id),
  category   VARCHAR NOT NULL CHECK (category IN
               ('volatility','event_risk','valuation','liquidity','concentration')),
  advice     VARCHAR NOT NULL CHECK (advice IN ('tighten','loosen','keep')),
  confidence VARCHAR NOT NULL CHECK (confidence IN ('low','medium','high')),
  reasoning  VARCHAR NOT NULL,
  cited_ids  VARCHAR[],                  -- decision_ids / run_ids / item_ids it relied on
  PRIMARY KEY (memo_id, category)
);

-- A subgroup pattern the agent proposed; a hypothesis until a later cycle confirms it.
CREATE TABLE IF NOT EXISTS memo_hypotheses (
  hypothesis_id VARCHAR PRIMARY KEY,
  memo_id       VARCHAR NOT NULL REFERENCES memos(memo_id),
  category      VARCHAR,
  statement     VARCHAR NOT NULL,
  subgroup      JSON NOT NULL            -- the structured filter subgroup_stats was given
);

-- Status changes are new rows (append-only); the latest check is the status.
CREATE TABLE IF NOT EXISTS hypothesis_checks (
  hypothesis_id VARCHAR NOT NULL REFERENCES memo_hypotheses(hypothesis_id),
  memo_id       VARCHAR NOT NULL REFERENCES memos(memo_id),   -- the cycle that checked it
  status        VARCHAR NOT NULL CHECK (status IN ('pending','held','failed')),
  evidence      JSON,                                        -- n, success rate, etc. on new trades
  PRIMARY KEY (hypothesis_id, memo_id)
);

-- Every read-only tool call the Calibration Agent made for a memo.
CREATE TABLE IF NOT EXISTS memo_tool_calls (
  memo_id   VARCHAR NOT NULL REFERENCES memos(memo_id),
  seq       INTEGER NOT NULL,
  tool      VARCHAR NOT NULL,   -- cycle_summary | search_trades | read_trade | subgroup_stats | past_memos
  args      JSON NOT NULL,
  result    JSON,
  called_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (memo_id, seq)
);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------
-- The run each experiment should use per (ticker, date): the latest complete one.
CREATE OR REPLACE VIEW current_runs AS
SELECT * FROM runs
WHERE complete
QUALIFY row_number() OVER (PARTITION BY ticker, trade_date ORDER BY started_at DESC) = 1;

CREATE OR REPLACE VIEW hypothesis_status AS
SELECT h.*, c.status, c.memo_id AS checked_in_memo, c.evidence
FROM memo_hypotheses h
LEFT JOIN (
  SELECT hc.*, m.cycle_date FROM hypothesis_checks hc JOIN memos m USING (memo_id)
  QUALIFY row_number() OVER (PARTITION BY hypothesis_id ORDER BY m.cycle_date DESC) = 1
) c USING (hypothesis_id);
