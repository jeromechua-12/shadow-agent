# Data layer (`shadow/data`)

Everything the experiment produces goes into one local DuckDB file: what the upstream
agents saw and said, the risk team's decisions, their outcomes, both books, and the
Calibration Agent's memos. This page shows how to fill it and read from it.
The schema is in [`schema.sql`](schema.sql).

All commands run from `research_agent/` with the venv active. Paths do not depend on
the working directory.

## Where things live

| Path (repo root) | What | In git? |
|---|---|---|
| `upstream_cache/<TICKER>/<date>.json` | **Real cache**: one upstream run (analysts → Trader) per ticker and date. Starts empty | No |
| `upstream_cache/<TICKER>/_superseded/` | Earlier runs of the same pair, kept as history | No |
| `warehouse/sample_upstream_cache/` | Test runs (the 2 × 5 sample below); same layout as the real cache | No |
| `warehouse/raw/alpaca_news/` | Downloaded Benzinga news, served to runs from here | No |
| `warehouse/raw/payloads/`, `warehouse/raw/evidence/` | The exact text every data call returned, and a per-run index | No |
| `warehouse/calibration.duckdb` | **Main database**: real runs and real rows only | No |
| `warehouse/dev.duckdb` | **Dev database**: samples and `smoke_test` stand-ins | No |

Override the roots with `SHADOW_WAREHOUSE_DIR`, `SHADOW_UPSTREAM_CACHE_DIR` and
`SHADOW_SAMPLE_CACHE_DIR`.

The two caches never mix: the main database loads only `upstream_cache/`, and the dev
database loads only the sample cache. News (`raw/alpaca_news/`) and saved payloads are
shared, since they are the same for any run that asks for them.

## 1. Load the cache into DuckDB

```powershell
python -m shadow.data.loader            # upstream_cache/ -> warehouse/calibration.duckdb
python -m shadow.data.loader --dev      # sample cache   -> warehouse/dev.duckdb
python -m shadow.data.checks            # data checks; exits 1 if any error (SKIP = could not run)
python -m shadow.data.checks --dev
python -m shadow.data.loader --backfill-conditions   # recompute conditions missing FRED/prices
```

Loading again only adds what is new, and relinks news items that reached the news store
after the run was first loaded. A rerun of a (ticker, date) is a new run; use the
`current_runs` view to get the latest complete run per pair.

Market conditions (ATR, RSI, VIX, yield curve…) are append-only too: a missing input
(e.g. no FRED key, or `--no-market`) is recorded as `not_configured`, and
`--backfill-conditions` adds a newer row once it can be filled. Read them through the
`latest_conditions` view; what each row was computed from is in `condition_inputs`.

The outcome-date check needs a trading calendar; the command line builds it from SPY's
prices. Called from Python without `calendar=`, that check reports SKIP rather than OK.

The schema has a version (`schema_info`). If the code's version changes, delete the
database file and load again; the cache files are the source of truth.

DuckDB allows one writing process per file. Close other connections (or open them
with `read_only=True`) before loading.

## 2. Write your component's rows

Use the helpers in [`writers.py`](writers.py). They check every field, derive what
can be derived, and raise `InvalidRecord` listing all problems at once. Nothing is
written unless the whole record is valid. Rows are only ever added: never
`UPDATE` or `DELETE` (see Rules).

```python
from shadow.data.db import connect
from shadow.data.writers import insert_decision, insert_outcome, insert_book_snapshot, insert_memo

con = connect()            # main database; connect(dev_db_path()) for the dev one
```

**Decision logger: one Portfolio Manager decision.** Ticker, date, proposed action and
direction come from the run. A reduced trade is split into an approved and a rejected
part for you. Only the pair's current run (`current_runs`: latest, complete) can be
decided on, and each experiment version gets one decision set per ticker and date.

```python
ids = insert_decision(
    con,
    run_id="42dd0c20a62b",                 # from current_runs
    experiment_version="llm_stats",
    verdict="reduced",                     # approved | reduced | rejected
    proposed_size=1.0, final_size=0.5,     # fractions of the per-ticker allocation
    risk_ratings={"volatility": 2, "event_risk": 1, "valuation": 1,
                  "liquidity": 0, "concentration": 0},   # 0-3 each
    main_reason="volatility",              # a category or "none"
    explanation="ATR is 3.1% of price; halve the position.",
    pm_rating="Overweight",
    memo_id="memo_…",                      # the memo in force, or None
)
# -> ["dec_…approved part", "dec_…rejected part"]
```

**Outcome judge: the 10-day result of one part.** Dates, score and label are fixed here,
so every component uses the same rule: entry is the first trading day after the trade
date (next-day open), and the result is known on the 10th trading day after the entry.
The trading calendar comes from SPY's price history (pass `calendar=` to use your own).

```python
insert_outcome(con, decision_id=ids[0], stock_return=0.031, spy_return=0.008)
# -> {"entry_date": "2026-09-02", "resolved_on": "2026-09-17", "score": 0.023, "label": "correct_approval"}
```

Pass `entry_date` / `resolved_on` only if you want them checked against the calendar;
`horizon_days` must be 10.

Labels: |score| ≤ 1% is `noise`; otherwise an approved part is `correct_approval` or
`approval_loss`, and a rejected part is `missed_gain` or `correct_rejection`.

**Simulator: one day of one book.** Add days in order; drawdown is computed from the
running peak.

```python
insert_book_snapshot(con, experiment_version="llm_stats", book="decision",   # or "shadow"
                     snapshot_date="2026-09-02", cash=40_000.0, value=101_250.0,
                     positions={"NVDA": 120, "JPM": 50})
# -> -0.012 (drawdown)
```

**Calibration Agent: a memo and everything behind it**, in one transaction. The hard
limits are enforced: risk level 1–5, the first memo of a version at level 3, one step
per cycle, and no step up while the decision book is more than 10% below its peak or
when there is no decision-book snapshot from the last 7 days. A hypothesis can only be
checked by a later memo of the same experiment version.

```python
insert_memo(
    con, experiment_version="llm_stats", cycle_date="2026-09-28", risk_level=3,
    memo_text="Event risk: loosen for general economic news, keep for earnings. …",
    advice={
        "event_risk": {"advice": "loosen", "confidence": "medium",
                       "reasoning": "7 of 9 macro-headline rejections missed gains",
                       "cited_ids": ["dec_…", "dec_…"]},
        "volatility": {"advice": "keep", "confidence": "low", "reasoning": "thin evidence", "cited_ids": []},
        # … valuation, liquidity, concentration (all five are required)
    },
    hypotheses=[{"statement": "Rejections over Fed headlines miss gains", "category": "event_risk",
                 "subgroup": {"main_reason": "event_risk", "news_keyword": "fed"}}],
    hypothesis_checks=[{"hypothesis_id": "hyp_…", "status": "held", "evidence": {"n": 6}}],
    tool_calls=[{"tool": "subgroup_stats", "args": {...}, "result": {...}}],
    safety_passed=True, safety_flags={}, model="deepseek/deepseek-v4-flash-0731",
)
```

For anything else there is `loader.append_rows(con, table, rows)`, which checks table
and column names against the schema.

## 3. Read with the query functions (Calibration Agent tools)

```python
from shadow.data.db import connect
from shadow.data.queries import CalibrationTools

con = connect(read_only=True)
tools = CalibrationTools(con, as_of="2026-09-28", experiment_version="llm_stats")

tools.cycle_summary()                     # this cycle and all history, per category; books; risk level
tools.search_trades({"main_reason": "event_risk", "vix": {"min": 20}}, order="worst", limit=10)
tools.read_trade("dec_…")                 # decision, outcome, proposal, conditions, reports, news seen
tools.subgroup_stats({"main_reason": "event_risk", "news_keyword": "fed"}, split_by="verdict")
tools.past_memos(limit=3)                 # earlier memos, advice, and each hypothesis's status as of as_of
```

`as_of` and `experiment_version` are fixed when the tools are created; give the agent
only the methods, so it can pass filters but never move the date. Every function then
enforces the point-in-time rule: a trade is visible only if it was decided **and**
its outcome was resolved on or before `as_of`. Trades still waiting are counted
(`decisions_awaiting_outcome`), never shown. Memos, hypothesis checks and book
snapshots after `as_of` are ignored.

Filters (anything else raises a `ValueError` naming the allowed ones):

| Kind | Names | Value |
|---|---|---|
| Category | `ticker`, `verdict`, `part`, `label`, `main_reason`, `proposed_action`, `pm_rating`, `sentiment_confidence` | a string or a list |
| Range | `trade_date`, `score`, `part_size`, `risk_volatility` … `risk_concentration`, `atr_pct`, `volatility_20d`, `rsi_14`, `pct_vs_sma200`, `vix`, `yield_curve`, `sentiment_score` | `{"min": x, "max": y}` |
| Text | `news_keyword` (headline or summary the run saw) | a string |
| IDs | `decision_ids` | a list |

`split_by`: `ticker`, `verdict`, `part`, `main_reason`, `proposed_action`, `pm_rating`,
`sentiment_confidence`, `month`.

Statistics: success = `correct_rejection` or `correct_approval`; `noise` is left out.
Each group reports counts per label, the success rate, a Beta(1,1)-Binomial posterior
(mean, 90% interval, P(rate > 0.5)), mean score of approved and rejected parts,
`thin_evidence` when fewer than 10 scored trades, and up to 50 decision IDs to cite.

## 4. Produce upstream runs

```powershell
python -m shadow.data.cache_runner --tickers NVDA --dates 2026-09-01 --dry-run
python -m shadow.data.cache_runner --tickers NVDA JPM --dates 2026-09-01 2026-09-02 --max-runs 4 --workers 2
python -m shadow.data.cache_runner --tickers NVDA --dates 2026-09-01 --rerun-incomplete
python -m shadow.data.cache_runner --tickers JPM --dates 2026-09-03 --rerun    # even if complete
python -m shadow.data.cache_runner --tickers JPM --dates 2026-09-03 --sample   # into the sample cache
```

Every command prints the cache folder it writes to. A cached pair is skipped unless
`--rerun-incomplete` or `--rerun` asks for it.

A run costs $0.02–0.034 on OpenRouter and takes 5–9 minutes; the 2 × 5 sample
(9 runs, `--workers 2`) took 35 minutes and $0.22. `--max-runs` (default 1) refuses
bigger batches, and the runner prints the batch's cost at the end. Runs of one ticker
never overlap, so `--workers` helps only with several tickers.

Settings are pinned in [`settings.py`](settings.py): DeepSeek V4 Flash, temperature 0.2,
16384 max tokens, one debate round, news from Alpaca, statements from SEC EDGAR first.

A run is marked incomplete when a report or debate turn is empty, when the Trader's
proposal came back as free text without numbers, or when a data source failed for a
reason a rerun can fix (e.g. a missing API key). When an agent loops to the token cap
and returns nothing (DeepSeek's hidden reasoning sometimes does, mostly the
Fundamentals Analyst), the pair is rerun once at temperature 0.6; the cache file
records the temperature used and `retry_of`. Earlier attempts stay in `_superseded/`.

## 5. Try it on the dev database

```powershell
python -m shadow.data.smoke_test --as-of 2026-09-28 --out smoke.json
```

Loads the sample cache into `warehouse/dev.duckdb`, adds stand-in decisions (a simple ATR rule),
real 10-day outcomes and a template memo under `experiment_version="smoke_test"`, runs
the checks and the five tools. It refuses to touch the main database. The stand-ins
are not the real risk team or outcome judge; they only give the tools realistic rows.
There is no stand-in simulator, so `books` is empty there.

**The sample** (in `warehouse/sample_upstream_cache/`): NVDA and JPM on 2026-09-01, 02, 03,
04 and 08 (09-07 was Labor Day): 12 runs including retries, $0.30. The folder also
keeps the four NVDA 09-01 test runs made while setting up (temperature 0.0 and 8192
tokens; superseded).

| Pair | Result |
|---|---|
| JPM all five days; NVDA 09-01, 09-03, 09-04 | complete at temperature 0.2 (JPM 09-03 on a rerun: the first attempt's Trader answered in free text) |
| NVDA 09-02 | complete on the retry at temperature 0.6 |
| NVDA 09-08 | **known gap**: the Fundamentals Analyst looped to the token cap at 0.2 and again at 0.6. Left incomplete on purpose; `current_runs` has no row for it |

Expect about one run in four to need the 0.6 retry, and the occasional pair that
still fails; check `incomplete_runs` after every batch.

## Rules

- **Append-only.** Never `UPDATE` or `DELETE`. A changed decision is a new
  experiment version; a hypothesis changes status through a new check row.
- **Point-in-time.** Every row keeps the date it is about and when it became known.
  Outcomes carry `resolved_on`; the query functions use it.
- **Numbers in numeric columns**, never only in text.
- **No SQL built from agent text.** Use the query functions' filters.
- **Keep `smoke_test` and samples in `dev.duckdb`.**

## Known limitations

Found in review and deliberately not fixed yet (numbers are the review's).

- **8. Experiments cannot be rerun the append-only way.** `decisions`, `memos` and
  `book_snapshots` are keyed by `experiment_version` with no run ID, so rerunning an
  experiment needs a new version string. To be decided with the group.
- **11. `--max-runs` counts pairs, not LLM runs.** One pair can make up to 8 full runs
  (2 temperatures × 4 rate-limit attempts), so the real cost of a batch can be higher
  than `--max-runs` suggests.
- **15. There is a seventh status, `error`**, a catch-all marked retryable, beyond the
  six listed in CLAUDE.md.
- **17. The ticker identity lookup has no evidence row.** It reads today's yfinance
  profile outside `route_to_vendor`; its text is kept in `outputs.instrument_context`.
- **18. Superseding can overwrite.** It uses `os.replace`, so two cache files without a
  `run_id` both become `…unknown.json`, and a complete cache file can be replaced by an
  incomplete rerun. The database is unaffected: `current_runs` takes the latest
  complete run.
- **19. Conditions use adjusted prices** (`auto_adjust=True`), so `close` is not the
  exact traded price on that date; ratios (ATR %, RSI, distance from the 200-day
  average) are mostly unaffected.
- **20. Filter edge cases.** A non-string `trade_date` filter raises `TypeError`
  instead of `ValueError`, and numeric filters accept NaN and infinity.
- **21. A two-part decision is not written in one transaction.** If the second part's
  insert fails, the first part stays.
- **22. `first_seen_at` falls back to `created_at`** when a news item has no `fetched_at`.

Tests: `python -m pytest shadow/tests -q` (offline, no LLM calls).
