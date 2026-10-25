# Two-book simulator (`shadow/sim`)

Runs the **decision book** (the Portfolio Manager's final ratings) and the **shadow book**
(the Trader's proposals in full) over the same prices and costs, and writes one
`book_snapshots` row per book per trading day. Offline: no LLM calls.

```powershell
python -m shadow.sim.simulator --experiment-version llm_stats --book decision --to 2026-09-28
python -m shadow.sim.simulator --experiment-version llm_stats --book shadow   --to 2026-09-28
python -m shadow.sim.simulator --experiment-version llm_stats --book decision --to 2026-09-28 --dry-run
```

Call it for both books every calibration cycle, up to the cycle date, **before** the
Calibration Agent runs (`insert_memo` needs a decision-book snapshot from the last 7 days
to allow a risk increase).

## Rules

| | |
|---|---|
| Ratings → positions | Buy 100% of the per-ticker allocation, Overweight 50%, Hold no order (unchanged), Underweight halve what is held, Sell exit. Long-only. |
| Decision book follows | `decisions.pm_rating` (must be filled for every decision) |
| Shadow book follows | `decisions.proposed_action` (Trader's Buy / Sell) |
| Rejected trade | the PM rated it Hold, so the decision book keeps what it held |
| Fills | next trading day's open; marked at that day's close |
| Money | $100k per book, allocation $10k per ticker, 10 bps per trade, cap 1× allocation (`config.py`) |
| Cash short | sells settle first; remaining buys scaled by one common factor |

## Safety

- **Never ahead of the decisions.** A book stops on the first trading day after the latest
  decision. If the last days had no decisions (all Holds), pass `--decisions-through DATE`.
  Snapshots are append-only, so a day written too early could never be corrected.
- **A missing price stops the run** at that day (prices are read as reported, not gap-filled).
  Fix the data and run again; it resumes from the last snapshot.
- Resuming is automatic: state is rebuilt from the latest snapshot's cash and positions.

Tests: `python -m pytest shadow/tests/test_sim_book.py shadow/tests/test_sim_simulator.py -q`
