"""Run the upstream agents (analysts → researchers → Research Manager → Trader) and cache them.

Each (ticker, date) is run once and saved to ``upstream_cache/<TICKER>/<date>.json``
so every experiment version sees the same Trader proposals. The graph is
compiled with ``interrupt_after=["Trader"]``: the risk team and Portfolio
Manager never run here. There is no portfolio and no memory log, so the cached
outputs depend only on (ticker, date).

A cached (ticker, date) is skipped. ``--rerun-incomplete`` reruns the ones whose
cache file is marked incomplete (a report missing, or a data source that failed
for a reason a rerun could fix, such as FRED without an API key). The earlier
file is kept under ``_superseded/``; nothing is overwritten or deleted.

Usage (from anywhere):
    python -m shadow.data.cache_runner --tickers NVDA --dates 2026-09-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from shadow.data import alpaca_news, evidence
from shadow.data.paths import REPO_ROOT, sample_cache_dir, upstream_cache_dir, warehouse_dir
from shadow.data.settings import (
    ANALYSTS,
    CACHE_VERSION,
    RETRY_TEMPERATURE,
    UPSTREAM_TEMPERATURE,
    model_settings,
    shadow_config,
)
from tradingagents.dataflows.config import run_config

logger = logging.getLogger(__name__)

STOP_AFTER = "Trader"
NEXT_NODE = "Aggressive Analyst"
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BASE_DELAY_S = 30  # doubles per retry: 30, 60, 120 s (+ jitter)
STAGGER_S = 10
REPORT_KEYS = ("market_report", "sentiment_report", "news_report", "fundamentals_report")


class UsageCounter(BaseCallbackHandler):
    """Token totals, plus one diagnostic row per LLM call (node, finish reason, sizes)."""

    def __init__(self):
        self.llm_calls = self.input_tokens = self.output_tokens = 0
        self.calls: list[dict] = []
        self._nodes: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._nodes[run_id] = (metadata or {}).get("langgraph_node")

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        self.llm_calls += 1
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                usage = getattr(msg, "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                meta = getattr(msg, "response_metadata", None) or {}
                extra = getattr(msg, "additional_kwargs", None) or {}
                self.calls.append({
                    "node": self._nodes.get(run_id),
                    "finish_reason": meta.get("finish_reason") or (g.generation_info or {}).get("finish_reason"),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
                    "content_chars": len(msg.content) if msg is not None and isinstance(msg.content, str) else None,
                    "reasoning_chars": len(extra.get("reasoning_content") or extra.get("reasoning") or ""),
                    "tool_calls": len(getattr(msg, "tool_calls", None) or []),
                    # A tool call cut off by the token cap lands here, not in tool_calls.
                    "invalid_tool_calls": len(getattr(msg, "invalid_tool_calls", None) or []),
                    "model": meta.get("model_name"),
                    # OpenRouter reports the call's price in its usage block.
                    "cost_usd": (meta.get("token_usage") or {}).get("cost"),
                })

    def as_dict(self):
        costs = [c["cost_usd"] for c in self.calls if c["cost_usd"] is not None]
        return {"llm_calls": self.llm_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(sum(costs), 6) if costs else None}


def cache_path(ticker: str, trade_date: str) -> Path:
    return upstream_cache_dir() / ticker.upper() / f"{trade_date}.json"


def read_cache(ticker: str, trade_date: str) -> dict | None:
    path = cache_path(ticker, trade_date)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _code_version() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def _is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return "RateLimit" in text or "429" in text


def _incomplete_reasons(outputs: dict, calls: list[dict], structured: dict | None = None) -> list[str]:
    reasons = [f"empty {k}" for k in (*REPORT_KEYS, "investment_plan", "trader_investment_plan")
               if not (outputs.get(k) or "").strip()]
    # The books and the decision logger need the proposal as numbers; a free-text
    # fallback has none, and they are never parsed back out of prose.
    if structured is not None and not structured.get("trader_proposal"):
        reasons.append("trader proposal not structured (free-text fallback)")
    # Each debate turn is stored as "<Speaker>: <reply>"; a blank reply leaves only the label.
    for side in ("bull_history", "bear_history"):
        text = outputs["investment_debate"].get(side) or ""
        if len(text.split(":", 1)[-1].strip()) == 0:
            reasons.append(f"empty {side}")
    for c in calls:
        if c["retryable"] and c["status"] != "ok":
            reasons.append(f"{c['method']} {c['status']}: {(c['detail'] or '')[:120]}")
    return sorted(set(reasons))


def _write_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _supersede(path: Path, old: dict) -> None:
    """Move an existing cache file aside (append-only: it is kept, not deleted)."""
    dest = path.parent / "_superseded" / f"{path.stem}.{old.get('run_id', 'unknown')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, dest)


def run_upstream(ticker: str, trade_date: str, temperature: float = UPSTREAM_TEMPERATURE) -> dict:
    """One upstream run for (ticker, date); returns the cache record (not yet written)."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    alpaca_news.register()
    evidence.install()
    config = shadow_config(temperature)
    if config["llm_provider"] != "openrouter":
        raise RuntimeError(f"llm_provider is {config['llm_provider']!r}; expected openrouter (check research_agent/.env)")

    usage = UsageCounter()
    graph = TradingAgentsGraph(selected_analysts=ANALYSTS, config=config, callbacks=[usage])
    upstream = graph.workflow.compile(checkpointer=InMemorySaver(), interrupt_after=[STOP_AFTER])

    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    with run_config(config), alpaca_news.analysis_date(trade_date), evidence.capture() as rec:
        try:
            # The same initial state propagate() builds, minus the memory log
            # (past_context) and the portfolio, and without settling past decisions.
            init_state = graph.propagator.create_initial_state(
                ticker, trade_date,
                past_context="",
                instrument_context=graph.resolve_instrument_context(ticker, "stock", trade_date),
                portfolio_context="",
            )
            args = graph.propagator.get_graph_args()
            args["config"]["configurable"] = {"thread_id": run_id}
            for _ in upstream.stream(init_state, **args):
                pass
            snapshot = upstream.get_state(args["config"])
        finally:
            # Saved even when the run fails: a fetched news page is evidence either way.
            evidence_index = evidence.persist(rec.calls, run_id, ticker, trade_date)
    if tuple(snapshot.next) != (NEXT_NODE,):
        raise RuntimeError(f"graph stopped at {snapshot.next!r}, expected before {NEXT_NODE!r}")
    state = snapshot.values

    debate = state.get("investment_debate_state") or {}
    outputs = {
        "instrument_context": state.get("instrument_context", ""),
        **{k: state.get(k, "") for k in REPORT_KEYS},
        "investment_debate": {k: debate.get(k, "") for k in
                              ("bull_history", "bear_history", "history", "judge_decision")},
        "investment_plan": state.get("investment_plan", ""),
        "trader_investment_plan": state.get("trader_investment_plan", ""),
    }
    calls = rec.calls
    reasons = _incomplete_reasons(outputs, calls, rec.structured)
    return {
        "cache_version": CACHE_VERSION,
        "run_id": run_id,
        "ticker": ticker.upper(),
        "trade_date": trade_date,
        "created_at": started_at.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 1),
        "complete": not reasons,
        "incomplete_reasons": reasons,
        "model": model_settings(config),
        "settings": {
            "analysts": list(ANALYSTS),
            "max_debate_rounds": config["max_debate_rounds"],
            "data_vendors": config["data_vendors"],
            "tool_vendors": config["tool_vendors"],
            "news_article_limit": config["news_article_limit"],
            "news_max_per_day": alpaca_news.MAX_PER_DAY,
            "global_news_symbols": alpaca_news.GLOBAL_SYMBOLS,
            "stopped_after": STOP_AFTER,
            "portfolio": None,
            "memory_log": None,
        },
        "code_version": _code_version(),
        "usage": usage.as_dict(),
        "llm_calls": usage.calls,
        "outputs": outputs,
        # Structured fields as the agents produced them; None means the agent
        # fell back to free text and only the prose above exists.
        "structured": {k: rec.structured.get(k) for k in ("sentiment_report", "research_plan", "trader_proposal")},
        "data_sources": {
            "evidence_index": evidence_index.relative_to(warehouse_dir()).as_posix(),
            "summary": evidence.source_summary(calls),
            "calls": [evidence.summary(c) for c in calls],
        },
    }


class RateGate:
    """Shared pause: after one worker is rate limited, no worker starts a run until it ends."""

    def __init__(self):
        self._until = 0.0
        self._lock = threading.Lock()

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                left = self._until - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 5))


def run_with_retries(ticker: str, trade_date: str, gate: RateGate | None = None,
                     temperature: float = UPSTREAM_TEMPERATURE) -> dict:
    gate = gate or RateGate()
    delay = RATE_LIMIT_BASE_DELAY_S
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        gate.wait()
        try:
            return run_upstream(ticker, trade_date, temperature)
        except Exception as e:
            if not _is_rate_limit(e) or attempt == RATE_LIMIT_RETRIES:
                raise
            # Jitter so paused workers do not all restart in the same second.
            wait = delay + random.uniform(0, delay / 2)
            logger.warning("Rate limited on %s %s (%s); pausing all workers %.0fs", ticker, trade_date, e, wait)
            gate.pause(wait)
            delay *= 2
    raise AssertionError("unreachable")


def openrouter_spend() -> float | None:
    """Total USD spent on this OpenRouter key so far (None if it cannot be read)."""
    try:
        resp = requests.get("https://openrouter.ai/api/v1/key", timeout=20,
                            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        return float(resp.json()["data"]["usage"])
    except Exception:
        return None


def spend_since(before: float | None, settle_s: int = 30) -> float | None:
    """Spend since ``before``; OpenRouter updates the total a few seconds late."""
    if before is None:
        return None
    deadline = time.monotonic() + settle_s
    after = openrouter_spend()
    while after is not None and after <= before and time.monotonic() < deadline:
        time.sleep(5)
        after = openrouter_spend()
    return None if after is None else round(after - before, 6)


def empty_at_token_cap(record: dict) -> list[str]:
    """Nodes whose reply was cut off by the token cap with no text and no tool call."""
    return sorted({c["node"] for c in record.get("llm_calls") or []
                   if c.get("finish_reason") == "length" and not c.get("content_chars") and not c.get("tool_calls")})


def _run_once_or_retry_hotter(ticker: str, trade_date: str, gate: RateGate, previous: dict | None) -> dict:
    """Run at the pinned temperature; if an agent looped to the token cap, run once more hotter.

    A pair whose previous cached run already looped starts at the retry
    temperature. The looping attempt is kept under _superseded/.
    """
    start = RETRY_TEMPERATURE if previous is not None and empty_at_token_cap(previous) else UPSTREAM_TEMPERATURE
    record = run_with_retries(ticker, trade_date, gate, start)
    looped = empty_at_token_cap(record)
    if record["complete"] or not looped or start == RETRY_TEMPERATURE:
        return record
    print(f"retrying {ticker} {trade_date} at temperature {RETRY_TEMPERATURE}: "
          f"{', '.join(looped)} returned nothing at the token cap", flush=True)
    path = cache_path(ticker, trade_date)
    _write_atomic(path.parent / "_superseded" / f"{trade_date}.{record['run_id']}.json", record)
    retry = run_with_retries(ticker, trade_date, gate, RETRY_TEMPERATURE)
    retry["retry_of"] = {"run_id": record["run_id"], "temperature": start,
                         "reason": f"empty reply at token cap: {', '.join(looped)}",
                         "cost_usd": record["usage"].get("cost_usd")}
    return retry


def _run_and_save(ticker: str, trade_date: str, gate: RateGate, ticker_lock: threading.Lock) -> bool:
    """Run one pair and write its cache file (or a _failed record). True on success."""
    path = cache_path(ticker, trade_date)
    # One run per ticker at a time: runs of the same ticker share price and
    # news cache files.
    with ticker_lock:
        try:
            record = _run_once_or_retry_hotter(ticker, trade_date, gate, read_cache(ticker, trade_date))
        except Exception as e:
            logger.exception("Run failed for %s %s", ticker, trade_date)
            _write_atomic(path.parent / "_failed" / f"{trade_date}.{uuid.uuid4().hex[:12]}.json", {
                "ticker": ticker, "trade_date": trade_date,
                "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": f"{type(e).__name__}: {e}", "cache_version": CACHE_VERSION,
            })
            return False
        old = read_cache(ticker, trade_date)
        if old is not None:
            _supersede(path, old)
        _write_atomic(path, record)
    state = "complete" if record["complete"] else "INCOMPLETE"
    lines = [f"saved {path} ({state}, temperature {record['model']['temperature']}, "
             f"{record['duration_s']}s, {record['usage']})"]
    lines += [f"  - {reason}" for reason in record["incomplete_reasons"]]
    print("\n".join(lines), flush=True)
    return True


def plan(tickers: list[str], dates: list[str], rerun_incomplete: bool,
         rerun: bool = False) -> list[tuple[str, str, str]]:
    """(ticker, date, why) for every pair that needs a run, tickers interleaved.

    Runs of one ticker never overlap, so alternating tickers lets parallel
    workers start different tickers instead of queueing behind one. ``rerun``
    reruns the named pairs even if cached (the old file is kept as superseded).
    """
    todo = []
    for d in dates:
        for t in tickers:
            cached = read_cache(t, d)
            if cached is None:
                todo.append((t, d, "not cached"))
            elif rerun:
                todo.append((t, d, f"rerun requested (replaces run {cached.get('run_id')})"))
            elif rerun_incomplete and not cached.get("complete"):
                todo.append((t, d, "incomplete: " + "; ".join(cached.get("incomplete_reasons", []))))
    return todo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache upstream (analysts → Trader) outputs.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--dates", nargs="+", required=True, help="trading days, YYYY-MM-DD")
    parser.add_argument("--rerun-incomplete", action="store_true",
                        help="also rerun cached pairs marked incomplete")
    parser.add_argument("--max-runs", type=int, default=1,
                        help="refuse to start more LLM runs than this (default 1)")
    parser.add_argument("--workers", type=int, default=1,
                        help="pairs run in parallel (default 1; keep small for OpenRouter limits)")
    parser.add_argument("--rerun", action="store_true",
                        help="rerun the named pairs even if cached (old files kept as superseded)")
    parser.add_argument("--sample", action="store_true",
                        help="write to the sample cache (warehouse/sample_upstream_cache), not the real one")
    parser.add_argument("--dry-run", action="store_true", help="only list what would run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.sample:
        # cache_path() reads this on every call, so the whole batch goes to the sample folder.
        os.environ["SHADOW_UPSTREAM_CACHE_DIR"] = str(sample_cache_dir())
    print(f"cache: {upstream_cache_dir()}")

    tickers = [t.upper() for t in args.tickers]
    todo = plan(tickers, args.dates, args.rerun_incomplete, args.rerun)
    for t, d, why in todo:
        print(f"to run: {t} {d} ({why})")
    if not todo:
        print("nothing to run: all cached")
        return 0
    if args.dry_run:
        return 0
    if len(todo) > args.max_runs:
        print(f"refusing: {len(todo)} runs planned but --max-runs is {args.max_runs}")
        return 2

    gate = RateGate()
    ticker_locks = {t: threading.Lock() for t, _, _ in todo}
    spend_before = openrouter_spend()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = []
        for i, (t, d, _) in enumerate(todo):
            futures.append(pool.submit(_run_and_save, t, d, gate, ticker_locks[t]))
            if i < args.workers - 1:
                time.sleep(STAGGER_S)  # spread the first wave's requests
        ok = sum(f.result() for f in futures)
    spent = spend_since(spend_before)
    cost = f"${spent:.4f}" if spent is not None else "unknown"
    print(f"done: {ok}/{len(todo)} runs saved in {time.perf_counter() - t0:.0f}s; "
          f"OpenRouter spend for this batch: {cost} (key total, includes any other use of the key)")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
