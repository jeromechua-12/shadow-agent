"""Load the upstream cache, raw evidence and news store into DuckDB; append rows for teammates.

Loading is idempotent: a run already in ``runs`` is skipped, and news items are
inserted once. Superseded cache files are loaded too (they are history);
``current_runs`` picks the latest complete run per (ticker, date).

Other components append their rows with ``append_rows`` (decision logger ->
decisions, outcome judge -> outcomes, simulator -> book_snapshots, Calibration
Agent -> memos and friends). Table and column names are checked against the
schema, and values are always bound as parameters.

Usage:
    python -m shadow.data.loader            # load everything into warehouse/calibration.duckdb
    python -m shadow.data.loader --no-market  # skip price/FRED lookups for conditions
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import duckdb

from shadow.data import alpaca_news
from shadow.data.db import init_db
from shadow.data.paths import upstream_cache_dir, warehouse_dir

logger = logging.getLogger(__name__)

MarketProvider = Callable[[str, str], dict]

DIRECTION = {"Buy": 1, "Hold": 0, "Sell": -1}
AGENT_BY_NODE = {"market": "market", "sentiment": "sentiment", "social": "sentiment",
                 "news": "news", "fundamentals": "fundamentals"}


def _agent(node: str | None) -> str | None:
    low = (node or "").lower()
    return next((agent for key, agent in AGENT_BY_NODE.items() if key in low), None)


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    rows = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = ? "
                       "ORDER BY ordinal_position", [table]).fetchall()
    if not rows:
        raise ValueError(f"unknown table {table!r}")
    return [r[0] for r in rows]


def append_rows(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict], ignore_existing: bool = False) -> int:
    """Insert ``rows`` (dicts keyed by column name) into ``table``; return rows inserted.

    Rows are only ever added. With ``ignore_existing`` a row whose primary key
    is already present is skipped (for idempotent reloads); otherwise it raises,
    so a component that writes the same record twice finds out.
    """
    if not rows:
        return 0
    allowed = set(_columns(con, table))
    cols = sorted({c for r in rows for c in r})
    unknown = [c for c in cols if c not in allowed]
    if unknown:
        raise ValueError(f"{table} has no column(s) {unknown}")
    values = [[_json_ready(r.get(c)) for c in cols] for r in rows]
    verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
    before = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    # Table and column names were checked against the schema above; values are bound.
    con.executemany(f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", values)
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] - before


def _json_ready(v):
    """dict values go to JSON columns as text; lists stay lists (DuckDB arrays)."""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v


# ---- news store -> text_items -------------------------------------------------

def load_news_store(con: duckdb.DuckDBPyConnection, store_dir: Path | None = None) -> int:
    """Insert every article in the Alpaca store (all keys) into text_items; return new rows."""
    store = alpaca_news.AlpacaNewsStore(store_dir) if store_dir else alpaca_news.default_store()
    rows = []
    for path in sorted(Path(store.root).glob("*.jsonl")):
        key = path.stem
        for a in store.load(key):
            if not a.get("created_at"):
                continue  # undated: cannot be placed in time, never served either
            rows.append({
                "item_id": f"alpaca_news:{a['id']}", "source": "alpaca_news", "source_id": str(a["id"]),
                "store_key": key, "symbols": a.get("symbols") or [], "created_at": a["created_at"],
                "updated_at": a.get("updated_at"), "first_seen_at": a.get("fetched_at") or a["created_at"],
                "title": a.get("headline"), "body": a.get("summary"), "author": a.get("author"),
                "publisher": a.get("source"), "url": a.get("url"),
            })
    return append_rows(con, "text_items", rows, ignore_existing=True)


# ---- one cache file -> runs and friends ---------------------------------------

def _evidence_calls(record: dict, warehouse: Path | None) -> list[dict]:
    """The run's data calls: the persisted index when present, else the cache file's copy."""
    rel = (record.get("data_sources") or {}).get("evidence_index")
    if rel:
        index = Path(warehouse or warehouse_dir()) / rel
        if index.exists():
            return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
        logger.warning("Evidence index %s missing; using the cache file's summary", index)
    return (record.get("data_sources") or {}).get("calls") or []


def _outputs_rows(run_id: str, record: dict) -> list[dict]:
    out, st = record["outputs"], record.get("structured") or {}
    debate = out.get("investment_debate") or {}
    rows = [
        ("market", out.get("market_report"), None),
        ("sentiment", out.get("sentiment_report"), st.get("sentiment_report")),
        ("news", out.get("news_report"), None),
        ("fundamentals", out.get("fundamentals_report"), None),
        ("bull", debate.get("bull_history"), None),
        ("bear", debate.get("bear_history"), None),
        ("research_manager", out.get("investment_plan"), st.get("research_plan")),
        ("trader", out.get("trader_investment_plan"), st.get("trader_proposal")),
    ]
    return [{"run_id": run_id, "agent": a, "round": 1, "text": t, "structured": s}
            for a, t, s in rows if (t or "").strip()]


def load_cache_file(con: duckdb.DuckDBPyConnection, path: Path, market: MarketProvider | None = None,
                    warehouse: Path | None = None) -> str | None:
    """Load one upstream cache file. Returns its run_id, or None if it was already loaded."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    run_id, ticker, day = record["run_id"], record["ticker"], record["trade_date"]
    if con.execute("SELECT 1 FROM runs WHERE run_id = ?", [run_id]).fetchone():
        return None

    model, settings, usage = record["model"], record["settings"], record.get("usage") or {}
    code = record.get("code_version") or {}
    calls = _evidence_calls(record, warehouse)
    proposal = (record.get("structured") or {}).get("trader_proposal")
    sentiment = (record.get("structured") or {}).get("sentiment_report") or {}

    con.begin()
    try:
        append_rows(con, "runs", [{
            "run_id": run_id, "ticker": ticker, "trade_date": day, "started_at": record["created_at"],
            "duration_s": record.get("duration_s"), "cache_version": record["cache_version"],
            "complete": record["complete"], "incomplete_reasons": record.get("incomplete_reasons") or [],
            "llm_provider": model.get("provider"), "deep_model": model.get("deep_think_llm"),
            "quick_model": model.get("quick_think_llm"), "temperature": model.get("temperature"),
            "max_tokens": model.get("max_tokens"), "debate_rounds": settings.get("max_debate_rounds"),
            "analysts": settings.get("analysts"), "data_vendors": settings.get("data_vendors"),
            "tool_vendors": settings.get("tool_vendors"), "code_commit": code.get("commit"),
            "code_dirty": code.get("dirty"), "llm_calls": usage.get("llm_calls"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "cost_usd": usage.get("cost_usd"), "cache_file": str(path),
        }])
        append_rows(con, "llm_calls", [
            {"run_id": run_id, "seq": i, **{k: c.get(k) for k in (
                "node", "model", "finish_reason", "input_tokens", "output_tokens", "reasoning_tokens",
                "content_chars", "tool_calls", "cost_usd")}}
            for i, c in enumerate(record.get("llm_calls") or [])])

        append_rows(con, "evidence_calls", [{
            "call_id": c["call_id"], "run_id": run_id, "node": c.get("node"), "agent": _agent(c.get("node")),
            "kind": c["kind"], "method": c["method"],
            "args": {"args": c.get("args") or [], "kwargs": c.get("kwargs") or {}},
            "as_of": day, "fetched_at": c["fetched_at"], "vendor": c.get("vendor"),
            "vendors_tried": [a["vendor"] for a in c.get("attempts") or []] or ([c["vendor"]] if c.get("vendor") else []),
            "status": c["status"], "retryable": bool(c.get("retryable")), "detail": c.get("detail"),
            "seconds": c.get("seconds"), "output_chars": c.get("output_chars"),
            "payload_path": c.get("payload_path"), "payload_sha256": c.get("output_sha256"),
        } for c in calls])

        known_items = {r[0] for r in con.execute("SELECT item_id FROM text_items").fetchall()}
        links, missing = [], 0
        for c in calls:
            rank = 0
            for group in c.get("items") or []:
                for source_id in group["ids"]:
                    item_id = f"{group['source']}:{source_id}"
                    if item_id not in known_items:
                        missing += 1
                        continue
                    rank += 1
                    links.append({"run_id": run_id, "call_id": c["call_id"], "item_id": item_id, "rank": rank})
        if missing:
            logger.warning("Run %s: %d served items not in text_items (load the news store first)", run_id, missing)
        append_rows(con, "run_text_items", links, ignore_existing=True)

        append_rows(con, "agent_outputs", _outputs_rows(run_id, record))

        if proposal:
            action = proposal.get("action")
            append_rows(con, "trader_proposals", [{
                "run_id": run_id, "ticker": ticker, "trade_date": day, "action": action,
                "direction": DIRECTION.get(action), "entry_price": proposal.get("entry_price"),
                "stop_loss": proposal.get("stop_loss"), "position_sizing": proposal.get("position_sizing"),
                "reasoning": proposal.get("reasoning"), "structured_ok": True}])
        elif (record["outputs"].get("trader_investment_plan") or "").strip():
            append_rows(con, "trader_proposals", [{
                "run_id": run_id, "ticker": ticker, "trade_date": day,
                "reasoning": record["outputs"]["trader_investment_plan"], "structured_ok": False}])

        by_method: dict[str, set] = {}
        for c in calls:
            by_method.setdefault(c["method"], set()).add(c["status"])
        append_rows(con, "conditions", [{
            "run_id": run_id, "as_of": day,
            **((market(ticker, day) or {}) if market else {}),
            "sentiment_score": sentiment.get("overall_score"),
            "sentiment_band": sentiment.get("overall_band"),
            "sentiment_confidence": sentiment.get("confidence"),
            "news_items_seen": len({link["item_id"] for link in links}),
            "missing_sources": sorted(m for m, s in by_method.items() if "ok" not in s),
        }])
        con.commit()
    except Exception:
        con.rollback()
        raise
    return run_id


def cache_files(cache_dir: Path | None = None) -> list[Path]:
    """Every cache file, current and superseded (failed-run records excluded)."""
    root = Path(cache_dir) if cache_dir else upstream_cache_dir()
    files = list(root.glob("*/*.json")) + list(root.glob("*/_superseded/*.json"))
    return sorted(p for p in files if not p.parent.name.startswith("_tradingagents"))


def load_all(con: duckdb.DuckDBPyConnection | None = None, cache_dir: Path | None = None,
             store_dir: Path | None = None, market: MarketProvider | None = None,
             warehouse: Path | None = None) -> dict:
    """News store first (runs link to its items), then every cache file."""
    con = con or init_db()
    news = load_news_store(con, store_dir)
    loaded, skipped, failed = [], 0, []
    for path in cache_files(cache_dir):
        try:
            run_id = load_cache_file(con, path, market=market, warehouse=warehouse)
        except Exception as e:
            logger.exception("Could not load %s", path)
            failed.append(f"{path}: {e}")
            continue
        if run_id is None:
            skipped += 1
        else:
            loaded.append(run_id)
    return {"news_items_added": news, "runs_loaded": len(loaded), "runs_already_loaded": skipped,
            "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the upstream cache into DuckDB.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--no-market", action="store_true", help="skip price and FRED lookups for conditions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from shadow.data.conditions import market_conditions

    con = init_db(args.db)
    summary = load_all(con, market=None if args.no_market else market_conditions)
    print(json.dumps(summary, indent=1))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
