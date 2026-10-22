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
import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from shadow.data import alpaca_news, evidence
from shadow.data.db import init_db
from shadow.data.paths import sample_cache_dir, upstream_cache_dir, warehouse_dir

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


def _served(call: dict) -> list[str]:
    """text_items ids a call showed, in the order shown."""
    return [f"{g['source']}:{i}" for g in call.get("items") or [] for i in g["ids"]]


def _link_items(con: duckdb.DuckDBPyConnection, run_id: str, calls: list[dict]) -> int:
    """Link each call to the items it served that are in text_items; return links added.

    Safe to repeat: an item missing from the news store now is linked on a later
    load, once the store has it. Rank is the item's position in what the call showed.
    """
    known = {r[0] for r in con.execute("SELECT item_id FROM text_items").fetchall()}
    links, missing = [], 0
    for c in calls:
        for rank, item_id in enumerate(_served(c), start=1):
            if item_id in known:
                links.append({"run_id": run_id, "call_id": c["call_id"], "item_id": item_id, "rank": rank})
            else:
                missing += 1
    if missing:
        logger.warning("Run %s: %d served items not in text_items yet; reload once the news store has them",
                       run_id, missing)
    return append_rows(con, "run_text_items", links, ignore_existing=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


NOT_COMPUTED = {"prices": "not_configured", "vix": "not_configured", "yield_curve": "not_configured"}


def _insert_conditions(con, run_id: str, day: str, result: dict | None, base: dict,
                       warehouse: Path | None) -> dict:
    """One conditions row plus its inputs as evidence; returns {input: status}.

    ``result`` is a market provider's answer, or None when conditions were not
    computed (every input is then recorded as not_configured, for a backfill).
    """
    computed_at = _now()
    values, inputs = (result or {}).get("values") or {}, (result or {}).get("inputs") or []
    status = {**NOT_COMPUTED, **{i["input"]: i["status"] for i in inputs}}
    raw = Path(warehouse or warehouse_dir()) / "raw"
    input_rows = []
    for i in inputs:
        payload_path = sha = None
        if i.get("payload") is not None:
            sha = hashlib.sha256(i["payload"].encode("utf-8")).hexdigest()
            rel = Path("payloads") / sha[:2] / f"{sha}.txt"
            evidence._write_once(raw / rel, i["payload"])
            payload_path = rel.as_posix()
        input_rows.append({"run_id": run_id, "computed_at": computed_at, "input": i["input"],
                           "source": i.get("source"), "series": i.get("series"), "status": i["status"],
                           "detail": i.get("detail"), "fetched_at": i.get("fetched_at") or computed_at,
                           "payload_path": payload_path, "payload_sha256": sha})
    append_rows(con, "conditions", [{"run_id": run_id, "computed_at": computed_at, "as_of": day,
                                     "input_status": status, **values, **base}])
    append_rows(con, "condition_inputs", input_rows)
    return status


def backfill_conditions(con: duckdb.DuckDBPyConnection, market: MarketProvider,
                        warehouse: Path | None = None) -> int:
    """Recompute conditions for runs whose latest row is missing an input; return rows added.

    A new row is added (the old one stays) only when it has more inputs ok than
    the latest one, so rerunning while FRED is still down adds nothing.
    """
    rows = con.execute(
        "SELECT c.run_id, r.ticker, c.as_of, CAST(c.input_status AS VARCHAR), c.sentiment_score, c.sentiment_band, "
        "c.sentiment_confidence, c.news_items_seen, c.missing_sources FROM latest_conditions c "
        "JOIN runs r ON r.run_id = c.run_id ORDER BY c.as_of, r.ticker").fetchall()
    added = 0
    for run_id, ticker, as_of, status_json, score, band, conf, seen, missing in rows:
        old = json.loads(status_json or "{}")
        if old and all(v == "ok" for v in old.values()):
            continue
        result = market(ticker, as_of.isoformat()) or {}
        new = {**NOT_COMPUTED, **{i["input"]: i["status"] for i in result.get("inputs") or []}}
        if sum(v == "ok" for v in new.values()) <= sum(v == "ok" for v in old.values()):
            continue
        base = {"sentiment_score": score, "sentiment_band": band, "sentiment_confidence": conf,
                "news_items_seen": seen, "missing_sources": missing}
        con.begin()
        try:
            _insert_conditions(con, run_id, as_of.isoformat(), result, base, warehouse)
            con.commit()
        except Exception:
            con.rollback()
            raise
        added += 1
    return added


def load_cache_file(con: duckdb.DuckDBPyConnection, path: Path, market: MarketProvider | None = None,
                    warehouse: Path | None = None) -> dict:
    """Load one upstream cache file.

    Returns {"run_id", "new": True if the run was added now, "links_added"}. A run
    loaded before is not added again, but its news links are refreshed.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    run_id, ticker, day = record["run_id"], record["ticker"], record["trade_date"]
    if con.execute("SELECT 1 FROM runs WHERE run_id = ?", [run_id]).fetchone():
        return {"run_id": run_id, "new": False, "links_added": _link_items(con, run_id, _evidence_calls(record, warehouse))}

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
            "served_item_ids": _served(c),
        } for c in calls])
        links_added = _link_items(con, run_id, calls)

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
        _insert_conditions(con, run_id, day, (market(ticker, day) or {}) if market else None, {
            "sentiment_score": sentiment.get("overall_score"),
            "sentiment_band": sentiment.get("overall_band"),
            "sentiment_confidence": sentiment.get("confidence"),
            "news_items_seen": len({i for c in calls for i in _served(c)}),
            "missing_sources": sorted(m for m, s in by_method.items() if "ok" not in s),
        }, warehouse)
        con.commit()
    except Exception:
        con.rollback()
        raise
    return {"run_id": run_id, "new": True, "links_added": links_added}


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
    loaded, skipped, links, failed = 0, 0, 0, []
    for path in cache_files(cache_dir):
        try:
            result = load_cache_file(con, path, market=market, warehouse=warehouse)
        except Exception as e:
            logger.exception("Could not load %s", path)
            failed.append(f"{path}: {e}")
            continue
        loaded += result["new"]
        skipped += not result["new"]
        links += result["links_added"]
    return {"news_items_added": news, "runs_loaded": loaded, "runs_already_loaded": skipped,
            "links_added": links, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the upstream cache into DuckDB.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true",
                        help="use warehouse/dev.duckdb, loaded from the sample cache")
    parser.add_argument("--cache-dir", help="cache folder to load (default: upstream_cache, or the sample cache with --dev)")
    parser.add_argument("--no-market", action="store_true",
                        help="skip price and FRED lookups (conditions marked not_configured, backfill later)")
    parser.add_argument("--backfill-conditions", action="store_true",
                        help="after loading, recompute conditions whose inputs were missing (e.g. FRED key added)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from shadow.data.conditions import market_conditions
    from shadow.data.db import resolve_db

    db = resolve_db(args.db, args.dev)
    cache = Path(args.cache_dir) if args.cache_dir else (sample_cache_dir() if args.dev else upstream_cache_dir())
    print(f"loading {cache} into {db}")
    con = init_db(db)
    summary = load_all(con, cache_dir=cache, market=None if args.no_market else market_conditions)
    if args.backfill_conditions:
        summary["conditions_backfilled"] = backfill_conditions(con, market_conditions)
    print(json.dumps(summary, indent=1))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
