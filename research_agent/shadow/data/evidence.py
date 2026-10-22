"""Record every data call an upstream run makes, without editing tradingagents.

Hooks, installed once by ``install()`` and inert unless a ``capture()`` block is
active in the calling context:

- ``tradingagents.agents.tools.route_to_vendor``: one record per tool data call
  (method, args, which vendor served it, status, output, fetch time).
- every implementation in ``router.VENDOR_METHODS``: one attempt per vendor
  tried, so a fallback chain such as "sec_edgar,yfinance" shows who served it.
- the sentiment analyst's StockTwits and Reddit fetches, which bypass the router.
- the render functions of the sentiment, research-plan and trader schemas, so
  their structured fields are kept as numbers before they become prose.
- a log handler on the ``tradingagents`` logger: vendor warnings logged during
  a call are attached to it (that is where an HTTP 403 shows up).

Status values: ok, no_data, unavailable, withheld, blocked, not_configured,
error. ``retryable`` marks failures a later rerun could fix (missing key,
throttling, network); a feed that never serves past dates is not retryable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from shadow.data.paths import warehouse_dir
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)

logger = logging.getLogger(__name__)

STATUSES = ("ok", "no_data", "unavailable", "withheld", "blocked", "not_configured", "error")

# Methods whose vendors deliberately withhold present-day-only data for past dates.
_WITHHOLDING = {"get_fundamentals", "get_prediction_markets"}
# Phrases of a source that only serves recent items (date_window.coverage_gap).
_NO_HISTORY = ("only serves recent items", "window extends past today")


@dataclass
class Recorder:
    calls: list[dict] = field(default_factory=list)
    structured: dict[str, dict | None] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, call: dict) -> None:
        with self._lock:
            self.calls.append(call)


_recorder: ContextVar[Recorder | None] = ContextVar("shadow_evidence_recorder", default=None)
_current_call: ContextVar[dict | None] = ContextVar("shadow_evidence_call", default=None)


@contextmanager
def capture():
    """Record every data call made in this context (and in tool threads it spawns)."""
    rec = Recorder()
    token = _recorder.set(rec)
    try:
        yield rec
    finally:
        _recorder.reset(token)


# ---- classification ----------------------------------------------------------

def _exc_status(exc: BaseException) -> tuple[str, bool]:
    if isinstance(exc, VendorNotConfiguredError):
        return "not_configured", True
    if isinstance(exc, VendorRateLimitError):
        return "unavailable", True
    if isinstance(exc, NoMarketDataError):
        return "no_data", False
    return "error", True


def _text_status(method: str, text: str) -> tuple[str, bool]:
    """Status of a returned string, from the sentinels the vendors and router emit."""
    head = text.lstrip()[:600]
    low = head.lower()
    if head.startswith("NO_DATA_AVAILABLE"):
        return "no_data", False
    if head.startswith("DATA_UNAVAILABLE"):
        return "unavailable", True
    if method in _WITHHOLDING and "withheld" in low:
        return "withheld", False
    if head.startswith("<") and "unavailable" in low:
        return "unavailable", not any(p in low for p in _NO_HISTORY)
    if head.startswith("<no ") or (head.startswith("No ") and " found" in head[:120]):
        return "no_data", False
    if head.startswith("FRED: "):  # invalid indicator name supplied by the model
        return "error", False
    if not head:
        return "no_data", False
    return "ok", False


def _refine(call: dict) -> None:
    """Sharpen a failed call's status using its vendor attempts and logged warnings."""
    if call["status"] == "ok":
        return
    attempts = call["attempts"]
    if attempts and all(a["status"] == "not_configured" for a in attempts):
        call["status"], call["retryable"] = "not_configured", True
    evidence = " ".join([call.get("detail") or "", *call["log"], *(a.get("detail") or "" for a in attempts)])
    if "403" in evidence or "Forbidden" in evidence:
        call["status"], call["retryable"] = "blocked", False
    if not call.get("detail"):
        failed = [a for a in attempts if a["status"] != "ok"]
        call["detail"] = (failed[-1].get("detail") if failed else None) or (call["log"][-1] if call["log"] else None)


# ---- records -----------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _node() -> str | None:
    """The LangGraph node making the call (tool calls run inside a ToolNode)."""
    try:
        from langgraph.config import get_config

        return get_config().get("metadata", {}).get("langgraph_node")
    except Exception:
        return None


def _new_call(kind: str, method: str, args: tuple, kwargs: dict) -> dict:
    return {
        "call_id": uuid.uuid4().hex[:12],
        "kind": kind,  # router | social
        "node": _node(),
        "method": method,
        "args": [str(a) if not isinstance(a, (int, float, type(None))) else a for a in args],
        "kwargs": {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for k, v in kwargs.items()},
        "fetched_at": _now(),
        "seconds": None,
        "vendor": None,
        "status": None,
        "retryable": False,
        "detail": None,
        "attempts": [],
        "log": [],
        "items": [],
        "output": None,
    }


def note_items(source: str, key: str, ids: list) -> None:
    """Record which stored text items (e.g. news articles) the current call served.

    Called by vendors that serve itemised text, so a run can be linked to the
    exact articles its agents saw. No-op outside a captured call.
    """
    call = _current_call.get()
    if call is not None:
        call["items"].append({"source": source, "key": key, "ids": [str(i) for i in ids]})


def _finish(call: dict, started: float, output: str | None) -> None:
    call["seconds"] = round(time.perf_counter() - started, 3)
    if output is not None:
        call["output"] = output
        call["output_chars"] = len(output)
        call["output_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()


def summary(call: dict) -> dict:
    """The call without its output text (for compact listings)."""
    return {k: v for k, v in call.items() if k != "output"}


# ---- wrappers ----------------------------------------------------------------

def _wrap_route(original):
    def route_to_vendor(method, *args, **kwargs):
        rec = _recorder.get()
        if rec is None:
            return original(method, *args, **kwargs)
        _wrap_vendor_table()
        call = _new_call("router", method, args, kwargs)
        started = time.perf_counter()
        token = _current_call.set(call)
        try:
            out = original(method, *args, **kwargs)
        except Exception as e:
            call["status"], call["retryable"] = _exc_status(e)
            call["detail"] = f"{type(e).__name__}: {e}"
            _finish(call, started, None)
            _refine(call)
            rec.add(call)
            raise
        finally:
            _current_call.reset(token)
        text = out if isinstance(out, str) else str(out)
        call["status"], call["retryable"] = _text_status(method, text)
        served = [a["vendor"] for a in call["attempts"] if a["status"] == "ok"]
        call["vendor"] = served[-1] if served else None
        if call["status"] != "ok":
            call["detail"] = text.strip()[:200]
        _finish(call, started, text)
        _refine(call)
        rec.add(call)
        return out

    route_to_vendor.__shadow_wrapped__ = original
    return route_to_vendor


def _wrap_vendor(method: str, vendor: str, impl):
    def vendor_impl(*args, **kwargs):
        call = _current_call.get()
        if call is None:
            return impl(*args, **kwargs)
        attempt = {"vendor": vendor, "status": None, "detail": None}
        started = time.perf_counter()
        try:
            out = impl(*args, **kwargs)
        except Exception as e:
            attempt["status"] = _exc_status(e)[0]
            attempt["detail"] = f"{type(e).__name__}: {e}"
            raise
        else:
            # A vendor can report failure in its return value instead of raising.
            text = out if isinstance(out, str) else str(out)
            status = _text_status(method, text)[0]
            attempt["status"] = "ok" if status in ("ok", "withheld") else status
            if attempt["status"] != "ok":
                attempt["detail"] = text.strip()[:200]
            return out
        finally:
            attempt["seconds"] = round(time.perf_counter() - started, 3)
            call["attempts"].append(attempt)

    vendor_impl.__shadow_wrapped__ = impl
    return vendor_impl


def _wrap_vendor_table() -> None:
    """Wrap every vendor implementation not wrapped yet (vendors can be added late)."""
    from tradingagents.dataflows import router

    for method, impls in router.VENDOR_METHODS.items():
        for vendor, impl in list(impls.items()):
            if not hasattr(impl, "__shadow_wrapped__"):
                impls[vendor] = _wrap_vendor(method, vendor, impl)


def _wrap_social(source: str, original):
    def fetch(ticker, *args, **kwargs):
        rec = _recorder.get()
        if rec is None:
            return original(ticker, *args, **kwargs)
        call = _new_call("social", original.__name__, (ticker, *args), kwargs)
        call["vendor"] = source
        started = time.perf_counter()
        token = _current_call.set(call)
        try:
            out = original(ticker, *args, **kwargs)
        finally:
            _current_call.reset(token)
        call["status"], call["retryable"] = _text_status(original.__name__, out)
        if call["status"] != "ok":
            call["detail"] = out.strip()[:200]
        _finish(call, started, out)
        _refine(call)
        rec.add(call)
        return out

    fetch.__shadow_wrapped__ = original
    return fetch


def _wrap_render(name: str, original):
    def render(obj):
        rec = _recorder.get()
        if rec is not None:
            try:
                rec.structured[name] = obj.model_dump(mode="json")
            except Exception:  # never let capture break a run
                logger.exception("Could not capture structured %s", name)
        return original(obj)

    render.__shadow_wrapped__ = original
    return render


class _CallLogHandler(logging.Handler):
    """Attach tradingagents warnings to the data call that was running when they fired."""

    def emit(self, record: logging.LogRecord) -> None:
        call = _current_call.get()
        if call is not None:
            with suppress(Exception):
                call["log"].append(f"{record.name}: {record.getMessage()}")


_installed = False


def install() -> None:
    """Install every hook. Idempotent."""
    global _installed
    if _installed:
        return
    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    import tradingagents.agents.managers.research_manager as research_manager
    import tradingagents.agents.tools as tools
    import tradingagents.agents.trader.trader as trader

    tools.route_to_vendor = _wrap_route(tools.route_to_vendor)
    sentiment.fetch_stocktwits_messages = _wrap_social("stocktwits", sentiment.fetch_stocktwits_messages)
    sentiment.fetch_reddit_posts = _wrap_social("reddit", sentiment.fetch_reddit_posts)
    sentiment.render_sentiment_report = _wrap_render("sentiment_report", sentiment.render_sentiment_report)
    research_manager.render_research_plan = _wrap_render("research_plan", research_manager.render_research_plan)
    trader.render_trader_proposal = _wrap_render("trader_proposal", trader.render_trader_proposal)

    handler = _CallLogHandler(level=logging.WARNING)
    logging.getLogger("tradingagents").addHandler(handler)
    _wrap_vendor_table()
    _installed = True


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def persist(calls: list[dict], run_id: str, ticker: str, trade_date: str, root: Path | None = None) -> Path:
    """Save a run's data calls under ``<root>`` (default ``<warehouse>/raw``).

    payloads/<sha[:2]>/<sha>.txt
        each distinct output once, named by its SHA-256 (identical price
        tables fetched by several runs are stored once); never rewritten.
    evidence/<TICKER>/<date>/<run_id>.jsonl
        one line per call: method, args, vendor, status, reason, vendor
        attempts, fetched_at, items served, and the payload path. A new
        run_id gives a new file, so nothing is overwritten.

    Sets ``payload_path`` on each call and returns the index path.
    """
    root = Path(root) if root is not None else warehouse_dir() / "raw"
    lines = []
    for c in calls:
        if c.get("output") is not None:
            sha = c["output_sha256"]
            rel = Path("payloads") / sha[:2] / f"{sha}.txt"
            _write_once(root / rel, c["output"])
            c["payload_path"] = rel.as_posix()
        else:
            c["payload_path"] = None
        lines.append(json.dumps({"run_id": run_id, "ticker": ticker.upper(), "trade_date": trade_date,
                                 **summary(c)}, ensure_ascii=False))
    index = root / "evidence" / ticker.upper() / trade_date / f"{run_id}.jsonl"
    if index.exists():
        raise FileExistsError(f"evidence index already exists: {index}")
    _write_once(index, "\n".join(lines) + "\n")
    return index


def load_payload(call: dict, root: Path | None = None) -> str | None:
    """The saved output of a persisted call, verified against its hash."""
    if not call.get("payload_path"):
        return None
    root = Path(root) if root is not None else warehouse_dir() / "raw"
    text = (root / call["payload_path"]).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != call["output_sha256"]:
        raise ValueError(f"payload hash mismatch: {call['payload_path']}")
    return text


def source_summary(calls: list[dict]) -> dict[str, dict]:
    """Per data source: call count, status counts, vendors that served it."""
    out: dict[str, dict] = {}
    for c in calls:
        s = out.setdefault(c["method"], {"calls": 0, "statuses": {}, "vendors": [], "details": []})
        s["calls"] += 1
        s["statuses"][c["status"]] = s["statuses"].get(c["status"], 0) + 1
        if c["vendor"] and c["vendor"] not in s["vendors"]:
            s["vendors"].append(c["vendor"])
        if c["status"] != "ok" and c["detail"] and c["detail"] not in s["details"]:
            s["details"].append(c["detail"][:200])
    return out
