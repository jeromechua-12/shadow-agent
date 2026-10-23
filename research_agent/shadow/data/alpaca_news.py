"""Historical news from Alpaca (Benzinga) as a TradingAgents news vendor.

Yahoo only serves recent news, so a run dated in the past sees none. Alpaca's
news API serves the Benzinga archive by date. Articles are downloaded once into
``<warehouse>/raw/alpaca_news/`` and every later run is served from there, so
two runs for the same (ticker, date) see the same articles.

Point-in-time rule: an article belongs to a run dated D only if its
``created_at`` falls before midnight UTC after D (the same half-open window as
``tradingagents.dataflows.date_window.in_window``). ``updated_at`` is stored but
never used for filtering, since an edit made later must not move an article
into an earlier window.

Storage, one pair of files per query key (a ticker, or ``_global``):
    <KEY>.jsonl          one article per line, append-only, unique by id
    <KEY>.coverage.json  the UTC days already downloaded completely

Usage:
    python -m shadow.data.alpaca_news NVDA AAPL --start 2026-08-25 --end 2026-09-01
    python -m shadow.data.alpaca_news --global --start 2026-08-25 --end 2026-09-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from shadow.data import evidence
from shadow.data.paths import load_env, warehouse_dir
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import in_window
from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorRateLimitError

logger = logging.getLogger(__name__)

API_URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50  # the API maximum
MAX_RETRIES = 5
# At most this many articles per UTC day, so one busy day cannot fill the whole
# article limit and hide the rest of the window.
MAX_PER_DAY = 4
GLOBAL_KEY = "_global"
# Market-wide ETFs standing in for "global news": Alpaca has no topic search,
# and a query with no symbols returns every article for every ticker.
GLOBAL_SYMBOLS = ["SPY", "QQQ", "DIA", "TLT", "GLD", "USO"]

# Config overlay that routes only the two news tools to Alpaca. news_data as a
# whole also covers get_insider_transactions, which Alpaca does not serve.
CONFIG_OVERRIDES = {"tool_vendors": {"get_news": "alpaca", "get_global_news": "alpaca"}}

_lock = threading.Lock()

# The run's analysis date, when one is set. The tools layer already clamps
# requested windows (as_of_window), but the vendor enforces it too, so a direct
# call or a future tool change cannot serve news from after the run's date.
_analysis_date: ContextVar[date | None] = ContextVar("alpaca_news_analysis_date", default=None)


@contextmanager
def analysis_date(day: str | date):
    """Serve no article created after ``day`` (UTC) to calls made inside this block."""
    token = _analysis_date.set(day if isinstance(day, date) else date.fromisoformat(day))
    try:
        yield
    finally:
        _analysis_date.reset(token)


def _clamp(end: date) -> date:
    limit = _analysis_date.get()
    return min(end, limit) if limit is not None else end


def default_store_dir() -> Path:
    return warehouse_dir() / "raw" / "alpaca_news"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _spans(days: list[date]) -> list[tuple[date, date]]:
    """Group sorted days into contiguous (first, last) spans."""
    spans: list[tuple[date, date]] = []
    for d in days:
        if spans and d == spans[-1][1] + timedelta(days=1):
            spans[-1] = (spans[-1][0], d)
        else:
            spans.append((d, d))
    return spans


class AlpacaNewsStore:
    """Local, append-only article store plus the downloader that fills it."""

    def __init__(self, root: Path | None = None, session: requests.Session | None = None):
        self.root = Path(root) if root is not None else default_store_dir()
        self.session = session or requests.Session()

    # ---- files ---------------------------------------------------------------

    def _articles_path(self, key: str) -> Path:
        return self.root / f"{key}.jsonl"

    def _coverage_path(self, key: str) -> Path:
        return self.root / f"{key}.coverage.json"

    def load(self, key: str) -> list[dict]:
        path = self._articles_path(key)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def covered_days(self, key: str) -> set[date]:
        path = self._coverage_path(key)
        if not path.exists():
            return set()
        data = json.loads(path.read_text(encoding="utf-8"))
        return {date.fromisoformat(d) for d in data["days"]}

    def _append(self, key: str, articles: list[dict]) -> int:
        """Append articles whose id is not stored yet; return how many were new."""
        seen = {a["id"] for a in self.load(key)}
        new = [a for a in articles if a["id"] not in seen]
        if new:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._articles_path(key).open("a", encoding="utf-8") as f:
                for a in new:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
        return len(new)

    def _mark_covered(self, key: str, symbols: list[str] | None, days: list[date]) -> None:
        covered = self.covered_days(key) | set(days)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._coverage_path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "key": key,
            "symbols": symbols,
            "days": sorted(d.isoformat() for d in covered),
        }, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    # ---- download ------------------------------------------------------------

    def _headers(self) -> dict:
        load_env()
        key_id = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key_id or not secret:
            raise VendorNotConfiguredError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set")
        return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}

    def _get(self, params: dict) -> dict:
        """One API page, retrying throttles and transient failures with backoff."""
        headers = self._headers()
        delay = 2.0
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.session.get(API_URL, headers=headers, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise VendorRateLimitError(f"Alpaca news unreachable: {e}") from e
            else:
                if resp.status_code in (401, 403):
                    raise VendorNotConfiguredError(f"Alpaca rejected the API keys (HTTP {resp.status_code})")
                if resp.status_code != 429 and resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
                if attempt == MAX_RETRIES:
                    raise VendorRateLimitError(f"Alpaca news HTTP {resp.status_code} after {MAX_RETRIES} retries")
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            logger.warning("Alpaca news request failed (attempt %d); retrying in %.0fs", attempt + 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise AssertionError("unreachable")

    def _fetch_span(self, symbols: list[str] | None, first: date, last: date) -> list[dict]:
        """Every article created in the UTC days [first, last], following page_token."""
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        params = {
            "start": f"{first.isoformat()}T00:00:00Z",
            "end": f"{(last + timedelta(days=1)).isoformat()}T00:00:00Z",
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        out: list[dict] = []
        while True:
            body = self._get(params)
            for a in body.get("news") or []:
                out.append({
                    "id": a["id"],
                    "created_at": a.get("created_at"),
                    "updated_at": a.get("updated_at"),
                    "headline": a.get("headline", ""),
                    "summary": a.get("summary", ""),
                    "author": a.get("author", ""),
                    "source": a.get("source", ""),
                    "url": a.get("url", ""),
                    "symbols": a.get("symbols", []),
                    "fetched_at": fetched_at,
                })
            token = body.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    def ensure(self, key: str, symbols: list[str] | None, start: date, end: date) -> int:
        """Download the days in [start, end] not yet stored for ``key``; return new articles.

        Only days that ended before the download began are marked covered: a
        day still in progress is fetched again next time.
        """
        with _lock:
            covered = self.covered_days(key)
            missing = [d for d in _days(start, end) if d not in covered]
            added = 0
            for first, last in _spans(missing):
                articles = self._fetch_span(symbols, first, last)
                added += self._append(key, articles)
                today = datetime.now(timezone.utc).date()
                self._mark_covered(key, symbols, [d for d in _days(first, last) if d < today])
                logger.info("Alpaca news %s %s..%s: %d fetched, %d new", key, first, last, len(articles), added)
            return added

    # ---- point-in-time reads -------------------------------------------------

    def articles_between(self, key: str, start: date, end: date) -> list[dict]:
        """Stored articles created in [start, end + 1 day) UTC, newest first."""
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
        hits = []
        for a in self.load(key):
            created = _parse_ts(a.get("created_at"))
            # An undated article cannot be placed in time, so it is never served.
            if created is not None and in_window(created, start_dt, end_dt):
                hits.append((created, a))
        hits.sort(key=lambda t: t[0], reverse=True)
        return [a for _, a in hits]


_default_store: AlpacaNewsStore | None = None


def default_store() -> AlpacaNewsStore:
    global _default_store
    if _default_store is None or _default_store.root != default_store_dir():
        _default_store = AlpacaNewsStore()
    return _default_store


def spread_by_day(articles: list[dict], total: int, per_day: int = MAX_PER_DAY) -> list[dict]:
    """Up to ``total`` articles, at most ``per_day`` from each UTC day.

    ``articles`` must be newest first; the result keeps that order, so the
    newest days fill first and each day contributes its newest articles.
    """
    taken: Counter = Counter()
    out = []
    for a in articles:
        day = _parse_ts(a["created_at"]).astimezone(timezone.utc).date()
        if taken[day] < per_day:
            taken[day] += 1
            out.append(a)
            if len(out) == total:
                break
    return out


def _format(articles: list[dict]) -> str:
    """Article list in the Yahoo vendor's layout, plus the creation time."""
    out = ""
    for a in articles:
        source = (a.get("source") or "unknown").capitalize()
        out += f"### {a['headline']} (source: {source})\n"
        created = _parse_ts(a.get("created_at"))
        if created is not None:
            out += f"Published: {created.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC\n"
        if a.get("summary"):
            out += f"{a['summary']}\n"
        if a.get("url"):
            out += f"Link: {a['url']}\n"
        out += "\n"
    return out


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Company news for ``ticker`` created in [start_date, end_date] (UTC days)."""
    store = default_store()
    symbol = ticker.upper()
    start, end = date.fromisoformat(start_date), _clamp(date.fromisoformat(end_date))
    end_date = end.isoformat()
    if start > end:
        evidence.note_items("alpaca_news", symbol, [])
        return f"No news found for {ticker} between {start_date} and {end_date}"
    store.ensure(symbol, [symbol], start, end)
    articles = spread_by_day(store.articles_between(symbol, start, end), get_config()["news_article_limit"])
    evidence.note_items("alpaca_news", symbol, [a["id"] for a in articles])
    if not articles:
        return f"No news found for {ticker} between {start_date} and {end_date}"
    return f"## {ticker} News, from {start_date} to {end_date}:\n\n{_format(articles)}"


def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Market-wide news (articles tagged with a broad-market ETF) up to ``curr_date``."""
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    end = _clamp(date.fromisoformat(curr_date))
    curr_date = end.isoformat()
    start = end - timedelta(days=look_back_days)
    store = default_store()
    store.ensure(GLOBAL_KEY, GLOBAL_SYMBOLS, start, end)
    articles = spread_by_day(store.articles_between(GLOBAL_KEY, start, end), limit)
    evidence.note_items("alpaca_news", GLOBAL_KEY, [a["id"] for a in articles])
    if not articles:
        return f"No global news found between {start.isoformat()} and {curr_date}"
    return f"## Global Market News, from {start.isoformat()} to {curr_date}:\n\n{_format(articles)}"


def register() -> None:
    """Add "alpaca" to the router's vendor table. Idempotent.

    The router looks ``VENDOR_METHODS`` up at call time, so adding entries here
    is enough; nothing under ``tradingagents/`` changes. Select the vendor with
    ``CONFIG_OVERRIDES``.
    """
    from tradingagents.dataflows import router

    router.VENDOR_METHODS["get_news"]["alpaca"] = get_news
    router.VENDOR_METHODS["get_global_news"]["alpaca"] = get_global_news
    if "alpaca" not in router.VENDOR_LIST:
        router.VENDOR_LIST.append("alpaca")


def daily_counts(store: AlpacaNewsStore, key: str, start: date, end: date) -> dict[date, int]:
    """Articles per UTC creation day in [start, end], zero-filled."""
    counts = Counter(
        _parse_ts(a["created_at"]).astimezone(timezone.utc).date()
        for a in store.articles_between(key, start, end)
    )
    return {d: counts.get(d, 0) for d in _days(start, end)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Alpaca news into the local store.")
    parser.add_argument("tickers", nargs="*", help="ticker symbols")
    parser.add_argument("--global", dest="global_news", action="store_true",
                        help=f"also download market-wide news ({', '.join(GLOBAL_SYMBOLS)})")
    parser.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last day, YYYY-MM-DD")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    store = default_store()
    jobs = [(t.upper(), [t.upper()]) for t in args.tickers]
    if args.global_news:
        jobs.append((GLOBAL_KEY, GLOBAL_SYMBOLS))
    for key, symbols in jobs:
        added = store.ensure(key, symbols, start, end)
        counts = daily_counts(store, key, start, end)
        print(f"{key}: {added} new, {sum(counts.values())} stored in range")
        for d, n in counts.items():
            print(f"  {d} {d:%a}  {n}")


if __name__ == "__main__":
    main()
