"""Raw evidence capture: statuses, vendor attribution, persistence. Fake vendors only."""

import json
import logging

import pytest

from shadow.data import evidence
from tradingagents.dataflows import router
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)


@pytest.fixture(autouse=True)
def hooks():
    evidence.install()


def fake(result=None, exc=None):
    def impl(*args, **kwargs):
        if exc is not None:
            raise exc
        return result
    return impl


def route(method, *args):
    import tradingagents.agents.tools as tools  # the wrapped name the tools call
    return tools.route_to_vendor(method, *args)


@pytest.mark.parametrize("method, text, status, retryable", [
    ("get_stock_data", "Date,Open\n2026-09-01,1", "ok", False),
    ("get_stock_data", "NO_DATA_AVAILABLE: No usable market data for 'X'", "no_data", False),
    ("get_macro_indicators", "DATA_UNAVAILABLE: optional macro_data could not be retrieved", "unavailable", True),
    ("get_fundamentals", "# Company Fundamentals for NVDA\n\nProfile fundamentals are withheld for this date.", "withheld", False),
    ("get_news", "Headline: why shares were withheld from sale", "ok", False),  # 'withheld' in news text is just news
    ("fetch_reddit_posts", "<Reddit search unavailable for 2026-08-25..2026-09-01: it only serves recent items>", "unavailable", False),
    ("fetch_stocktwits_messages", "<stocktwits unavailable: HTTPError>", "unavailable", True),
    ("get_news", "No news found for NVDA between 2026-08-25 and 2026-09-01", "no_data", False),
    ("fetch_stocktwits_messages", "<no StockTwits messages for $NVDA within 2026-08-25..2026-09-01>", "no_data", False),
])
def test_text_status(method, text, status, retryable):
    assert evidence._text_status(method, text) == (status, retryable)


def test_fallback_chain_records_serving_vendor(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_balance_sheet", {
        "sec_edgar": fake(exc=NoMarketDataError("NVDA", "NVDA", "no quarterly balance sheet filed")),
        "yfinance": fake("Date,Total Assets\n2026-06-30,1"),
    })
    with run_config({"tool_vendors": {"get_balance_sheet": "sec_edgar,yfinance"}}), evidence.capture() as rec:
        route("get_balance_sheet", "NVDA", "quarterly", "2026-09-01")
    (call,) = rec.calls
    assert (call["vendor"], call["status"]) == ("yfinance", "ok")
    assert [(a["vendor"], a["status"]) for a in call["attempts"]] == [("sec_edgar", "no_data"), ("yfinance", "ok")]
    assert call["args"] == ["NVDA", "quarterly", "2026-09-01"]
    assert call["fetched_at"] and call["output_sha256"]


def test_missing_key_is_not_configured(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_macro_indicators",
                        {"fred": fake(exc=VendorNotConfiguredError("FRED_API_KEY is not set"))})
    with run_config({}), evidence.capture() as rec:
        out = route("get_macro_indicators", "vix", "2026-09-01", 90)
    assert out.startswith("DATA_UNAVAILABLE")  # the router degrades optional macro data
    (call,) = rec.calls
    assert (call["status"], call["retryable"], call["vendor"]) == ("not_configured", True, None)
    assert "FRED_API_KEY" in call["detail"]


def test_throttled_vendor_is_unavailable_and_retryable(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_stock_data",
                        {"yfinance": fake(exc=VendorRateLimitError("429 Too Many Requests"))})
    with run_config({"data_vendors": {"core_stock_apis": "yfinance"}}), evidence.capture() as rec:
        route("get_stock_data", "NVDA", "2026-08-01", "2026-09-01")
    assert (rec.calls[0]["status"], rec.calls[0]["retryable"]) == ("unavailable", True)


def test_raising_call_is_recorded_then_reraised(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_stock_data", {"yfinance": fake(exc=KeyError("boom"))})
    with run_config({"data_vendors": {"core_stock_apis": "yfinance"}}), evidence.capture() as rec, \
            pytest.raises(KeyError):
        route("get_stock_data", "NVDA", "2026-08-01", "2026-09-01")
    assert rec.calls[0]["status"] == "error"


def test_social_403_is_blocked():
    def fetch_stocktwits_messages(ticker, **kwargs):
        logging.getLogger("tradingagents.dataflows.vendors.stocktwits").warning(
            "StockTwits fetch failed for %s: HTTP Error 403: Forbidden", ticker)
        return "<stocktwits unavailable: HTTPError>"

    wrapped = evidence._wrap_social("stocktwits", fetch_stocktwits_messages)
    with evidence.capture() as rec:
        wrapped("NVDA", limit=30)
    call = rec.calls[0]
    assert (call["vendor"], call["status"], call["retryable"]) == ("stocktwits", "blocked", False)
    assert any("403" in line for line in call["log"])


@pytest.mark.parametrize("attr, source", [("fetch_reddit_posts", "reddit"),
                                          ("fetch_stocktwits_messages", "stocktwits")])
def test_dropped_social_source_is_not_called_and_not_retryable(monkeypatch, attr, source):
    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    from tradingagents.dataflows.vendors import reddit, stocktwits

    def no_network(*args, **kwargs):
        raise AssertionError("a dropped source must not be fetched")
    monkeypatch.setattr(reddit, "urlopen", no_network)
    monkeypatch.setattr(stocktwits, "urlopen", no_network)
    with evidence.capture() as rec:
        out = getattr(sentiment, attr)("NVDA", start_date="2026-08-25", end_date="2026-09-01", screen=None)
    call = rec.calls[0]
    assert "dropped from the experiment" in out and "NVDA" in out
    assert (call["method"], call["vendor"], call["status"], call["retryable"]) == (attr, source, "unavailable", False)


def test_inert_without_capture(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_stock_data", {"yfinance": fake("rows")})
    with run_config({"data_vendors": {"core_stock_apis": "yfinance"}}):
        assert route("get_stock_data", "NVDA", "2026-08-01", "2026-09-01") == "rows"


def test_note_items_links_served_articles(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_news", {"alpaca": lambda *a: (
        evidence.note_items("alpaca_news", "NVDA", [101, 102]) or "## NVDA News")})
    with run_config({"tool_vendors": {"get_news": "alpaca"}}), evidence.capture() as rec:
        route("get_news", "NVDA", "2026-08-25", "2026-09-01")
    assert rec.calls[0]["items"] == [{"source": "alpaca_news", "key": "NVDA", "ids": ["101", "102"]}]


def test_payload_bytes_match_hash_with_newlines(tmp_path, monkeypatch):
    """Multi-line output keeps its exact bytes on disk (no \\r\\n translation on Windows)."""
    import hashlib

    text = "Date,Close\n2026-09-01,217.2\n"
    monkeypatch.setitem(router.VENDOR_METHODS, "get_stock_data", {"yfinance": fake(text)})
    with run_config({"data_vendors": {"core_stock_apis": "yfinance"}}), evidence.capture() as rec:
        route("get_stock_data", "NVDA", "2026-08-01", "2026-09-01")
    evidence.persist(rec.calls, "run1", "NVDA", "2026-09-01", root=tmp_path)
    (payload,) = (tmp_path / "payloads").rglob("*.txt")
    assert payload.read_bytes() == text.encode()
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == payload.stem


def test_persist_dedupes_payloads_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS, "get_stock_data", {"yfinance": fake("same table")})
    with run_config({"data_vendors": {"core_stock_apis": "yfinance"}}), evidence.capture() as rec:
        route("get_stock_data", "NVDA", "2026-08-01", "2026-09-01")
        route("get_stock_data", "NVDA", "2026-08-01", "2026-09-01")
    index = evidence.persist(rec.calls, "run1", "nvda", "2026-09-01", root=tmp_path)

    assert index == tmp_path / "evidence" / "NVDA" / "2026-09-01" / "run1.jsonl"
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all("output" not in r for r in rows)
    assert rows[0]["payload_path"] == rows[1]["payload_path"]
    assert len(list((tmp_path / "payloads").rglob("*.txt"))) == 1
    assert evidence.load_payload(rows[0], root=tmp_path) == "same table"

    with pytest.raises(FileExistsError):
        evidence.persist(rec.calls, "run1", "NVDA", "2026-09-01", root=tmp_path)

    (tmp_path / rows[0]["payload_path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence.load_payload(rows[0], root=tmp_path)
