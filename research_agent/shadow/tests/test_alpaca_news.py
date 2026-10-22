"""Alpaca news store: pagination, download-once, and the created_at cut-off. No network."""

from datetime import date

import pytest

from shadow.data import alpaca_news
from shadow.data.alpaca_news import AlpacaNewsStore


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.headers = body, status, {}

    def json(self):
        return self.body

    def raise_for_status(self):
        pass


class FakeSession:
    """Serves two pages for any query and records every request."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        return FakeResponse(self.pages[params.get("page_token", "first")])


def article(i, created, updated=None):
    return {"id": i, "created_at": created, "updated_at": updated or created,
            "headline": f"headline {i}", "summary": f"summary {i}", "author": "a",
            "source": "benzinga", "url": f"https://x/{i}", "symbols": ["NVDA"]}


PAGES = {
    "first": {"news": [article(1, "2026-08-27T13:00:00Z"),
                       article(2, "2026-08-28T23:59:59Z")], "next_page_token": "p2"},
    "p2": {"news": [article(3, "2026-08-29T00:00:00Z"),
                    # edited after the cut-off but created before it: still served
                    article(4, "2026-08-28T10:00:00Z", updated="2026-09-03T08:00:00Z")],
           "next_page_token": None},
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    return AlpacaNewsStore(tmp_path, session=FakeSession(PAGES))


def test_follows_page_token_and_stores_once(store):
    assert store.ensure("NVDA", ["NVDA"], date(2026, 8, 27), date(2026, 8, 29)) == 4
    assert len(store.session.calls) == 2
    assert store.session.calls[1]["page_token"] == "p2"
    # Same range again: served from disk, no request, no duplicates.
    assert store.ensure("NVDA", ["NVDA"], date(2026, 8, 27), date(2026, 8, 29)) == 0
    assert len(store.session.calls) == 2
    assert len(store.load("NVDA")) == 4
    assert {a["fetched_at"] is not None for a in store.load("NVDA")} == {True}


def test_only_missing_days_are_downloaded(store):
    store.ensure("NVDA", ["NVDA"], date(2026, 8, 27), date(2026, 8, 28))
    store.session.calls.clear()
    store.ensure("NVDA", ["NVDA"], date(2026, 8, 27), date(2026, 8, 30))
    assert [c["start"] for c in store.session.calls if "page_token" not in c] == ["2026-08-29T00:00:00Z"]


def test_created_at_cutoff_is_end_of_day_utc(store):
    store.ensure("NVDA", ["NVDA"], date(2026, 8, 27), date(2026, 8, 29))
    ids = [a["id"] for a in store.articles_between("NVDA", date(2026, 8, 27), date(2026, 8, 28))]
    # 3 was created at midnight after the end date, so it is excluded; newest first.
    assert ids == [2, 4, 1]


def test_get_news_text_and_cutoff(store, monkeypatch):
    monkeypatch.setattr(alpaca_news, "default_store", lambda: store)
    text = alpaca_news.get_news("NVDA", "2026-08-27", "2026-08-28")
    assert text.startswith("## NVDA News, from 2026-08-27 to 2026-08-28:")
    assert "### headline 2 (source: Benzinga)" in text
    assert "Published: 2026-08-28 23:59 UTC" in text
    assert "headline 3" not in text
    assert alpaca_news.get_news("NVDA", "2026-08-25", "2026-08-25").startswith("No news found")


def test_register_routes_news_tools_only():
    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import run_config

    alpaca_news.register()
    with run_config(alpaca_news.CONFIG_OVERRIDES):
        assert router.get_vendor("news_data", "get_news") == "alpaca"
        assert router.get_vendor("news_data", "get_global_news") == "alpaca"
        assert router.get_vendor("news_data", "get_insider_transactions") == "yfinance"
    assert router.VENDOR_METHODS["get_news"]["alpaca"] is alpaca_news.get_news


def test_missing_keys_is_not_configured(tmp_path, monkeypatch):
    from tradingagents.dataflows.errors import VendorNotConfiguredError

    monkeypatch.setattr(alpaca_news, "load_env", lambda: None)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    s = AlpacaNewsStore(tmp_path, session=FakeSession(PAGES))
    with pytest.raises(VendorNotConfiguredError):
        s.ensure("NVDA", ["NVDA"], date(2026, 8, 27), date(2026, 8, 27))
