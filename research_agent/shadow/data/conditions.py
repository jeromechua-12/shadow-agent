"""Market numbers at decision time, by plain code (no LLM), for the ``conditions`` table.

Every input is cut at the analysis date: prices come from ``load_ohlcv``, which
drops rows after ``as_of``; FRED series are read at the vintage known on
``as_of`` (realtime_start = realtime_end = as_of), so later revisions never leak.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PRICE_ROWS_KEPT = 260


def price_conditions(prices: pd.DataFrame) -> dict:
    """close, atr_pct, volatility_20d, rsi_14, pct_vs_sma200 from rows dated <= as_of.

    ``prices`` needs High, Low and Close columns in date order. A figure with
    too little history is None rather than a number from a short window.
    """
    if prices is None or prices.empty:
        return {}
    high, low, close = (prices[c].astype(float).reset_index(drop=True) for c in ("High", "Low", "Close"))
    last = float(close.iloc[-1])
    prev = close.shift(1)
    true_range = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    returns = close.pct_change()

    def wilder_rsi(n=14):
        if len(close) <= n:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().iloc[-1]
        if loss == 0:
            return 100.0
        return float(100 - 100 / (1 + gain / loss))

    return {
        "close": last,
        "atr_pct": float(true_range.iloc[-14:].mean() / last) if len(close) > 14 else None,
        "volatility_20d": float(returns.iloc[-20:].std() * math.sqrt(252)) if len(close) > 20 else None,
        "rsi_14": wilder_rsi(),
        "pct_vs_sma200": float(last / close.iloc[-200:].mean() - 1) if len(close) >= 200 else None,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fred_input(name: str, series_id: str, as_of: str) -> tuple[float | None, dict]:
    """Latest observation of a FRED series as published on ``as_of``, and its evidence record."""
    rec = {"input": name, "source": "fred", "series": series_id, "status": "ok", "detail": None,
           "fetched_at": _now(), "payload": None}
    key = os.environ.get("FRED_API_KEY")
    if not key:
        rec.update(status="not_configured", detail="FRED_API_KEY is not set")
        return None, rec
    end = date.fromisoformat(as_of)
    params = {
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": (end - timedelta(days=30)).isoformat(), "observation_end": as_of,
        "realtime_start": as_of, "realtime_end": as_of, "sort_order": "desc",
    }
    try:
        resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
    except Exception as e:
        rec.update(status="unavailable", detail=f"{type(e).__name__}: {e}"[:300].replace(key, "<key>"))
        return None, rec
    rec["payload"] = json.dumps(observations, sort_keys=True)
    for obs in observations:
        if obs["value"] not in (".", ""):
            return float(obs["value"]), rec
    rec.update(status="no_data", detail=f"no {series_id} observation in the 30 days to {as_of}")
    return None, rec


def market_conditions(ticker: str, as_of: str) -> dict:
    """Default provider for the loader: prices from Yahoo (cached), VIX and 10y-2y from FRED.

    Returns {"values": {column: number}, "inputs": [evidence record per input]}. A
    missing input leaves its values None and says why in its record, so the
    loader can mark the row for a backfill.
    """
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    prices = {"input": "prices", "source": "yfinance", "series": ticker, "status": "ok", "detail": None,
              "fetched_at": _now(), "payload": None}
    try:
        df = load_ohlcv(ticker, as_of)
        values = price_conditions(df)
        if not values:
            prices.update(status="no_data", detail=f"no prices for {ticker} up to {as_of}")
        else:
            # The rows the numbers were computed from (200-day average needs 200).
            prices["payload"] = df.tail(PRICE_ROWS_KEPT).to_csv(index=False, lineterminator="\n")
    except Exception as e:
        prices.update(status="unavailable", detail=f"{type(e).__name__}: {e}"[:300])
        values = {}
    vix, vix_rec = fred_input("vix", "VIXCLS", as_of)
    curve, curve_rec = fred_input("yield_curve", "T10Y2Y", as_of)
    return {"values": {**values, "vix": vix, "yield_curve": curve}, "inputs": [prices, vix_rec, curve_rec]}


def trading_calendar(through: str | None = None) -> list[date]:
    """NYSE trading days from SPY's price history (about 5 years, up to ``through`` or today)."""
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    df = load_ohlcv("SPY", through or date.today().isoformat())
    return sorted({pd.Timestamp(d).date() for d in df["Date"]})
