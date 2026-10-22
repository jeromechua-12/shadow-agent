"""Market numbers at decision time, by plain code (no LLM), for the ``conditions`` table.

Every input is cut at the analysis date: prices come from ``load_ohlcv``, which
drops rows after ``as_of``; FRED series are read at the vintage known on
``as_of`` (realtime_start = realtime_end = as_of), so later revisions never leak.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)


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


def fred_value_as_of(series_id: str, as_of: str) -> float | None:
    """Latest observation of a FRED series as it was published on ``as_of``."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return None
    end = date.fromisoformat(as_of)
    params = {
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": (end - timedelta(days=30)).isoformat(), "observation_end": as_of,
        "realtime_start": as_of, "realtime_end": as_of, "sort_order": "desc",
    }
    try:
        resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
        resp.raise_for_status()
        for obs in resp.json().get("observations", []):
            if obs["value"] not in (".", ""):
                return float(obs["value"])
    except Exception as e:
        logger.warning("FRED %s as of %s unavailable: %s", series_id, as_of, e)
    return None


def market_conditions(ticker: str, as_of: str) -> dict:
    """Default provider for the loader: prices from Yahoo (cached), VIX and 10y-2y from FRED."""
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    try:
        out = price_conditions(load_ohlcv(ticker, as_of))
    except Exception as e:
        logger.warning("Prices for %s as of %s unavailable: %s", ticker, as_of, e)
        out = {}
    out["vix"] = fred_value_as_of("VIXCLS", as_of)
    out["yield_curve"] = fred_value_as_of("T10Y2Y", as_of)
    return out
