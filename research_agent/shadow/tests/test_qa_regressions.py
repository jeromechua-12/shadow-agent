"""Focused QA regressions for known edge cases in the shadow pipeline."""

import math

import pytest

from shadow.data import cache_runner
from shadow.data.queries import build_filters


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_numeric_query_bounds_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        build_filters({"score": {"min": value}})


@pytest.mark.parametrize("value", [20260901, None, []])
def test_trade_date_filter_rejects_non_string_bounds_as_value_error(value):
    with pytest.raises(ValueError, match="trade_date"):
        build_filters({"trade_date": {"min": value}})


def test_max_runs_caps_model_attempts_including_token_cap_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOW_UPSTREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache_runner, "STAGGER_S", 0)
    monkeypatch.setattr(cache_runner, "openrouter_spend", lambda: None)
    calls = []

    def fake_run(ticker, trade_date, temperature=0.2):
        calls.append(temperature)
        looped = temperature < cache_runner.RETRY_TEMPERATURE
        return {
            "run_id": f"run-{len(calls)}",
            "complete": not looped,
            "incomplete_reasons": ["empty fundamentals_report"] if looped else [],
            "model": {"temperature": temperature},
            "duration_s": 0,
            "usage": {"cost_usd": 0.01},
            "llm_calls": [{
                "node": "Fundamentals Analyst",
                "finish_reason": "length" if looped else "stop",
                "content_chars": 0 if looped else 100,
                "tool_calls": 0,
            }],
        }

    monkeypatch.setattr(cache_runner, "run_upstream", fake_run)
    result = cache_runner.main([
        "--tickers", "NVDA", "--dates", "2026-09-01", "--max-runs", "1",
    ])

    assert result == 0
    assert len(calls) == 1
    saved = cache_runner.read_cache("NVDA", "2026-09-01")
    assert not saved["complete"]
    assert "max-runs budget exhausted before token-cap retry" in saved["incomplete_reasons"]
