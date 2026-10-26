"""Focused QA regressions for known edge cases in the shadow pipeline."""

import math

import pytest

from shadow.data.queries import build_filters


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_numeric_query_bounds_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        build_filters({"score": {"min": value}})


@pytest.mark.parametrize("value", [20260901, None, []])
def test_trade_date_filter_rejects_non_string_bounds_as_value_error(value):
    with pytest.raises(ValueError, match="trade_date"):
        build_filters({"trade_date": {"min": value}})
