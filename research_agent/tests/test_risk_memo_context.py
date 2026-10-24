"""Risk memo: the current calibration memo, threaded to the risk team only.

Upstream agents must stay blind to it, so their cached outputs are valid for every setup.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from tradingagents.graph.propagation import Propagator


@pytest.mark.unit
def test_initial_state_has_no_memo_by_default():
    state = Propagator().create_initial_state("AAPL", "2025-04-01")
    assert state["risk_memo"] == "" and state["risk_memo_id"] == ""


@pytest.mark.unit
def test_initial_state_carries_the_memo():
    state = Propagator().create_initial_state("AAPL", "2025-04-01", risk_memo="MEMO", risk_memo_id="memo_1")
    assert (state["risk_memo"], state["risk_memo_id"]) == ("MEMO", "memo_1")


@pytest.mark.unit
@pytest.mark.parametrize("module", [
    "tradingagents.agents.analysts.market_analyst",
    "tradingagents.agents.analysts.news_analyst",
    "tradingagents.agents.analysts.sentiment_analyst",
    "tradingagents.agents.analysts.fundamentals_analyst",
    "tradingagents.agents.researchers.bull_researcher",
    "tradingagents.agents.researchers.bear_researcher",
    "tradingagents.agents.managers.research_manager",
    "tradingagents.agents.trader.trader",
])
def test_upstream_agents_stay_blind_to_the_memo(module):
    assert "risk_memo" not in inspect.getsource(importlib.import_module(module))
