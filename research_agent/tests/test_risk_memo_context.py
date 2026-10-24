"""Risk memo: the current calibration memo, threaded to the risk team only.

Upstream agents must stay blind to it, so their cached outputs are valid for every setup.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.context import get_risk_memo_from_state
from tradingagents.graph.propagation import Propagator

MEMO = "Calibration memo for feedback as of 2025-03-31\nEvent Risk: loosen (medium confidence)."
RISK_TEAM = [
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
    ("tradingagents.agents.managers.portfolio_manager", "create_portfolio_manager"),
]


class _LLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt, *a, **k):
        self.prompts.append(prompt)
        return AIMessage("Rating: Hold\n\nnothing to do")

    def with_structured_output(self, *a, **k):
        raise NotImplementedError  # force the free-text path


def _prompt(module, factory, memo):
    state = {
        "company_of_interest": "AAPL", "trade_date": "2025-04-01", "asset_type": "stock",
        "instrument_context": "", "market_report": "M", "sentiment_report": "S",
        "news_report": "N", "fundamentals_report": "F", "investment_plan": "P",
        "trader_investment_plan": "T", "past_context": "", "portfolio_context": "",
        "risk_memo": memo, "risk_memo_id": "memo_1" if memo else "",
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": "", "judge_decision": ""},
    }
    llm = _LLM()
    getattr(importlib.import_module(module), factory)(llm)(state)
    return llm.prompts[0]


@pytest.mark.unit
def test_empty_memo_adds_nothing():
    assert get_risk_memo_from_state({}) == get_risk_memo_from_state({"risk_memo": "  "}) == ""


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", RISK_TEAM)
def test_risk_team_sees_the_memo_verbatim(module, factory):
    assert MEMO in _prompt(module, factory, MEMO)


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", RISK_TEAM)
def test_memo_only_adds_its_block(module, factory):
    block = get_risk_memo_from_state({"risk_memo": MEMO})
    assert _prompt(module, factory, MEMO).replace(block, "") == _prompt(module, factory, "")


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
