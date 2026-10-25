"""Risk memo: the current Calibration Agent memo reaches the risk analysts and PM.

Without a memo the prompts must stay exactly as upstream, so the no-feedback
setup is a true baseline. The memo is threaded from ``propagate`` into state
and keyed into the checkpoint signature.
"""

from __future__ import annotations

import importlib

import pytest

MEMO = {"memo_id": "memo_1", "memo_text": "Calibration memo for feedback as of 2026-08-31\nTighten event risk."}

RISK_TEAM = [
    ("tradingagents.agents.managers.portfolio_manager", "create_portfolio_manager"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
]


def _prompt(module, factory, **extra):
    """The prompt one risk-team agent sends for a fixed state plus ``extra``."""
    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            from langchain_core.messages import AIMessage
            seen.append(prompt)
            return AIMessage("Rating: Hold\n\nnothing to do")

        def with_structured_output(self, *a, **k):
            raise NotImplementedError  # force the free-text path

    state = {
        "company_of_interest": "AAPL", "trade_date": "2026-09-01", "asset_type": "stock",
        "instrument_context": "", "market_report": "M", "sentiment_report": "S",
        "news_report": "N", "fundamentals_report": "F", "investment_plan": "P",
        "trader_investment_plan": "T", "past_context": "", "portfolio_context": "",
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": "", "judge_decision": ""},
        **extra,
    }
    getattr(importlib.import_module(module), factory)(_LLM())(state)
    return seen[0]


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", RISK_TEAM)
def test_risk_team_sees_the_memo_text(module, factory):
    assert MEMO["memo_text"] in _prompt(module, factory, risk_memo=MEMO["memo_text"])


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", RISK_TEAM)
def test_no_memo_leaves_the_prompt_unchanged(module, factory):
    """No memo adds nothing; a memo adds only its own text, no stray blank lines elsewhere."""
    without = _prompt(module, factory)
    assert _prompt(module, factory, risk_memo="") == without
    assert _prompt(module, factory, risk_memo=None) == without
    with_memo = _prompt(module, factory, risk_memo="MEMO_MARKER")
    sep = "\n\n" if factory == "create_portfolio_manager" else "\n"
    assert with_memo.replace("MEMO_MARKER" + sep, "", 1) == without


def _bare_graph(tmp_path):
    from tradingagents.decision_log import TradingMemoryLog
    from tradingagents.graph.propagation import Propagator
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_log_path": str(tmp_path / "m.md"), "max_debate_rounds": 1,
                    "max_risk_discuss_rounds": 1}
    graph.memory_log = TradingMemoryLog(graph.config)
    graph.propagator = Propagator()
    graph.selected_analysts = ["market"]
    graph.settle_pending = lambda t: None
    graph.resolve_instrument_context = lambda t, a="stock", d=None: ""
    graph._memory_as_of = lambda d: None
    return graph


@pytest.mark.unit
def test_create_run_state_carries_the_memo(tmp_path):
    graph = _bare_graph(tmp_path)
    state = graph.create_run_state("AAPL", "2026-09-01", memo=MEMO)
    assert (state["risk_memo"], state["risk_memo_id"]) == (MEMO["memo_text"], "memo_1")
    empty = graph.create_run_state("AAPL", "2026-09-01")
    assert (empty["risk_memo"], empty["risk_memo_id"]) == ("", None)


@pytest.mark.unit
def test_checkpoint_signature_changes_with_the_memo(tmp_path):
    graph = _bare_graph(tmp_path)
    other = {"memo_id": "memo_2", "memo_text": "x"}
    sigs = {graph._run_signature("stock"), graph._run_signature("stock", memo=MEMO),
            graph._run_signature("stock", memo=other)}
    assert len(sigs) == 3


@pytest.mark.unit
def test_completed_run_clears_the_checkpoint_of_its_memo(tmp_path, monkeypatch):
    import tradingagents.graph.trading_graph as tg

    graph = _bare_graph(tmp_path)
    graph.config.update({"checkpoint_enabled": True, "data_cache_dir": str(tmp_path),
                         "results_dir": str(tmp_path)})
    graph.debug = False
    graph._resuming = False
    graph.propagator.get_graph_args = lambda callbacks=None: {}
    graph.process_signal = lambda d: "Hold"
    graph._log_state = lambda *a, **k: None
    graph.graph = type("G", (), {"invoke": lambda self, i, **k: {"final_trade_decision": "Rating: Hold\n\nx"}})()

    cleared = []
    monkeypatch.setattr(tg, "clear_checkpoint", lambda d, t, dt, signature: cleared.append(signature))
    graph._run_graph("AAPL", "2026-09-01", "stock", checkpoint_thread_id=None, memo=MEMO)

    assert cleared == [graph._run_signature("stock", memo=MEMO)]
