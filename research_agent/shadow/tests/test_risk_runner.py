"""Risk-stage runner: fake LLM, temporary DuckDB, fake prices; nothing touches the network."""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from shadow.data import risk_runner
from shadow.data.db import connect, init_db
from shadow.data.loader import append_rows
from shadow.data.writers import CATEGORIES, insert_memo
from tradingagents.agents import schemas
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph

DAYS = [date(2026, 9, 1) + timedelta(days=i) for i in range(12)]
CACHED = DAYS[1:6]                       # five cached dates; cycle_days=2 -> cycles on CACHED[2], CACHED[4]
PROPOSALS = {"NVDA": "Buy", "AAPL": "Sell", "MSFT": "Hold"}


class FakeModel(BaseChatModel):
    """Answers the debaters with text; the PM's structured call returns ``decision`` or fails."""

    decision: schemas.PortfolioDecision | None = None
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def with_structured_output(self, schema, **kwargs):
        def answer(prompt):
            self.calls.append(prompt)
            if self.decision is None:
                raise ValueError("no structured output")
            return self.decision
        return RunnableLambda(answer)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="**Rating**: Hold\n\nargument"))])


def _decision(rating: str) -> schemas.PortfolioDecision:
    return schemas.PortfolioDecision(rating=rating, executive_summary="s", investment_thesis="t",
                                     volatility=2, event_risk=1, valuation=0, liquidity=0, concentration=3,
                                     main_reason="volatility")


def _graph(tmp_path, monkeypatch, model):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=None, max_risk_discuss_rounds=1)

    class Client:
        def get_llm(self):
            return model
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: Client())
    return trading_graph.TradingAgentsGraph(selected_analysts=("market",), config=cfg)


def _prices(ticker, as_of):
    return pd.DataFrame({"Date": [d.isoformat() for d in DAYS], "Open": 100.0, "Close": 101.0})


def _fake_calibration(cycles):
    def calibrate(*, as_of, experiment_version, db_path):
        con = connect(db_path)
        try:
            append_rows(con, "calibration_attempts", [{
                "attempt_id": f"att_{as_of}", "calibration_run_id": f"cal_{as_of}",
                "experiment_version": experiment_version, "cycle_date": as_of, "attempt_no": 1,
                "created_at": "2026-10-25T00:00:00+00:00", "model": "fake", "status": "ok",
                "safety_passed": True}])
            insert_memo(con, experiment_version=experiment_version, cycle_date=as_of, risk_level=3,
                        memo_text=f"MEMO {as_of}", safety_passed=True,
                        advice={c: {"advice": "keep", "confidence": "low", "reasoning": "r"} for c in CATEGORIES})
        finally:
            con.close()
        cycles.append(as_of)
    return calibrate


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A DuckDB with three tickers on five cached dates, and their cache files."""
    monkeypatch.setenv("SHADOW_UPSTREAM_CACHE_DIR", str(tmp_path / "upstream"))
    path = tmp_path / "test.duckdb"
    init_db(path).close()
    con = connect(path)
    runs, proposals = [], []
    for day in CACHED:
        for ticker, action in PROPOSALS.items():
            run_id = f"{ticker}_{day}"
            runs.append({"run_id": run_id, "ticker": ticker, "trade_date": day,
                         "started_at": "2026-10-22T10:00:00+00:00", "cache_version": 1, "complete": True})
            proposals.append({"run_id": run_id, "ticker": ticker, "trade_date": day, "action": action,
                              "direction": {"Buy": 1, "Hold": 0, "Sell": -1}[action], "reasoning": "r",
                              "structured_ok": True})
            cache = tmp_path / "upstream" / ticker / f"{day}.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "run_id": run_id,
                "outputs": {"instrument_context": f"{ticker} Inc.", "market_report": "m",
                            "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
                            "investment_debate": {"bull_history": "b", "bear_history": "b", "history": "h",
                                                  "judge_decision": "j"},
                            "investment_plan": "plan", "trader_investment_plan": f"FINAL: **{action.upper()}**"},
            }), encoding="utf-8")
    append_rows(con, "runs", runs)
    append_rows(con, "trader_proposals", proposals)
    con.close()
    return path


def _run(path, graph, tmp_path, cycles, **kw):
    return risk_runner.run(path, graph, cycle_days=2, checkpoint_dir=tmp_path / "ckpt", calendar=DAYS,
                           price_provider=_prices, calibrate=_fake_calibration(cycles), **kw)


@pytest.mark.unit
@pytest.mark.parametrize("action, pm, expected", [
    ("Buy", "Buy", (1.0, "approved", "Buy")),
    ("Buy", "Overweight", (0.5, "reduced", "Overweight")),
    ("Buy", "Hold", (0.0, "rejected", "Hold")),
    ("Buy", "Sell", (0.0, "rejected", "Hold")),
    ("Sell", "Sell", (1.0, "approved", "Sell")),
    ("Sell", "Underweight", (0.5, "reduced", "Underweight")),
    ("Sell", "Overweight", (0.0, "rejected", "Hold")),
])
def test_sizes_follow_the_pm_rating_and_a_rejection_is_logged_as_hold(action, pm, expected):
    assert risk_runner.sized_decision(action, pm) == expected


@pytest.mark.unit
def test_cycles_count_from_the_first_cached_date(setup):
    con = connect(setup, read_only=True)
    dates, cycles = risk_runner.schedule(con, cycle_days=2)
    assert dates == CACHED and cycles == {CACHED[2], CACHED[4]}
    assert risk_runner.schedule(con, through=CACHED[3], cycle_days=2) == (CACHED[:4], {CACHED[2]})
    con.close()


@pytest.mark.unit
def test_a_full_replay_logs_both_setups_and_feeds_the_memo_to_feedback_only(setup, tmp_path, monkeypatch):
    model = FakeModel(decision=_decision("Overweight"))
    cycles = []
    done = _run(setup, _graph(tmp_path, monkeypatch, model), tmp_path, cycles)

    assert cycles == [CACHED[2].isoformat(), CACHED[4].isoformat()]
    assert done == {"decisions": 2 * 2 * len(CACHED), "no_decision": 0, "cycles": 2}
    con = connect(setup, read_only=True)
    rows = con.execute("SELECT experiment_version, ticker, trade_date, verdict, pm_rating, memo_id, risk_volatility, "
                       "risk_concentration, main_reason FROM decisions ORDER BY 1, 3, 2").fetchall()
    assert {r[1] for r in rows} == {"NVDA", "AAPL"}                     # the Hold proposal is skipped
    nvda = {r[0]: r for r in rows if r[1] == "NVDA" and r[2] == CACHED[0]}
    assert nvda["original"][3:5] == ("reduced", "Overweight")
    aapl = [r for r in rows if r[1] == "AAPL" and r[2] == CACHED[0]][0]
    assert aapl[3:5] == ("rejected", "Hold")                            # Trader Sell, PM Overweight
    assert all(r[5] is None for r in rows if r[0] == "original")
    memos = dict(con.execute("SELECT cycle_date, memo_id FROM memos").fetchall())
    for r in rows:
        if r[0] == "feedback":
            in_force = [c for c in memos if c <= r[2]]
            assert r[5] == (memos[max(in_force)] if in_force else None)
    assert rows[0][6:] == (2, 3, "volatility")
    books = con.execute("SELECT experiment_version, book, max(snapshot_date) FROM book_snapshots "
                        "GROUP BY 1, 2").fetchall()
    assert {(v, b) for v, b, _ in books} == {(v, b) for v in risk_runner.VERSIONS for b in ("decision", "shadow")}
    assert all(last == CACHED[-1] for _, _, last in books)
    con.close()


@pytest.mark.unit
def test_the_risk_team_sees_the_memo_and_the_decision_book_with_drawdown(setup, tmp_path, monkeypatch):
    model = FakeModel(decision=_decision("Buy"))
    _run(setup, _graph(tmp_path, monkeypatch, model), tmp_path, [], through=CACHED[2])
    pm_prompts = [c for c in model.calls if isinstance(c, str)]
    assert any(f"MEMO {CACHED[2]}" in p for p in pm_prompts)
    assert all("MEMO" not in p for p in pm_prompts[:4])                 # before the first cycle
    assert all("Drawdown from peak:" in p and "Cash available" in p for p in pm_prompts)
    assert any("Current position in NVDA" in p for p in pm_prompts)


@pytest.mark.unit
def test_a_free_text_pm_logs_nothing_and_a_restart_reruns_nothing(setup, tmp_path, monkeypatch):
    model = FakeModel(decision=None)
    graph = _graph(tmp_path, monkeypatch, model)
    done = _run(setup, graph, tmp_path, [], through=CACHED[0])
    assert done == {"decisions": 0, "no_decision": 4, "cycles": 0}
    calls = len(model.calls)
    con = connect(setup, read_only=True)
    assert con.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0
    con.close()

    again = _run(setup, graph, tmp_path, [], through=CACHED[0])
    assert again["decisions"] == 0 and len(model.calls) == calls        # finished threads are not re-run


@pytest.mark.unit
def test_a_restart_skips_logged_decisions(setup, tmp_path, monkeypatch):
    model = FakeModel(decision=_decision("Buy"))
    graph = _graph(tmp_path, monkeypatch, model)
    _run(setup, graph, tmp_path, [], through=CACHED[1])
    calls = len(model.calls)
    assert _run(setup, graph, tmp_path, [], through=CACHED[1])["decisions"] == 0
    assert len(model.calls) == calls
