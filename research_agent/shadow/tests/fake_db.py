"""A small fake evidence database for query and check tests. All values are made up.

Calibration date used by the tests: AS_OF = 2026-09-28, experiment "llm_stats".

Visible at AS_OF (decided and resolved by then):
    D1  NVDA 09-01 approved Buy            +3.0%  correct_approval
    D2  NVDA 09-02 rejected event_risk     +5.0%  missed_gain        (saw a Fed headline)
    D3a AAPL 09-02 reduced, approved part  -4.0%  approval_loss
    D3r AAPL 09-02 reduced, rejected part  -4.0%  correct_rejection
    D4  AAPL 09-10 rejected event_risk     -2.0%  correct_rejection  (saw an earnings headline)
    D8  AAPL 09-03 approved Buy            +0.5%  noise
Hidden at AS_OF:
    D5  NVDA 09-20 rejected event_risk, resolved 10-05 (outcome in the future)
    D6  NVDA 09-30 traded after AS_OF
    D7  NVDA 09-01 under another experiment version
Memos: M1 cycle 08-31 (risk 3); M2 cycle 10-26 (risk 4, future). Hypothesis H1 from M1,
checked "failed" in M2, so it must still read "pending" at AS_OF.
"""

from shadow.data.loader import append_rows

AS_OF = "2026-09-28"
VERSION = "llm_stats"

RUNS = {  # run_id: (ticker, trade_date, vix)
    "r1": ("NVDA", "2026-09-01", 15.0), "r2": ("NVDA", "2026-09-02", 22.0),
    "r3": ("AAPL", "2026-09-02", 22.0), "r4": ("AAPL", "2026-09-10", 18.0),
    "r5": ("NVDA", "2026-09-20", 25.0), "r6": ("NVDA", "2026-09-30", 30.0),
    "r8": ("AAPL", "2026-09-03", 16.0),
}

# decision_id, run, version, verdict, part, proposed, final, part_size, main_reason, memo,
# stock_return, spy_return, resolved_on, label
TRADES = [
    ("D1", "r1", VERSION, "approved", "approved_part", 1.0, 1.0, 1.0, "none", "M1", 0.04, 0.01, "2026-09-16", "correct_approval"),
    ("D2", "r2", VERSION, "rejected", "rejected_part", 1.0, 0.0, 1.0, "event_risk", "M1", 0.06, 0.01, "2026-09-17", "missed_gain"),
    ("D3a", "r3", VERSION, "reduced", "approved_part", 1.0, 0.5, 0.5, "volatility", "M1", -0.03, 0.01, "2026-09-17", "approval_loss"),
    ("D3r", "r3", VERSION, "reduced", "rejected_part", 1.0, 0.5, 0.5, "volatility", "M1", -0.03, 0.01, "2026-09-17", "correct_rejection"),
    ("D4", "r4", VERSION, "rejected", "rejected_part", 1.0, 0.0, 1.0, "event_risk", "M1", -0.01, 0.01, "2026-09-24", "correct_rejection"),
    ("D8", "r8", VERSION, "approved", "approved_part", 1.0, 1.0, 1.0, "none", "M1", 0.015, 0.01, "2026-09-18", "noise"),
    ("D5", "r5", VERSION, "rejected", "rejected_part", 1.0, 0.0, 1.0, "event_risk", "M1", 0.11, 0.01, "2026-10-05", "missed_gain"),
    ("D6", "r6", VERSION, "rejected", "rejected_part", 1.0, 0.0, 1.0, "event_risk", "M1", 0.11, 0.01, "2026-10-14", "missed_gain"),
    ("D7", "r1", "rules_template", "rejected", "rejected_part", 1.0, 0.0, 1.0, "valuation", None, 0.04, 0.01, "2026-09-16", "missed_gain"),
]


def _next_day(day: str) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def build(con) -> None:
    append_rows(con, "runs", [
        {"run_id": r, "ticker": t, "trade_date": d, "started_at": "2026-10-22T10:00:00+00:00",
         "cache_version": 1, "complete": True} for r, (t, d, _) in RUNS.items()])
    append_rows(con, "trader_proposals", [
        {"run_id": r, "ticker": t, "trade_date": d, "action": "Buy", "direction": 1, "entry_price": 100.0,
         "stop_loss": 95.0, "position_sizing": "full", "reasoning": "fake", "structured_ok": True}
        for r, (t, d, _) in RUNS.items()])
    append_rows(con, "conditions", [
        {"run_id": r, "computed_at": "2026-10-22T10:00:00+00:00", "as_of": d, "input_status": {"prices": "ok", "vix": "ok", "yield_curve": "ok"}, "vix": vix, "atr_pct": 0.02, "sentiment_score": 6.0,
         "sentiment_confidence": "medium", "missing_sources": []} for r, (t, d, vix) in RUNS.items()])
    append_rows(con, "agent_outputs", [
        {"run_id": r, "agent": "news", "round": 1, "text": f"news report for {t} on {d} " + "x" * 3000}
        for r, (t, d, _) in RUNS.items()])
    append_rows(con, "evidence_calls", [
        {"call_id": f"c_{r}", "run_id": r, "node": "tools_news", "agent": "news", "kind": "router",
         "method": "get_news", "args": {"args": [t, "2026-08-25", d], "kwargs": {}}, "as_of": d,
         "fetched_at": "2026-10-22T10:00:00+00:00", "vendor": "alpaca", "vendors_tried": ["alpaca"],
         "status": "ok", "retryable": False} for r, (t, d, _) in RUNS.items()])
    append_rows(con, "text_items", [
        {"item_id": "alpaca_news:1", "source": "alpaca_news", "source_id": "1", "store_key": "NVDA",
         "symbols": ["NVDA"], "created_at": "2026-09-01T15:00:00+00:00", "first_seen_at": "2026-10-22T10:00:00+00:00",
         "title": "Fed signals a pause in rate hikes", "publisher": "benzinga"},
        {"item_id": "alpaca_news:2", "source": "alpaca_news", "source_id": "2", "store_key": "AAPL",
         "symbols": ["AAPL"], "created_at": "2026-09-09T12:00:00+00:00", "first_seen_at": "2026-10-22T10:00:00+00:00",
         "title": "Apple earnings preview: services growth in focus", "publisher": "benzinga"},
    ])
    append_rows(con, "run_text_items", [
        {"run_id": "r2", "call_id": "c_r2", "item_id": "alpaca_news:1", "rank": 1},
        {"run_id": "r4", "call_id": "c_r4", "item_id": "alpaca_news:2", "rank": 1},
    ])
    append_rows(con, "memos", [
        {"memo_id": "M1", "experiment_version": VERSION, "cycle_date": "2026-08-31",
         "created_at": "2026-08-31T20:00:00+00:00", "risk_level": 3, "memo_text": "Keep all categories.",
         "safety_passed": True, "safety_flags": {}},
        {"memo_id": "M2", "experiment_version": VERSION, "cycle_date": "2026-10-26",
         "created_at": "2026-10-26T20:00:00+00:00", "risk_level": 4, "previous_risk_level": 3,
         "memo_text": "FUTURE MEMO", "safety_passed": True, "safety_flags": {}},
    ])
    append_rows(con, "memo_advice", [
        {"memo_id": "M1", "category": c, "advice": "keep", "confidence": "low", "reasoning": "thin evidence",
         "cited_ids": []} for c in ("volatility", "event_risk", "valuation", "liquidity", "concentration")])
    append_rows(con, "memo_hypotheses", [
        {"hypothesis_id": "H1", "memo_id": "M1", "category": "event_risk",
         "statement": "Rejections over macro headlines miss gains", "subgroup": {"main_reason": "event_risk"}}])
    append_rows(con, "hypothesis_checks", [
        {"hypothesis_id": "H1", "memo_id": "M2", "status": "failed", "evidence": {"n": 3}}])
    append_rows(con, "decisions", [
        {"decision_id": i, "run_id": r, "experiment_version": v, "memo_id": m, "ticker": RUNS[r][0],
         "trade_date": RUNS[r][1], "decided_at": f"{RUNS[r][1]}T20:00:00+00:00", "part": part,
         "verdict": verdict, "proposed_action": "Buy", "direction": 1, "proposed_size": prop,
         "final_size": final, "part_size": size, "pm_rating": "Buy", "risk_volatility": 1, "risk_event": 2,
         "risk_valuation": 1, "risk_liquidity": 0, "risk_concentration": 0, "main_reason": reason,
         "explanation": f"fake explanation {i}"}
        for i, r, v, verdict, part, prop, final, size, reason, m, *_ in TRADES])
    append_rows(con, "outcomes", [
        {"decision_id": i, "horizon_days": 10, "entry_date": _next_day(RUNS[r][1]), "resolved_on": resolved,
         "stock_return": sr, "spy_return": spy, "score": sr - spy, "label": label}
        for i, r, *_, sr, spy, resolved, label in TRADES])
    snaps = [("2026-08-31", 100.0, 100.0), ("2026-09-15", 103.0, 104.0), ("2026-09-28", 101.0, 106.0),
             ("2026-10-05", 250.0, 90.0)]  # the last one is after AS_OF and must be ignored
    append_rows(con, "book_snapshots", [
        {"snapshot_date": d, "book": book, "experiment_version": VERSION, "cash": 0.0, "value": v,
         "drawdown": min(0.0, v / max(x[i] for x in snaps[: k + 1]) - 1), "positions": {}}
        for k, (d, dv, sv) in enumerate(snaps)
        for book, v, i in (("decision", dv, 1), ("shadow", sv, 2))])
