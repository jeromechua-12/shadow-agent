"""Read-only query functions for the Calibration Agent.

The calibration date (``as_of``) and experiment version are fixed when the tools
are created, never taken from the agent. Every function enforces the
point-in-time rule itself:

- a trade is visible only if it was decided on or before ``as_of`` and its
  10-day outcome was resolved on or before ``as_of``;
- book snapshots, memos and hypothesis checks dated after ``as_of`` are ignored;
- trades still waiting for their outcome are only ever counted, never shown.

Agent-supplied filters are checked against an allowlist and bound as
parameters; no SQL is built from agent text. Results are plain JSON-ready
dicts so they can be returned as tool results and logged in memo_tool_calls.

Usage:
    tools = CalibrationTools(con, as_of="2026-10-26", experiment_version="llm_stats")
    tools.cycle_summary()
    tools.subgroup_stats({"main_reason": "event_risk", "news_keyword": "fed"}, split_by="verdict")
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import duckdb
import numpy as np

CATEGORIES = ("volatility", "event_risk", "valuation", "liquidity", "concentration")
SUCCESS = {"correct_rejection", "correct_approval"}
FAILURE = {"missed_gain", "approval_loss"}
LABELS = SUCCESS | FAILURE | {"noise"}

# filter name -> (SQL expression, allowed values or None for free text)
CATEGORICAL = {
    "ticker": ("d.ticker", None),
    "verdict": ("d.verdict", {"approved", "reduced", "rejected"}),
    "part": ("d.part", {"approved_part", "rejected_part"}),
    "label": ("o.label", LABELS),
    "main_reason": ("d.main_reason", set(CATEGORIES) | {"none"}),
    "proposed_action": ("d.proposed_action", {"Buy", "Hold", "Sell"}),
    "pm_rating": ("d.pm_rating", {"Buy", "Overweight", "Hold", "Underweight", "Sell"}),
    "sentiment_confidence": ("c.sentiment_confidence", {"low", "medium", "high"}),
}
# filter name -> SQL expression; value is {"min": x, "max": y} (either may be omitted)
RANGES = {
    "trade_date": "d.trade_date", "score": "o.score", "part_size": "d.part_size",
    "risk_volatility": "d.risk_volatility", "risk_event": "d.risk_event",
    "risk_valuation": "d.risk_valuation", "risk_liquidity": "d.risk_liquidity",
    "risk_concentration": "d.risk_concentration",
    "atr_pct": "c.atr_pct", "volatility_20d": "c.volatility_20d", "rsi_14": "c.rsi_14",
    "pct_vs_sma200": "c.pct_vs_sma200", "vix": "c.vix", "yield_curve": "c.yield_curve",
    "sentiment_score": "c.sentiment_score",
}
SPLITS = {"ticker", "verdict", "part", "main_reason", "proposed_action", "pm_rating",
          "sentiment_confidence", "month"}
ORDERS = {"recent": "d.trade_date DESC, d.decision_id",
          "worst": "o.score ASC, d.decision_id", "best": "o.score DESC, d.decision_id"}
TRADE_COLUMNS = (
    "decision_id", "run_id", "ticker", "trade_date", "verdict", "part", "proposed_action", "direction",
    "proposed_size", "final_size", "part_size", "pm_rating", "main_reason", "risk_volatility",
    "risk_event", "risk_valuation", "risk_liquidity", "risk_concentration", "resolved_on",
    "stock_return", "spy_return", "score", "label", "atr_pct", "vix", "rsi_14", "sentiment_score",
    "sentiment_confidence",
)

# A run dated D may only have seen items created before midnight UTC after D. Built
# with timezone('UTC', ...) so the cutoff does not depend on the session's TimeZone.
_NEWS_CUTOFF = "timezone('UTC', CAST(d.trade_date + 1 AS TIMESTAMP))"

_KNOWN_TRADES = """
SELECT d.decision_id, d.run_id, d.ticker, d.trade_date, d.verdict, d.part, d.proposed_action,
       d.direction, d.proposed_size, d.final_size, d.part_size, d.pm_rating, d.main_reason,
       d.risk_volatility, d.risk_event, d.risk_valuation, d.risk_liquidity, d.risk_concentration,
       d.explanation, d.memo_id,
       o.resolved_on, o.stock_return, o.spy_return, o.score, o.label,
       c.atr_pct, c.vix, c.rsi_14, c.sentiment_score, c.sentiment_confidence
FROM decisions d
JOIN outcomes o ON o.decision_id = d.decision_id
LEFT JOIN latest_conditions c ON c.run_id = d.run_id
WHERE d.experiment_version = ? AND d.trade_date <= ? AND o.resolved_on <= ?
"""


def _plain(v):
    if isinstance(v, datetime) and v.tzinfo is not None:
        return v.astimezone(timezone.utc).isoformat()  # same text whatever the session timezone
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _dicts(cur: duckdb.DuckDBPyConnection, json_cols: tuple = ()) -> list[dict]:
    names = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        rec = {n: _plain(v) for n, v in zip(names, row, strict=True)}
        for c in json_cols:
            if isinstance(rec.get(c), str):
                rec[c] = json.loads(rec[c])
        out.append(rec)
    return out


def beta_stats(successes: int, failures: int, prior: tuple[float, float] = (1.0, 1.0)) -> dict:
    """Beta-Binomial posterior of the success rate: mean, 90% interval, P(rate > 0.5)."""
    a, b = prior[0] + successes, prior[1] + failures
    x = np.linspace(0, 1, 20001)[1:-1]
    log_pdf = (a - 1) * np.log(x) + (b - 1) * np.log1p(-x)
    pdf = np.exp(log_pdf - log_pdf.max())
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return {
        "posterior_mean": round(a / (a + b), 4),
        "ci90": [round(float(np.interp(0.05, cdf, x)), 4), round(float(np.interp(0.95, cdf, x)), 4)],
        "p_rate_above_half": round(float(1 - np.interp(0.5, x, cdf)), 4),
        "prior": list(prior),
    }


def group_stats(rows: list[dict], max_ids: int = 50) -> dict:
    """Counts, success rate (noise excluded), Beta-Binomial posterior, mean scores by part."""
    labels = dict.fromkeys(sorted(LABELS), 0)
    for r in rows:
        labels[r["label"]] += 1
    s = sum(labels[lab] for lab in SUCCESS)
    f = sum(labels[lab] for lab in FAILURE)

    def mean_score(part):
        vals = [r["score"] for r in rows if r["part"] == part]
        return round(sum(vals) / len(vals), 6) if vals else None

    return {
        "n": len(rows),
        "labels": labels,
        "n_scored": s + f,
        "successes": s,
        "success_rate": round(s / (s + f), 4) if s + f else None,
        "beta": beta_stats(s, f),
        "mean_score_approved_parts": mean_score("approved_part"),
        "mean_score_rejected_parts": mean_score("rejected_part"),
        "thin_evidence": (s + f) < 10,
        "decision_ids": [r["decision_id"] for r in sorted(rows, key=lambda r: r["trade_date"], reverse=True)][:max_ids],
    }


def _number(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"filter {name!r}: min/max must be numbers, got {v!r}")
    return v


def build_filters(filters: dict | None) -> tuple[str, list]:
    """SQL fragment (starting with AND) and parameters for an agent's filter dict."""
    sql, params = [], []
    for name, value in (filters or {}).items():
        if name in CATEGORICAL:
            expr, allowed = CATEGORICAL[name]
            values = value if isinstance(value, list) else [value]
            if not values or not all(isinstance(v, str) for v in values):
                raise ValueError(f"filter {name!r} takes a string or a list of strings")
            if allowed is not None and (bad := [v for v in values if v not in allowed]):
                raise ValueError(f"filter {name!r}: unknown value(s) {bad}; allowed: {sorted(allowed)}")
            sql.append(f"AND {expr} IN ({', '.join('?' * len(values))})")
            params += [v.upper() if name == "ticker" else v for v in values]
        elif name in RANGES:
            if not isinstance(value, dict) or not set(value) <= {"min", "max"} or not value:
                raise ValueError(f'filter {name!r} takes {{"min": ..., "max": ...}}')
            for bound, op in (("min", ">="), ("max", "<=")):
                if bound in value:
                    v = value[bound]
                    v = date.fromisoformat(v) if name == "trade_date" else _number(name, v)
                    sql.append(f"AND {RANGES[name]} {op} ?")
                    params.append(v)
        elif name == "news_keyword":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("filter 'news_keyword' takes a non-empty string")
            sql.append("AND EXISTS (SELECT 1 FROM run_text_items rti JOIN text_items ti ON ti.item_id = rti.item_id "
                       "WHERE rti.run_id = d.run_id AND ti.created_at < " + _NEWS_CUTOFF + " "
                       "AND contains(lower(coalesce(ti.title, '') || ' ' || coalesce(ti.body, '')), lower(?)))")
            params.append(value.strip())
        elif name == "decision_ids":
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                raise ValueError("filter 'decision_ids' takes a non-empty list of strings")
            sql.append(f"AND d.decision_id IN ({', '.join('?' * len(value))})")
            params += value
        else:
            allowed = sorted([*CATEGORICAL, *RANGES, "news_keyword", "decision_ids"])
            raise ValueError(f"unknown filter {name!r}; allowed: {allowed}")
    return " ".join(sql), params


class CalibrationTools:
    """The five read-only tools, bound to one calibration date and experiment version."""

    def __init__(self, con: duckdb.DuckDBPyConnection, as_of: str | date, experiment_version: str):
        self.con = con
        self.as_of = as_of if isinstance(as_of, date) else date.fromisoformat(as_of)
        self.version = experiment_version

    def _known(self, filters: dict | None = None, order: str = "recent", limit: int | None = None) -> list[dict]:
        where, params = build_filters(filters)
        sql = f"{_KNOWN_TRADES} {where} ORDER BY {ORDERS[order]}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return _dicts(self.con.execute(sql, [self.version, self.as_of, self.as_of, *params]))

    # ---- tools -----------------------------------------------------------------

    def cycle_summary(self, window_days: int = 28) -> dict:
        """This cycle (outcomes resolved in the last ``window_days``) and all history, per category."""
        start = self.as_of - timedelta(days=window_days)
        rows = self._known()
        window = [r for r in rows if r["resolved_on"] > start.isoformat()]
        by_category = {
            cat: {"this_cycle": group_stats([r for r in window if r["main_reason"] == cat], max_ids=20),
                  "all_history": group_stats([r for r in rows if r["main_reason"] == cat], max_ids=0)}
            for cat in (*CATEGORIES, "none")
        }
        pending = self.con.execute(
            "SELECT count(*) FROM decisions d LEFT JOIN outcomes o ON o.decision_id = d.decision_id "
            "AND o.resolved_on <= ? WHERE d.experiment_version = ? AND d.trade_date <= ? AND o.decision_id IS NULL",
            [self.as_of, self.version, self.as_of]).fetchone()[0]
        memo = self.con.execute(
            "SELECT memo_id, cycle_date, risk_level FROM memos WHERE experiment_version = ? AND cycle_date < ? "
            "ORDER BY cycle_date DESC LIMIT 1", [self.version, self.as_of]).fetchone()
        return {
            "as_of": self.as_of.isoformat(),
            "experiment_version": self.version,
            "window": {"from_exclusive": start.isoformat(), "to": self.as_of.isoformat()},
            "this_cycle": group_stats(window, max_ids=20),
            "all_history": group_stats(rows, max_ids=0),
            "by_category": by_category,
            "decisions_awaiting_outcome": pending,
            "current_memo": None if memo is None else
                {"memo_id": memo[0], "cycle_date": memo[1].isoformat(), "risk_level": memo[2]},
            "books": self._books(start),
        }

    def _books(self, start: date) -> dict:
        snaps = _dicts(self.con.execute(
            "SELECT book, snapshot_date, value, drawdown FROM book_snapshots "
            "WHERE experiment_version = ? AND snapshot_date <= ? ORDER BY book, snapshot_date",
            [self.version, self.as_of]))
        out = {}
        for book in ("decision", "shadow"):
            s = [r for r in snaps if r["book"] == book]
            if not s:
                continue
            before = [r for r in s if r["snapshot_date"] <= start.isoformat()]
            in_window = [r for r in s if r["snapshot_date"] > start.isoformat()]
            base = before[-1]["value"] if before else s[0]["value"]
            out[book] = {
                "value": s[-1]["value"], "drawdown": s[-1]["drawdown"],
                "return_this_cycle": round(s[-1]["value"] / base - 1, 6) if base else None,
                "max_drawdown_this_cycle": min((r["drawdown"] for r in in_window), default=None),
            }
        if "decision" in out and "shadow" in out:
            out["decision_minus_shadow_return_this_cycle"] = round(
                out["decision"]["return_this_cycle"] - out["shadow"]["return_this_cycle"], 6)
        return out

    def search_trades(self, filters: dict | None = None, limit: int = 20, order: str = "recent") -> dict:
        """Resolved trades matching ``filters``, newest first (or by worst / best score)."""
        if order not in ORDERS:
            raise ValueError(f"order must be one of {sorted(ORDERS)}")
        limit = max(1, min(int(limit), 100))
        rows = self._known(filters, order, limit + 1)
        return {"as_of": self.as_of.isoformat(), "count": min(len(rows), limit), "truncated": len(rows) > limit,
                "trades": [{k: r[k] for k in TRADE_COLUMNS} for r in rows[:limit]]}

    def read_trade(self, decision_id: str, max_chars: int = 1500) -> dict:
        """One resolved trade: decision, outcome, the Trader's proposal, conditions, reports and news."""
        rows = self._known({"decision_ids": [decision_id]})
        if not rows:
            # Same message whether the id is unknown, from another version, or not
            # resolved yet: the answer must not reveal anything about the future.
            raise LookupError(f"no resolved trade {decision_id!r} as of {self.as_of.isoformat()}")
        trade = rows[0]
        run_id = trade["run_id"]
        other_parts = [r for r in self._known() if r["run_id"] == run_id and r["decision_id"] != decision_id]
        proposal = _dicts(self.con.execute(
            "SELECT action, direction, entry_price, stop_loss, position_sizing, reasoning, structured_ok "
            "FROM trader_proposals WHERE run_id = ?", [run_id]))
        conditions = _dicts(self.con.execute("SELECT * EXCLUDE (run_id, computed_at) FROM latest_conditions WHERE run_id = ?", [run_id]),
                            json_cols=("input_status",))
        reports = {r["agent"]: r["text"][:max_chars] + ("…" if len(r["text"]) > max_chars else "")
                   for r in _dicts(self.con.execute(
                       "SELECT agent, text FROM agent_outputs WHERE run_id = ? ORDER BY agent, round", [run_id]))}
        news = _dicts(self.con.execute(
            "SELECT DISTINCT ti.item_id, ti.created_at, ti.publisher, ti.title, ti.url FROM run_text_items rti "
            "JOIN text_items ti ON ti.item_id = rti.item_id JOIN runs r ON r.run_id = rti.run_id "
            "WHERE rti.run_id = ? AND ti.created_at < timezone('UTC', CAST(r.trade_date + 1 AS TIMESTAMP)) "
            "ORDER BY ti.created_at DESC LIMIT 20", [run_id]))
        sources = _dicts(self.con.execute(
            "SELECT method, status, any_value(vendor) AS vendor, count(*) AS calls FROM evidence_calls "
            "WHERE run_id = ? GROUP BY method, status ORDER BY method", [run_id]))
        return {
            "as_of": self.as_of.isoformat(),
            "trade": {k: trade[k] for k in (*TRADE_COLUMNS, "explanation", "memo_id")},
            "other_parts_of_same_proposal": [{k: r[k] for k in TRADE_COLUMNS} for r in other_parts],
            "trader_proposal": proposal[0] if proposal else None,
            "conditions": conditions[0] if conditions else None,
            "reports": reports,
            "news_seen": news,
            "data_sources": sources,
        }

    def subgroup_stats(self, filters: dict | None = None, split_by: str | None = None) -> dict:
        """The statistics for any group of resolved trades, optionally split by one field."""
        if split_by is not None and split_by not in SPLITS:
            raise ValueError(f"split_by must be one of {sorted(SPLITS)}")
        rows = self._known(filters)
        out = {"as_of": self.as_of.isoformat(), "filters": filters or {}, "overall": group_stats(rows)}
        if split_by:
            groups: dict = {}
            for r in rows:
                key = r["trade_date"][:7] if split_by == "month" else r[split_by]
                groups.setdefault(str(key), []).append(r)
            out["split_by"] = split_by
            out["groups"] = {k: group_stats(v) for k, v in sorted(groups.items())}
        return out

    def past_memos(self, limit: int = 3) -> dict:
        """The latest memos before ``as_of`` with their advice, and every hypothesis's status as of now."""
        limit = max(1, min(int(limit), 12))
        memos = _dicts(self.con.execute(
            "SELECT memo_id, cycle_date, risk_level, previous_risk_level, memo_text, safety_passed, safety_flags "
            "FROM memos WHERE experiment_version = ? AND cycle_date < ? ORDER BY cycle_date DESC LIMIT ?",
            [self.version, self.as_of, limit]), json_cols=("safety_flags",))
        for m in memos:
            m["advice"] = _dicts(self.con.execute(
                "SELECT category, advice, confidence, reasoning, cited_ids FROM memo_advice WHERE memo_id = ? "
                "ORDER BY category", [m["memo_id"]]))
        hypotheses = _dicts(self.con.execute(
            """
            SELECT h.hypothesis_id, h.memo_id AS proposed_in, pm.cycle_date AS proposed_on, h.category,
                   h.statement, h.subgroup, coalesce(c.status, 'pending') AS status,
                   c.memo_id AS last_checked_in, c.evidence AS last_check_evidence
            FROM memo_hypotheses h
            JOIN memos pm ON pm.memo_id = h.memo_id
            LEFT JOIN (
              SELECT hc.*, m.cycle_date FROM hypothesis_checks hc JOIN memos m ON m.memo_id = hc.memo_id
              JOIN memo_hypotheses h2 ON h2.hypothesis_id = hc.hypothesis_id
              JOIN memos pm2 ON pm2.memo_id = h2.memo_id
              -- a check counts only from the same version, a later cycle, and before as_of
              WHERE m.cycle_date < ? AND m.experiment_version = pm2.experiment_version
                AND m.cycle_date > pm2.cycle_date
              QUALIFY row_number() OVER (PARTITION BY hc.hypothesis_id ORDER BY m.cycle_date DESC) = 1
            ) c ON c.hypothesis_id = h.hypothesis_id
            WHERE pm.experiment_version = ? AND pm.cycle_date < ?
            ORDER BY pm.cycle_date DESC, h.hypothesis_id
            """, [self.as_of, self.version, self.as_of]), json_cols=("subgroup", "last_check_evidence"))
        return {"as_of": self.as_of.isoformat(), "memos": memos, "hypotheses": hypotheses}


# Function forms, for callers that prefer them to the class.
def cycle_summary(con, as_of, experiment_version, **kw):
    return CalibrationTools(con, as_of, experiment_version).cycle_summary(**kw)


def search_trades(con, as_of, experiment_version, filters=None, **kw):
    return CalibrationTools(con, as_of, experiment_version).search_trades(filters, **kw)


def read_trade(con, as_of, experiment_version, decision_id, **kw):
    return CalibrationTools(con, as_of, experiment_version).read_trade(decision_id, **kw)


def subgroup_stats(con, as_of, experiment_version, filters=None, **kw):
    return CalibrationTools(con, as_of, experiment_version).subgroup_stats(filters, **kw)


def past_memos(con, as_of, experiment_version, **kw):
    return CalibrationTools(con, as_of, experiment_version).past_memos(**kw)
