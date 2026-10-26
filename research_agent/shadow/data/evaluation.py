"""End-of-project evaluation: SPY buy-and-hold vs original vs feedback.

Per setup: weekly returns (calendar weeks ending Friday) from daily book values,
Sharpe over the 3-month T-bill (^IRX) with a bootstrap CI, and Wilcoxon signed-rank tests on weekly return
differences. Per runner version: rejection precision, net value of rejections
and approval loss rate, on the stock-dates both versions decided (noise excluded).

Usage:
    python -m shadow.data.evaluation
    python -m shadow.data.evaluation --dev --json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shadow.data.risk_runner import VERSIONS
from shadow.data.writers import InvalidRecord, outcome_dates
from shadow.sim.config import DEFAULT, SimConfig

PriceProvider = Callable[[str, str], pd.DataFrame]  # (ticker, as_of) -> OHLCV rows up to as_of
SPY = "spy_buy_and_hold"
RISK_FREE = "^IRX"   # 13-week T-bill yield, percent a year
N_BOOT = 10_000
CI_LEVEL = 0.95
FLAT = 1e-12   # std at or below this counts as zero (float rounding on constant returns)


def book_values(con: duckdb.DuckDBPyConnection, version: str) -> pd.Series:
    """Daily decision-book value for one runner version, indexed by date."""
    rows = con.execute("""
        SELECT snapshot_date, value FROM book_snapshots
        WHERE book = 'decision' AND experiment_version = ?
        ORDER BY snapshot_date""", [version]).fetchall()
    return pd.Series({pd.Timestamp(d): v for d, v in rows}, dtype=float)


def spy_values(days: pd.DatetimeIndex, price_provider: PriceProvider, config: SimConfig = DEFAULT) -> pd.Series:
    """SPY buy-and-hold value: buy at the open of the first date, mark at each close."""
    df = price_provider("SPY", days[-1].date().isoformat())
    df = df.set_index(pd.to_datetime(df["Date"].astype(str).str[:10]))
    try:
        shares = config.starting_cash * (1 - config.transaction_cost) / float(df.loc[days[0], "Open"])
        return shares * df.loc[days, "Close"].astype(float)
    except KeyError as e:
        raise ValueError(f"missing SPY price: {e}") from e


def risk_free_weekly(days: pd.DatetimeIndex, price_provider: PriceProvider) -> float:
    """Weekly risk-free rate: mean 3-month T-bill yield (^IRX, percent) over ``days``."""
    df = price_provider(RISK_FREE, days[-1].date().isoformat())
    dates = pd.to_datetime(df["Date"].astype(str).str[:10])
    closes = df.loc[(dates >= days[0]) & (dates <= days[-1]), "Close"].astype(float).dropna()
    if closes.empty:
        raise ValueError(f"no {RISK_FREE} yields between {days[0].date()} and {days[-1].date()}")
    return (1 + closes.mean() / 100) ** (1 / 52) - 1


def weekly_returns(values: pd.Series, start: float) -> pd.Series:
    """Return per calendar week ending Friday; the first week is measured from ``start``."""
    weekly = values.groupby(values.index.to_period("W-FRI")).last()
    base = pd.Series([start], index=[weekly.index[0] - 1])
    return pd.concat([base, weekly]).pct_change().iloc[1:]


def sharpe(returns: np.ndarray) -> float | None:
    """Annualised Sharpe of weekly excess returns; None if under 2 weeks or flat."""
    if len(returns) < 2 or np.std(returns, ddof=1) <= FLAT:
        return None
    return float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(52))


def sharpe_ci(returns: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the Sharpe ratio; None if too few usable resamples."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns, size=(n_boot, len(returns)), replace=True)
    std = samples.std(axis=1, ddof=1)
    ok = std > FLAT
    if len(returns) < 2 or ok.sum() < n_boot / 2:
        return None
    stats = samples[ok].mean(axis=1) / std[ok] * math.sqrt(52)
    tail = (1 - CI_LEVEL) / 2 * 100
    return float(np.percentile(stats, tail)), float(np.percentile(stats, 100 - tail))


def wilcoxon_test(a: pd.Series, b: pd.Series) -> dict:
    """Wilcoxon signed-rank on weekly a - b over shared weeks; p is None if all differences are zero."""
    diff = (a - b).dropna()
    p = None if (diff == 0).all() else float(wilcoxon(diff).pvalue)
    return {"weeks": len(diff), "median_diff": float(diff.median()) if len(diff) else None, "p_value": p}


def decision_quality(con: duckdb.DuckDBPyConnection, version: str) -> dict:
    """Rejection precision, net rejection value and approval loss rate on stock-dates all versions decided."""
    placeholders = ", ".join("?" for _ in VERSIONS)
    rows = con.execute(f"""
        WITH shared AS (
          SELECT ticker, trade_date FROM decisions
          WHERE experiment_version IN ({placeholders})
          GROUP BY ticker, trade_date
          HAVING count(DISTINCT experiment_version) = {len(VERSIONS)})
        SELECT o.label, count(*), sum(-o.score * d.part_size)
        FROM decisions d
        JOIN shared s USING (ticker, trade_date)
        JOIN outcomes o USING (decision_id)
        WHERE d.experiment_version = ? AND o.label <> 'noise'
        GROUP BY o.label""", [*VERSIONS, version]).fetchall()
    n = {label: count for label, count, _ in rows}
    avoided = {label: value for label, _, value in rows}
    cr, mg = n.get("correct_rejection", 0), n.get("missed_gain", 0)
    al, ca = n.get("approval_loss", 0), n.get("correct_approval", 0)
    return {
        "rejections_scored": cr + mg,
        "rejection_precision": cr / (cr + mg) if cr + mg else None,
        "net_rejection_value": avoided.get("correct_rejection", 0.0) + avoided.get("missed_gain", 0.0),
        "approvals_scored": al + ca,
        "approval_loss_rate": al / (al + ca) if al + ca else None,
    }


def unscored_due(con: duckdb.DuckDBPyConnection, price_provider: PriceProvider, through: date) -> list[str]:
    """Decisions of the compared versions whose outcome was due by ``through`` but has no row."""
    df = price_provider("SPY", through.isoformat())
    calendar = sorted({date.fromisoformat(d) for d in df["Date"].astype(str).str[:10]})
    placeholders = ", ".join("?" for _ in VERSIONS)
    rows = con.execute(f"""
        SELECT d.decision_id, d.trade_date FROM decisions d
        LEFT JOIN outcomes o USING (decision_id)
        WHERE o.decision_id IS NULL AND d.experiment_version IN ({placeholders})
        ORDER BY d.trade_date, d.decision_id""", list(VERSIONS)).fetchall()
    due = []
    for decision_id, trade_date in rows:
        try:
            _, resolved = outcome_dates(trade_date, calendar)
        except InvalidRecord:
            continue   # no trading day after it in the price history: not due
        if resolved is not None and resolved <= through:
            due.append(decision_id)
    return due


def evaluate(con: duckdb.DuckDBPyConnection, price_provider: PriceProvider | None = None,
             n_boot: int = N_BOOT) -> dict:
    """Score every setup on the dates all runner versions have a decision-book snapshot."""
    values = pd.DataFrame({v: book_values(con, v) for v in VERSIONS}).dropna()
    if len(values) < 2:
        raise ValueError("need at least 2 shared snapshot dates across versions")
    if price_provider is None:
        from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
        price_provider = lambda t, as_of: load_ohlcv(t, as_of, fill_gaps=False)  # noqa: E731
    unscored = unscored_due(con, price_provider, values.index[-1].date())
    if unscored:
        raise ValueError(f"{len(unscored)} decision(s) are due but not scored (e.g. {unscored[:3]}); "
                         "run python -m shadow.data.outcome_judge first")
    values[SPY] = spy_values(values.index, price_provider)
    rf = risk_free_weekly(values.index, price_provider)

    start = DEFAULT.starting_cash
    weekly = {name: weekly_returns(values[name], start) for name in values.columns}
    setups = {}
    for name, r in weekly.items():
        dd = values[name] / values[name].cummax().clip(lower=start) - 1
        excess = r.to_numpy() - rf
        setups[name] = {"weeks": len(r), "total_return": float(values[name].iloc[-1] / start - 1),
                        "max_drawdown": float(dd.min()), "sharpe": sharpe(excess),
                        "sharpe_ci95": sharpe_ci(excess, n_boot)}
    return {
        "period": [values.index[0].date().isoformat(), values.index[-1].date().isoformat()],
        "risk_free_annual": float((1 + rf) ** 52 - 1),
        "setups": setups,
        "wilcoxon": {"feedback_vs_original": wilcoxon_test(weekly["feedback"], weekly["original"]),
                     "original_vs_spy": wilcoxon_test(weekly["original"], weekly[SPY]),
                     "feedback_vs_spy": wilcoxon_test(weekly["feedback"], weekly[SPY])},
        "decision_quality": {v: decision_quality(con, v) for v in VERSIONS},
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render(report: dict) -> str:
    """Plain-text version of ``evaluate``'s output."""
    lines = [f"evaluation period {report['period'][0]} to {report['period'][1]}",
             f"risk-free rate ({RISK_FREE} mean): {_pct(report['risk_free_annual'])} a year", "", "setups:"]
    for name, s in report["setups"].items():
        ci = s["sharpe_ci95"]
        ci_txt = "n/a" if ci is None else f"{ci[0]:.2f} to {ci[1]:.2f}"
        lines.append(f"  {name:17} sharpe={_num(s['sharpe']):6} 95% CI [{ci_txt}]  "
                     f"total_return={_pct(s['total_return'])}  max_drawdown={_pct(s['max_drawdown'])}  "
                     f"weeks={s['weeks']}")
    lines += ["", "wilcoxon signed-rank on weekly return differences:"]
    for name, w in report["wilcoxon"].items():
        p = "n/a" if w["p_value"] is None else f"{w['p_value']:.3f}"
        lines.append(f"  {name:21} p={p:6} median_diff={_pct(w['median_diff'])}  weeks={w['weeks']}")
    lines += ["", "decision quality (same stock-dates, noise excluded):"]
    for name, q in report["decision_quality"].items():
        lines.append(f"  {name:9} rejection_precision={_pct(q['rejection_precision'])} "
                     f"(n={q['rejections_scored']})  net_rejection_value={q['net_rejection_value']:+.3f}  "
                     f"approval_loss_rate={_pct(q['approval_loss_rate'])} (n={q['approvals_scored']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from shadow.data.db import check_schema_version, connect, resolve_db

    parser = argparse.ArgumentParser(description="Evaluate SPY buy-and-hold vs original vs feedback.")
    parser.add_argument("--db", help="DuckDB file (default warehouse/calibration.duckdb)")
    parser.add_argument("--dev", action="store_true", help="use warehouse/dev.duckdb")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON instead")
    args = parser.parse_args(argv)

    path = resolve_db(args.db, args.dev)
    con = connect(path, read_only=True)
    check_schema_version(con, path)
    report = evaluate(con)
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
