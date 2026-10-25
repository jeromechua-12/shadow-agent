"""End-of-project evaluation: SPY buy-and-hold vs original vs feedback.

Per setup: weekly returns (calendar weeks ending Friday) from daily book values,
Sharpe with a bootstrap CI, and Wilcoxon signed-rank tests on weekly return
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

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from shadow.data.risk_runner import VERSIONS
from shadow.sim.config import DEFAULT, SimConfig

PriceProvider = Callable[[str, str], pd.DataFrame]  # (ticker, as_of) -> OHLCV rows up to as_of
SPY = "spy_buy_and_hold"
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
    """SPY buy-and-hold value: cash on day one, buy at the next open, mark at each close."""
    df = price_provider("SPY", days[-1].date().isoformat())
    df = df.set_index(pd.to_datetime(df["Date"].astype(str).str[:10]))
    try:
        shares = config.starting_cash * (1 - config.transaction_cost) / float(df.loc[days[1], "Open"])
        closes = df.loc[days[1:], "Close"].astype(float)
    except KeyError as e:
        raise ValueError(f"missing SPY price: {e}") from e
    return pd.concat([pd.Series({days[0]: config.starting_cash}), shares * closes])


def weekly_returns(values: pd.Series) -> pd.Series:
    """Return per calendar week ending Friday; the first week is measured from the first value."""
    weekly = values.groupby(values.index.to_period("W-FRI")).last()
    base = pd.Series([values.iloc[0]], index=[weekly.index[0] - 1])
    return pd.concat([base, weekly]).pct_change().iloc[1:]


def sharpe(returns: np.ndarray) -> float | None:
    """Annualised Sharpe of weekly returns (risk-free 0); None if under 2 weeks or flat."""
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


def evaluate(con: duckdb.DuckDBPyConnection, price_provider: PriceProvider | None = None,
             n_boot: int = N_BOOT) -> dict:
    """Score every setup on the dates all runner versions have a decision-book snapshot."""
    values = pd.DataFrame({v: book_values(con, v) for v in VERSIONS}).dropna()
    if len(values) < 2:
        raise ValueError("need at least 2 shared snapshot dates across versions")
    if price_provider is None:
        from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
        price_provider = lambda t, as_of: load_ohlcv(t, as_of, fill_gaps=False)  # noqa: E731
    values[SPY] = spy_values(values.index, price_provider)

    weekly = {name: weekly_returns(values[name]) for name in values.columns}
    setups = {}
    for name, r in weekly.items():
        dd = values[name] / values[name].cummax() - 1
        setups[name] = {"weeks": len(r), "total_return": float(values[name].iloc[-1] / values[name].iloc[0] - 1),
                        "max_drawdown": float(dd.min()), "sharpe": sharpe(r.to_numpy()),
                        "sharpe_ci95": sharpe_ci(r.to_numpy(), n_boot)}
    return {
        "period": [values.index[0].date().isoformat(), values.index[-1].date().isoformat()],
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
    lines = [f"evaluation period {report['period'][0]} to {report['period'][1]}", "", "setups:"]
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
