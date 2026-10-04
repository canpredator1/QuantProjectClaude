"""What if we simply bought the most volatile S&P 500 stocks every week?

Each Friday close: rank members by trailing volatility. Buy the most volatile
N at Monday's open (first trading day of the next week), equal weight, sell at
the following week's open. Costs: 5 bps per unit of turnover.

Usage: python scripts/volatile_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphafactory.config import CFG  # noqa: E402
from alphafactory.data import build_panels  # noqa: E402

COST = CFG.cost_bps / 1e4
OUT = CFG.report_dir / "winners"


def stats(r: pd.Series, per_year: int = 52) -> dict:
    r = r.dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / per_year
    return {
        "avg_week": r.mean(),
        "median_week": r.median(),
        "cagr": wealth.iloc[-1] ** (1 / years) - 1,
        "vol": r.std() * np.sqrt(per_year),
        "sharpe": r.mean() / r.std() * np.sqrt(per_year),
        "max_dd": (wealth / wealth.cummax() - 1).min(),
        "worst_week": r.min(),
        "best_week": r.max(),
        "up_weeks": (r > 0).mean(),
        "x_money": wealth.iloc[-1],
    }


def main():
    p = build_panels(CFG)
    o, c, uni = p.open, p.close, p.universe
    ret = c.pct_change(fill_method=None)
    vol = ret.rolling(63, min_periods=50).std()

    d = c.index[c.index >= pd.Timestamp(CFG.eval_start)]
    week = d.to_period("W-FRI")
    last_day = pd.Series(d, index=d).groupby(week).max()          # signal day (Friday close)
    first_day = pd.Series(d, index=d).groupby(week).min()         # entry day (Monday open)
    sig_days = last_day.iloc[:-2].to_numpy()
    entry = first_day.iloc[1:-1].to_numpy()
    exit_ = first_day.iloc[2:].to_numpy()

    spy = pd.read_parquet(CFG.raw_dir / "prices.parquet", filters=[("ticker", "==", "SPY")]) \
        .set_index("date").sort_index()
    spy_open = spy.open * spy.adj_close / spy.close

    picks = {"Top 10 most volatile": 10, "Top 25": 25, "Top 50 (~10%)": 50}
    res = {k: [] for k in picks}
    res["Least volatile 50"] = []
    res["All members (equal weight)"] = []
    res["SPY"] = []
    prev = {k: pd.Series(dtype=float) for k in res}
    idx = []
    for s, e, x in zip(sig_days, entry, exit_):
        members = uni.loc[s] & vol.loc[s].notna()
        v = vol.loc[s][members]
        r = (o.loc[x] / o.loc[e] - 1).reindex(v.index)
        ok = r.notna()
        v, r = v[ok], r[ok]
        books = {k: v.nlargest(n).index for k, n in picks.items()}
        books["Least volatile 50"] = v.nsmallest(50).index
        books["All members (equal weight)"] = v.index
        for k, names in books.items():
            w = pd.Series(1.0 / len(names), index=names)
            to = w.sub(prev[k], fill_value=0).abs().sum()
            res[k].append(r[names].mean() - to * COST)
            prev[k] = w
        res["SPY"].append(spy_open.get(x, np.nan) / spy_open.get(e, np.nan) - 1)
        idx.append(pd.Timestamp(e))

    df = pd.DataFrame(res, index=idx)
    df.to_csv(OUT / "volatile_weekly_returns.csv")
    periods = {"2006–2026 (all)": (2006, 2026), "2006–2015": (2006, 2015),
               "2016–2026": (2016, 2026), "2020–2026": (2020, 2026)}
    lines = ["# Buying the most volatile stocks every week\n",
             "Rank S&P 500 members by 63-day volatility at Friday's close, buy at the next "
             "open, hold one week, equal weight, 5 bps per trade.\n"]
    for pname, (a, b) in periods.items():
        sub = df[(df.index.year >= a) & (df.index.year <= b)]
        t = pd.DataFrame({k: stats(sub[k]) for k in df.columns}).T
        lines.append(f"\n## {pname}\n")
        lines.append("| Portfolio | Avg week | Median week | Per year (CAGR) | Volatility | Sharpe | "
                     "Max drawdown | Worst week | Best week | $1 becomes |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, r in t.iterrows():
            lines.append(f"| {k} | {r.avg_week * 100:+.2f}% | {r.median_week * 100:+.2f}% | "
                         f"{r.cagr * 100:+.1f}% | {r.vol * 100:.0f}% | {r.sharpe:.2f} | "
                         f"{r.max_dd * 100:.0f}% | {r.worst_week * 100:+.0f}% | "
                         f"{r.best_week * 100:+.0f}% | ${r.x_money:.2f} |")
    yearly = df.groupby(df.index.year).apply(lambda g: (1 + g).prod() - 1)
    lines.append("\n## Year by year\n")
    lines.append("| Year | " + " | ".join(df.columns) + " |")
    lines.append("|---|" + "---|" * len(df.columns))
    for y, r in yearly.iterrows():
        lines.append(f"| {y} | " + " | ".join(f"{v * 100:+.0f}%" for v in r) + " |")
    (OUT / "VOLATILE.md").write_text("\n".join(lines) + "\n")
    print((OUT / "VOLATILE.md").read_text())


if __name__ == "__main__":
    main()
