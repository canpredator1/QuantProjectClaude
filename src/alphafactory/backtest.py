"""Turn daily scores into portfolios and charge realistic trading costs.

Timing: scores use data up to the close of day t. Positions are entered at
the open of t+1 and held to the open of t+2, so the daily P&L is
weight(t) * target(t), with target = open(t+1) -> open(t+2) return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import perf_table, quantile_weights, turnover


def to_wide(preds: pd.DataFrame, col: str) -> pd.DataFrame:
    return preds.pivot(index="date", columns="ticker", values=col)


def smooth(score: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponential smoothing of each stock's score over time (cuts turnover)."""
    if not halflife:
        return score
    sm = score.ewm(halflife=halflife, ignore_na=True).mean()
    return sm.where(score.notna())


def banded_weights(score: pd.DataFrame, enter_q: float, exit_q: float) -> pd.DataFrame:
    """Hysteresis: buy names entering the top enter_q, keep holding until they
    drop out of the top exit_q (same for shorts). Equal weight per side."""
    pct = score.rank(axis=1, pct=True).to_numpy()
    avail = ~np.isnan(pct)
    long_prev = np.zeros(pct.shape[1], bool)
    short_prev = np.zeros(pct.shape[1], bool)
    w = np.zeros(pct.shape)
    for i in range(len(pct)):
        p = np.nan_to_num(pct[i], nan=0.5)
        long_now = avail[i] & ((p > 1 - enter_q) | (long_prev & (p > 1 - exit_q)))
        short_now = avail[i] & ((p <= enter_q) | (short_prev & (p <= exit_q)))
        if long_now.any():
            w[i, long_now] = 1.0 / long_now.sum()
        if short_now.any():
            w[i, short_now] -= 1.0 / short_now.sum()
        long_prev, short_prev = long_now, short_now
    return pd.DataFrame(w, index=score.index, columns=score.columns)


def run_portfolio(score: pd.DataFrame, target: pd.DataFrame, q: float, cost_bps: float,
                  halflife: float = 0, exit_q: float | None = None) -> dict:
    s = smooth(score, halflife)
    if exit_q:
        w = banded_weights(s, q, exit_q)
    else:
        w = quantile_weights(s, q)
    t = target.reindex_like(w).fillna(0.0)

    ls_gross = (w * t).sum(axis=1)
    ls_to = turnover(w)
    w_long = w.clip(lower=0)
    long_gross = (w_long * t).sum(axis=1)
    long_to = turnover(w_long)
    bench = target.reindex_like(w).where(score.notna()).mean(axis=1)   # equal-weight universe
    c = cost_bps / 1e4
    return {
        "weights": w,
        "ls_gross": ls_gross,
        "ls_net": ls_gross - ls_to * c,
        "ls_to": ls_to,
        "long_gross": long_gross,
        "long_net": long_gross - long_to * c,
        "long_to": long_to,
        "bench_ew": bench,
    }


def period_mask(idx: pd.DatetimeIndex, start_year: int, end_year: int | None = None):
    m = idx.year >= start_year
    if end_year is not None:
        m &= idx.year <= end_year
    return m


def summarize(res: dict, start_year: int, end_year: int | None = None) -> dict:
    idx = res["ls_net"].index
    m = period_mask(idx, start_year, end_year)
    excess = (res["long_net"] - res["bench_ew"])[m]
    return {
        "long_short_net": perf_table(res["ls_net"][m], res["ls_to"]),
        "long_short_gross": perf_table(res["ls_gross"][m]),
        "long_only_net": perf_table(res["long_net"][m], res["long_to"]),
        "long_only_excess_vs_ew": perf_table(excess),
        "equal_weight_universe": perf_table(res["bench_ew"][m]),
    }
