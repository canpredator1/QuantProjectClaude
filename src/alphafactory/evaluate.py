"""Scoring signals and strategies honestly.

All statistics are cross-sectional: each day we ask whether the signal ranks
tomorrow's winners above tomorrow's losers among the stocks actually in the
index that day. That removes the market's direction from the question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rowwise_corr(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return num / den.replace(0, np.nan)


def quantile_weights(score: pd.DataFrame, q: float) -> pd.DataFrame:
    """Equal-weight long top-q / short bottom-q, +1 / -1 gross per side."""
    pct = score.rank(axis=1, pct=True)
    top = (pct > 1 - q).astype(float)
    bot = (pct <= q).astype(float).where(score.notna(), 0.0)
    top = top.where(score.notna(), 0.0)
    w = top.div(top.sum(axis=1).replace(0, np.nan), axis=0) \
        - bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)


def turnover(w: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes; trading 100% of a $1 book once = 1."""
    to = w.diff().abs().sum(axis=1)
    to.iloc[0] = w.iloc[0].abs().sum()      # the initial purchase
    return to


def daily_signal_stats(sig: pd.DataFrame, target: pd.DataFrame, universe: pd.DataFrame,
                       q: float, min_n: int) -> pd.DataFrame:
    """Per-day rank IC, decile long-short return and turnover for one signal."""
    valid = universe & sig.notna() & target.notna()
    s = sig.where(valid)
    t = target.where(valid)
    n = valid.sum(axis=1)
    ic = rowwise_corr(s.rank(axis=1), t.rank(axis=1))
    w = quantile_weights(s, q)
    ls = (w * t.fillna(0.0)).sum(axis=1)
    to = turnover(w)
    out = pd.DataFrame({"ic": ic, "ls": ls, "to": to, "n": n})
    out.loc[n < min_n, ["ic", "ls"]] = np.nan
    return out


def newey_west_t(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean()
    e = x - mu
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = e[lag:] @ e[:-lag] / n
        var += 2 * (1 - lag / (lags + 1)) * gamma
    if var <= 0:
        return np.nan
    return mu / np.sqrt(var / n)


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(252))


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summarize_signal(daily: pd.DataFrame, cost_bps: float, nw_lags: int) -> dict:
    ic = daily["ic"].dropna()
    sign = np.sign(ic.mean()) if len(ic) else 0.0
    ls = daily["ls"] * sign
    net = ls - daily["to"] * cost_bps / 1e4
    return {
        "ic_mean": ic.mean(),
        "ic_t_nw": newey_west_t(ic.to_numpy(), nw_lags),
        "ic_hit": (np.sign(ic) == sign).mean() if len(ic) else np.nan,
        "ls_ann_gross": ls.mean() * 252,
        "sharpe_gross": sharpe(ls),
        "sharpe_net": sharpe(net),
        "turnover": daily["to"].mean(),
        "days": len(ic),
    }


def perf_table(r: pd.Series, to: pd.Series | None = None) -> dict:
    r = r.dropna()
    out = {
        "ann_return": r.mean() * 252,
        "ann_vol": r.std() * np.sqrt(252),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": (r > 0).mean(),
        "days": len(r),
    }
    if to is not None:
        out["turnover"] = to.reindex(r.index).mean()
    return out


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float]) -> float:
    """Bailey & Lopez de Prado (2014): probability the true Sharpe exceeds
    the best Sharpe expected from `len(trial_sharpes)` unskilled trials.
    Sharpes here are per-day (not annualised)."""
    r = r.dropna()
    t = len(r)
    sr = r.mean() / r.std()
    n = max(len(trial_sharpes), 2)
    var_sr = np.var(np.asarray(trial_sharpes) / np.sqrt(252), ddof=1) if len(trial_sharpes) > 1 else 0.0
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n)
                             + gamma * stats.norm.ppf(1 - 1 / (n * np.e)))
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
