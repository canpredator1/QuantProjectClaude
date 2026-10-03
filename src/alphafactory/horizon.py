"""Longer-horizon labels, built from the cached feature store.

Round 1 showed the 1-day edge is real before costs but too short-lived to pay
for daily turnover. Here the same signals are scored against the return over
the next `h` days (open(t+1) -> open(t+1+h)), so a model can be trained on an
edge that lasts long enough to hold positions for several days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .combine import Store
from .config import CFG, Config
from .data import Panels


def horizon_labels(store: Store, panels: Panels, h: int, cfg: Config = CFG) -> pd.DataFrame:
    path = cfg.cache_dir / f"labels_h{h}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    o = panels.open
    fwd = (o.shift(-(h + 1)) / o.shift(-1) - 1.0).clip(-0.75, 0.75)
    di = o.index.get_indexer(store.rows.date)
    ti = o.columns.get_indexer(store.rows.ticker)
    lab = pd.DataFrame({"date": store.rows.date, "target_h": fwd.to_numpy()[di, ti]})
    lab["y_rank_h"] = lab.groupby("date").target_h.rank(pct=True) - 0.5
    lab.to_parquet(path)
    return lab


def daily_ic_from_features(store: Store, y: np.ndarray, cfg: Config = CFG,
                           name: str = "h") -> pd.DataFrame:
    """Per-day correlation between each signal's cross-sectional rank and the
    label's rank, computed with grouped sums (fast, no re-ranking)."""
    path = cfg.cache_dir / f"daily_ic_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    codes, dates = pd.factorize(store.rows.date, sort=True)
    D = len(dates)
    X = np.asarray(store.X)
    out = {}
    for k, sig in enumerate(store.names):
        x = X[:, k].astype("float64")
        m = ~np.isnan(x) & ~np.isnan(y)
        c, xv, yv = codes[m], x[m], y[m]
        n = np.bincount(c, minlength=D).astype(float)
        sx, sy = np.bincount(c, xv, D), np.bincount(c, yv, D)
        sxx, syy, sxy = np.bincount(c, xv * xv, D), np.bincount(c, yv * yv, D), np.bincount(c, xv * yv, D)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sx * sy / n
            den = np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
            ic = cov / den
        ic[n < cfg.min_stocks_per_day] = np.nan
        out[sig] = ic
    df = pd.DataFrame(out, index=pd.DatetimeIndex(dates))
    df.to_parquet(path)
    return df
