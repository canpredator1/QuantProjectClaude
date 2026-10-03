"""Compute every signal once and store what the later stages need.

Outputs (in data/cache):
  daily_ic.parquet, daily_ls.parquet, daily_to.parquet  dates x signals
  daily_ic_placebo.parquet   IC against a shuffled target (the null)
  features.npy               rows x signals, centred cross-sectional ranks
  rows.parquet               date, ticker, target, target rank per row

A "row" is one stock on one day: an index member at the close of t with a
known open(t+1) -> open(t+2) return.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import CFG, Config
from .data import Panels, build_panels, eval_dates
from .evaluate import daily_signal_stats, rowwise_corr
from .signals import REGISTRY, Context


def shuffled_target(target: pd.DataFrame, valid: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = target.to_numpy().copy()
    v = valid.to_numpy()
    for i in range(len(t)):
        idx = np.flatnonzero(v[i])
        t[i, idx] = t[i, rng.permutation(idx)]
    return pd.DataFrame(t, index=target.index, columns=target.columns).where(valid)


def run_factory(cfg: Config = CFG, panels: Panels | None = None, limit: int | None = None,
                verbose: bool = True) -> None:
    panels = panels or build_panels(cfg)
    dates = eval_dates(panels, cfg)
    uni = panels.universe.loc[dates]
    tgt = panels.target.loc[dates]
    rows_mask = uni & tgt.notna()
    di, ti = np.nonzero(rows_mask.to_numpy())

    y_rank = tgt.where(rows_mask).rank(axis=1, pct=True) - 0.5
    rows = pd.DataFrame({
        "date": dates[di],
        "ticker": panels.close.columns[ti],
        "target": tgt.to_numpy()[di, ti],
        "y_rank": y_rank.to_numpy()[di, ti],
    })
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(cfg.cache_dir / "rows.parquet")

    signals = REGISTRY[:limit] if limit else REGISTRY
    feats = np.lib.format.open_memmap(cfg.cache_dir / "features.npy", mode="w+",
                                      dtype=np.float32, shape=(len(rows), len(signals)))
    tshuf = shuffled_target(tgt, rows_mask)

    ctx = Context(panels)
    ic, ls, to, icp = {}, {}, {}, {}
    t0 = time.time()
    for k, sig in enumerate(signals):
        t1 = time.time()
        full = sig.fn(ctx).replace([np.inf, -np.inf], np.nan)
        s = full.loc[dates]
        d = daily_signal_stats(s, tgt, uni, cfg.quantile, cfg.min_stocks_per_day)
        ic[sig.name], ls[sig.name], to[sig.name] = d["ic"], d["ls"], d["to"]

        valid = uni & s.notna() & tgt.notna()
        p = rowwise_corr(s.where(valid).rank(axis=1), tshuf.where(valid).rank(axis=1))
        p[valid.sum(axis=1) < cfg.min_stocks_per_day] = np.nan
        icp[sig.name] = p

        ranks = s.where(uni).rank(axis=1, pct=True) - 0.5
        feats[:, k] = ranks.to_numpy()[di, ti].astype(np.float32)
        if verbose:
            print(f"[{k + 1:3d}/{len(signals)}] {sig.name:<40s} IC={d['ic'].mean():+.4f} "
                  f"({time.time() - t1:4.1f}s, total {time.time() - t0:5.0f}s)", flush=True)
    feats.flush()

    for name, data in [("daily_ic", ic), ("daily_ls", ls), ("daily_to", to),
                       ("daily_ic_placebo", icp)]:
        pd.DataFrame(data).to_parquet(cfg.cache_dir / f"{name}.parquet")
    pd.DataFrame([(s.name, s.family) for s in signals], columns=["signal", "family"]) \
        .to_parquet(cfg.cache_dir / "signal_meta.parquet")
