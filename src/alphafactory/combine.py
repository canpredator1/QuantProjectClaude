"""Walk-forward signal selection and combination.

For each test year Y:
  1. Training window = every evaluation day before Y, minus an embargo.
  2. Keep signals whose daily rank-IC has a Newey-West |t| above the hurdle
     inside the window, then drop near-duplicates.
  3. Fit combiners on the window and score every stock-day in Y.
Nothing from year Y (or later) is used to make year Y's predictions.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CFG, Config
from .evaluate import newey_west_t


class Store:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        d = cfg.cache_dir
        self.rows = pd.read_parquet(d / "rows.parquet")
        self.X = np.load(d / "features.npy", mmap_mode="r")
        self.ic = pd.read_parquet(d / "daily_ic.parquet")
        self.ic_placebo = pd.read_parquet(d / "daily_ic_placebo.parquet")
        self.ls = pd.read_parquet(d / "daily_ls.parquet")
        self.to = pd.read_parquet(d / "daily_to.parquet")
        self.meta = pd.read_parquet(d / "signal_meta.parquet")
        self.names = list(self.meta.signal)
        self.dates = pd.DatetimeIndex(self.ic.index)
        self.row_date = self.rows.date.to_numpy()


def signal_tstats(ic: pd.DataFrame, nw_lags: int) -> pd.Series:
    return ic.apply(lambda col: newey_west_t(col.to_numpy(), nw_lags))


def dedup(cands: list[str], X_sample: np.ndarray, names: list[str], max_corr: float) -> list[str]:
    """Greedy: walk candidates from strongest to weakest, keep a signal only if
    its |correlation| with everything kept so far is below max_corr."""
    if not cands:
        return []
    idx = [names.index(c) for c in cands]
    m = np.nan_to_num(np.asarray(X_sample[:, idx], dtype="float64"))
    corr = np.corrcoef(m, rowvar=False)
    corr = np.atleast_2d(corr)
    kept: list[int] = []
    for j in range(len(idx)):
        if all(abs(corr[j, k]) < max_corr for k in kept):
            kept.append(j)
    return [cands[j] for j in kept]


def select(store: Store, train_dates: pd.DatetimeIndex, cfg: Config = CFG,
           ic: pd.DataFrame | None = None) -> pd.DataFrame:
    ic = store.ic if ic is None else ic
    t = signal_tstats(ic.loc[train_dates], cfg.nw_lags)
    mean_ic = ic.loc[train_dates].mean()
    cands = t[t.abs() > cfg.tstat_threshold].abs().sort_values(ascending=False).index.tolist()
    sample_days = train_dates[::10]
    sample_rows = np.flatnonzero(np.isin(store.row_date, sample_days.to_numpy()))
    kept = dedup(cands, store.X[sample_rows], store.names, cfg.max_corr)
    out = pd.DataFrame({"t": t, "ic": mean_ic, "sign": np.sign(mean_ic)})
    out["passed"] = out.index.isin(cands)
    out["kept"] = out.index.isin(kept)
    return out


def split_dates(dates: pd.DatetimeIndex, year: int, embargo: int):
    """Training days end `embargo` trading days before the first day of year."""
    first_test = dates[dates >= pd.Timestamp(f"{year}-01-01")][0]
    cut = dates.get_loc(first_test) - embargo
    return dates[:cut], dates[dates.year == year]


def _train_lgbm(X: np.ndarray, y: np.ndarray, cfg: Config) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    return lgb.train(cfg.lgbm_params, ds, num_boost_round=cfg.lgbm_rounds)


def walk_forward(store: Store, cfg: Config = CFG, last_year: int | None = None,
                 verbose: bool = True, ic: pd.DataFrame | None = None,
                 label: np.ndarray | None = None, embargo: int | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """Returns (predictions for every test row, per-year selection tables).

    `ic` and `label` default to the next-day target; pass a longer-horizon IC
    table and label (with a matching embargo) to train on that horizon.
    """
    label = store.rows.y_rank.to_numpy() if label is None else label
    embargo = cfg.embargo_days if embargo is None else embargo
    years = sorted({d.year for d in store.dates if d.year >= cfg.first_test_year})
    if last_year is not None:
        years = [y for y in years if y <= last_year]
    preds = []
    selections = {}
    for year in years:
        t0 = time.time()
        train_dates, test_dates = split_dates(store.dates, year, embargo)

        sel = select(store, train_dates, cfg, ic)
        selections[year] = sel
        kept = sel.index[sel.kept].tolist()
        kept_idx = [store.names.index(k) for k in kept]
        signs = sel.loc[kept, "sign"].to_numpy()

        train_days = train_dates[::cfg.train_day_stride].to_numpy()
        tr = np.flatnonzero(np.isin(store.row_date, train_days))
        tr = tr[~np.isnan(label[tr])]
        te = np.flatnonzero(np.isin(store.row_date, test_dates.to_numpy()))
        y_tr = label[tr]
        X_te_all = np.asarray(store.X[te], dtype=np.float32)

        out = store.rows.iloc[te][["date", "ticker", "target"]].copy()
        # 1) Equal-weight vote of surviving signals, each pointed the right way.
        with np.errstate(invalid="ignore"):
            out["simple"] = np.nanmean(X_te_all[:, kept_idx] * signs, axis=1)
        # 2) LightGBM on the surviving signals.
        X_tr_all = np.asarray(store.X[tr], dtype=np.float32)
        booster = _train_lgbm(X_tr_all[:, kept_idx], y_tr, cfg)
        out["lgbm_selected"] = booster.predict(X_te_all[:, kept_idx])
        # 3) LightGBM on every signal (lets the model decide; more overfit risk).
        booster_all = _train_lgbm(X_tr_all, y_tr, cfg)
        out["lgbm_all"] = booster_all.predict(X_te_all)
        del X_tr_all
        preds.append(out)
        if verbose:
            print(f"{year}: train days {len(train_dates)}, passed {int(sel.passed.sum())}, "
                  f"kept {len(kept)}, train rows {len(tr):,}, test rows {len(te):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(preds, ignore_index=True), selections
