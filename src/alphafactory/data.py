"""Load the S&P 500 price history and turn it into aligned wide panels.

Every panel is a DataFrame indexed by trading date with one column per ticker.
Prices are fully adjusted (splits + dividends) so returns computed from any of
open/high/low/close are comparable through time.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG, Config


@dataclass
class Panels:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    dollar_volume: pd.DataFrame
    universe: pd.DataFrame        # bool: tradable S&P 500 member at close of t
    target: pd.DataFrame          # open(t+1) -> open(t+2) return
    market: pd.Series             # equal-weight universe close-to-close return

    @property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change(fill_method=None)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(cfg: Config = CFG) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(cfg.raw_dir / "prices.parquet")
    members = pd.read_parquet(cfg.raw_dir / "membership_intervals.parquet")
    recycled = pd.read_csv(cfg.raw_dir / "recycled_tickers.csv")
    # Recycled symbols point at a different company after the index member
    # left; drop them entirely rather than risk mixing two companies.
    members = members[~members.ticker.isin(recycled.ticker)]
    prices = prices[prices.ticker.isin(set(members.ticker))]
    prices = prices[prices.date >= pd.Timestamp(cfg.load_start)]
    return prices, members


def membership_mask(members: pd.DataFrame, dates: pd.DatetimeIndex,
                    tickers: pd.Index) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    for row in members.itertuples(index=False):
        if row.ticker not in mask.columns:
            continue
        end = row.end if pd.notna(row.end) else dates[-1] + pd.Timedelta(days=1)
        sel = (dates >= row.start) & (dates < end)
        mask.loc[sel, row.ticker] = True
    return mask


def build_panels(cfg: Config = CFG) -> Panels:
    cache = cfg.cache_dir / "panels.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    prices, members = load_raw(cfg)
    prices = prices.sort_values(["ticker", "date"])
    factor = prices.adj_close / prices.close
    prices = prices.assign(
        a_open=prices.open * factor,
        a_high=prices.high * factor,
        a_low=prices.low * factor,
        a_close=prices.adj_close,
        # close and volume share the same split basis in every source, so
        # their product is the real traded dollar value.
        dvol=(prices.close * prices.volume).where(prices.volume > 0),
    )

    def wide(col):
        return prices.pivot(index="date", columns="ticker", values=col).astype("float64")

    # Drop non-trading dates that appear only because one stray row exists
    # (e.g. Good Friday 2017-04-14); they would break shift-based targets.
    per_day = prices.groupby("date").size()
    keep = per_day.index[per_day >= 0.5 * per_day.rolling(21, min_periods=1, center=True).median()]
    prices = prices[prices.date.isin(keep)]

    close = wide("a_close")
    dates, tickers = close.index, close.columns
    open_, high, low = wide("a_open"), wide("a_high"), wide("a_low")
    # Repair rare inconsistent bars so high/low always bracket open/close.
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    dvol = wide("dvol")

    member = membership_mask(members, dates, tickers)
    history = close.notna().cumsum()
    raw_close = wide("close")
    universe = (
        member
        & close.notna()
        & open_.notna()
        & (history >= cfg.min_history_days)
        & (raw_close >= cfg.min_price)
    )

    # Decision at the close of t, enter at the open of t+1, exit at the open
    # of t+2. shift(-k) moves along trading dates, so a missing future bar
    # gives NaN (the stock simply has no target that day).
    target = open_.shift(-2) / open_.shift(-1) - 1.0
    target = target.clip(-cfg.target_clip, cfg.target_clip)

    rets = close.pct_change(fill_method=None)
    market = rets.where(universe.shift(1, fill_value=False)).mean(axis=1)

    panels = Panels(open_, high, low, close, dvol, universe, target, market)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panels, cache)
    return panels


def eval_dates(panels: Panels, cfg: Config = CFG) -> pd.DatetimeIndex:
    d = panels.close.index
    d = d[d >= pd.Timestamp(cfg.eval_start)]
    # The final two dates have no complete target.
    return d[:-2]
