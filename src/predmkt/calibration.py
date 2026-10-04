"""Is the market price an honest probability? (the favourite-longshot test)

Every binary market gives two contracts: YES at p and NO at 1 - p. Buying a
contract at price q pays 1 if it wins, so the return per $1 is
win / (q + fee) - 1. If prices were fair, every price bucket would return
about minus the fee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PM, PMConfig, fee_per_share

BUCKETS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0]


def contracts(snap: pd.DataFrame, cfg: PMConfig = PM) -> pd.DataFrame:
    yes = snap.assign(side="YES", price=snap.yes_price, won=snap.yes_won)
    no = snap.assign(side="NO", price=1 - snap.yes_price, won=1 - snap.yes_won)
    c = pd.concat([yes, no], ignore_index=True)
    c["fee"] = [fee_per_share(p, cat, cfg) for p, cat in zip(c.price, c.category)]
    c["ret"] = c.won / (c.price + c.fee) - 1.0
    c["bucket"] = pd.cut(c.price, BUCKETS, include_lowest=True)
    return c


def calibration_table(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby("bucket", observed=True)
    t = pd.DataFrame({
        "contracts": g.size(),
        "avg_price": g.price.mean(),
        "win_rate": g.won.mean(),
        "return_per_$": g.ret.mean(),
    })
    # standard error of the mean return, for an honest read of noise
    t["se"] = g.ret.std() / np.sqrt(t.contracts)
    t["gap"] = t.win_rate - t.avg_price
    return t


def strategy_test(snap: pd.DataFrame, lo: float, hi: float, days_before: int,
                  split_frac: float = 0.5, cfg: PMConfig = PM) -> dict:
    """Buy every contract priced in [lo, hi] `days_before` days out.
    Markets are split by end date: the first part is where a rule would be
    chosen, the second part is the out-of-sample check."""
    s = snap[snap.days_before == days_before].sort_values("end")
    cut = s.end.iloc[int(len(s) * split_frac)] if len(s) else None
    out = {}
    for name, part in [("first half", s[s.end < cut]), ("second half", s[s.end >= cut])]:
        c = contracts(part, cfg)
        pick = c[(c.price >= lo) & (c.price <= hi)]
        out[name] = {
            "bets": len(pick),
            "win_rate": pick.won.mean() if len(pick) else np.nan,
            "return_per_$": pick.ret.mean() if len(pick) else np.nan,
            "se": pick.ret.std() / np.sqrt(len(pick)) if len(pick) > 1 else np.nan,
            # money is locked until resolution: express per day held
            "return_per_$_per_day": pick.ret.mean() / days_before if len(pick) else np.nan,
        }
    return out
