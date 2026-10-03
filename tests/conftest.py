import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphafactory.data import Panels  # noqa: E402


def make_panels(prices: dict) -> Panels:
    o, h, l, c, dv = (prices[k] for k in ["open", "high", "low", "close", "dv"])
    universe = pd.DataFrame(True, index=c.index, columns=c.columns)
    universe.iloc[:30] = False
    target = o.shift(-2) / o.shift(-1) - 1
    market = c.pct_change(fill_method=None).where(universe.shift(1, fill_value=False)).mean(axis=1)
    return Panels(o, h, l, c, dv, universe, target, market)


def synthetic_prices(n_days=420, n_tickers=25, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    r = rng.normal(0.0003, 0.02, (n_days, n_tickers))
    close = 50 * np.exp(np.cumsum(r, axis=0))
    gap = rng.normal(0, 0.005, (n_days, n_tickers))
    open_ = close * np.exp(gap) / np.exp(r * 0.5)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, (n_days, n_tickers))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, (n_days, n_tickers))))
    dv = np.exp(rng.normal(17, 0.5, (n_days, n_tickers)))
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
    # A few holes, like real data.
    close_df = f(close)
    close_df.iloc[100:105, 3] = np.nan
    return {"open": f(open_), "high": f(hi), "low": f(lo), "close": close_df, "dv": f(dv)}


@pytest.fixture
def prices():
    return synthetic_prices()
