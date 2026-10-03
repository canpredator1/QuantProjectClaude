import numpy as np
import pandas as pd

from conftest import make_panels
from alphafactory.backtest import banded_weights, run_portfolio
from alphafactory.combine import dedup, split_dates
from alphafactory.evaluate import (daily_signal_stats, newey_west_t,
                                   quantile_weights, turnover)


def test_target_is_next_open_to_following_open(prices):
    p = make_panels(prices)
    o = prices["open"]
    t = 50
    expected = o.iloc[t + 2] / o.iloc[t + 1] - 1
    assert np.allclose(p.target.iloc[t], expected)
    assert p.target.iloc[-2:].isna().all().all()


def test_quantile_weights_are_dollar_neutral():
    rng = np.random.default_rng(0)
    s = pd.DataFrame(rng.normal(size=(5, 50)))
    s.iloc[2, :10] = np.nan
    w = quantile_weights(s, 0.1)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.clip(lower=0).sum(axis=1), 1)
    assert (w[s.isna()].fillna(0) == 0).all().all()


def test_turnover_counts_full_rebalance():
    w = pd.DataFrame([[0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, -0.5, 0.5]])
    to = turnover(w)
    assert np.allclose(to.to_numpy(), [1.0, 0.0, 1.0])


def test_perfect_signal_has_ic_one(prices):
    p = make_panels(prices)
    d = daily_signal_stats(p.target, p.target, p.universe, 0.2, 5)
    assert np.allclose(d["ic"].dropna(), 1.0)
    assert (d["ls"].dropna() > 0).all()


def test_newey_west_matches_plain_t_for_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1.0, 20000)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert abs(newey_west_t(x, 10) - plain) / plain < 0.05


def test_dedup_drops_duplicates():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    X = np.c_[a, a + 0.01 * rng.normal(size=1000), b]
    kept = dedup(["s0", "s1", "s2"], X, ["s0", "s1", "s2"], 0.8)
    assert kept == ["s0", "s2"]


def test_split_dates_embargo_and_no_overlap():
    dates = pd.bdate_range("2014-01-01", "2016-12-31")
    train, test = split_dates(dates, 2016, embargo=5)
    assert train.max() < test.min()
    gap = dates.get_loc(test.min()) - dates.get_loc(train.max())
    assert gap == 6
    assert (test.year == 2016).all()


def test_banded_weights_reduce_turnover():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 100))
    s = pd.DataFrame(base + 0.3 * rng.normal(size=(250, 100)))
    plain = turnover(quantile_weights(s, 0.1)).mean()
    banded = turnover(banded_weights(s, 0.1, 0.2)).mean()
    assert banded < plain


def test_costs_are_charged(prices):
    p = make_panels(prices)
    score = prices["close"].pct_change(fill_method=None).where(p.universe)
    res = run_portfolio(score, p.target, 0.2, cost_bps=10)
    diff = (res["ls_gross"] - res["ls_net"])
    assert np.allclose(diff, res["ls_to"] * 10 / 1e4)
