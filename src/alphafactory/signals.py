"""The signal library: a few hundred formulaic trading-signal ideas.

Every signal is a function of prices and volume known at the close of day t.
Nothing here looks at the future; tests/test_no_lookahead.py enforces that by
corrupting future data and checking that past signal values do not change.

Signals are generated from families with parameter grids. The sign of a
signal does not matter: the evaluation learns whether high values predict
high or low returns. What matters is that every idea tried is counted, so the
multiple-testing hurdle reflects the real number of attempts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .data import Panels


@dataclass(frozen=True)
class Signal:
    name: str
    family: str
    fn: Callable[["Context"], pd.DataFrame]


# --------------------------------------------------------------------------
# Time-series and cross-sectional operators
# --------------------------------------------------------------------------

def _mp(w: int) -> int:
    """Minimum observations for a rolling window: 80% of it."""
    return min(w, max(1, int(np.ceil(0.8 * w))))


def ts_mean(x, w):
    return x.rolling(w, min_periods=_mp(w)).mean()


def ts_sum(x, w):
    return x.rolling(w, min_periods=_mp(w)).sum()


def ts_std(x, w):
    return x.rolling(w, min_periods=_mp(w)).std()


def ts_max(x, w):
    return x.rolling(w, min_periods=_mp(w)).max()


def ts_min(x, w):
    return x.rolling(w, min_periods=_mp(w)).min()


def ts_rank(x, w):
    return x.rolling(w, min_periods=_mp(w)).rank(pct=True)


def ts_corr(x, y, w):
    both = x.notna() & y.notna()
    x, y = x.where(both), y.where(both)
    mx, my = ts_mean(x, w), ts_mean(y, w)
    cov = ts_mean(x * y, w) - mx * my
    vx = ts_mean(x * x, w) - mx * mx
    vy = ts_mean(y * y, w) - my * my
    den = np.sqrt(vx.clip(lower=0) * vy.clip(lower=0))
    return (cov / den).where(den > 1e-12).clip(-1, 1)


def ts_cov(x, y, w):
    both = x.notna() & y.notna()
    x, y = x.where(both), y.where(both)
    return ts_mean(x * y, w) - ts_mean(x, w) * ts_mean(y, w)


def ts_argmax_age(x: pd.DataFrame, w: int, use_min: bool = False) -> pd.DataFrame:
    """Days since the max (or min) of x within the trailing window."""
    a = x.to_numpy(dtype="float64")
    fill = np.inf if use_min else -np.inf
    a = np.where(np.isnan(a), fill, a)
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= w:
        for j in range(0, a.shape[1], 64):           # column chunks bound memory
            win = sliding_window_view(a[:, j:j + 64], w, axis=0)   # (T-w+1, n, w)
            idx = win.argmin(axis=2) if use_min else win.argmax(axis=2)
            out[w - 1:, j:j + 64] = w - 1 - idx
    out[np.isnan(x.to_numpy())] = np.nan
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def decay_linear(x: pd.DataFrame, w: int) -> pd.DataFrame:
    weights = np.arange(1, w + 1, dtype="float64")
    weights /= weights.sum()
    a = x.to_numpy(dtype="float64")
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= w:
        out[w - 1:] = sliding_window_view(a, w, axis=0) @ weights
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def delta(x, k):
    return x - x.shift(k)


# --------------------------------------------------------------------------
# Context: shared, lazily computed building blocks
# --------------------------------------------------------------------------

class Context:
    def __init__(self, panels: Panels):
        self.p = panels
        self._cache: dict[str, pd.DataFrame] = {}

    def get(self, key: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def rank(self, x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank among current universe members."""
        return x.where(self.p.universe).rank(axis=1, pct=True)

    # Basic series --------------------------------------------------------
    @property
    def c(self):
        return self.p.close

    @property
    def o(self):
        return self.p.open

    @property
    def h(self):
        return self.p.high

    @property
    def l(self):
        return self.p.low

    @property
    def dv(self):
        return self.p.dollar_volume

    @property
    def r(self):
        return self.get("r", lambda: self.c.pct_change(fill_method=None))

    @property
    def ldv(self):
        return self.get("ldv", lambda: np.log(self.dv))

    @property
    def m(self) -> pd.Series:
        return self.p.market

    @property
    def vwap(self):
        return self.get("vwap", lambda: (self.h + self.l + self.c) / 3.0)

    def beta(self, w):
        def f():
            m = pd.DataFrame(np.repeat(self.m.to_numpy()[:, None], self.r.shape[1], 1),
                             index=self.r.index, columns=self.r.columns)
            cov = ts_cov(self.r, m, w)
            var = ts_cov(m, m, w)
            return cov / var
        return self.get(f"beta{w}", f)

    def market_frame(self):
        def f():
            return pd.DataFrame(np.repeat(self.m.to_numpy()[:, None], self.r.shape[1], 1),
                                index=self.r.index, columns=self.r.columns)
        return self.get("mframe", f)

    def resid(self):
        """Market-residual daily return using yesterday's 63-day beta."""
        return self.get("resid", lambda: self.r - self.beta(63).shift(1) * self.market_frame())

    def ret(self, w, skip=0):
        return self.c.shift(skip) / self.c.shift(w) - 1.0

    def vol(self, w):
        return self.get(f"vol{w}", lambda: ts_std(self.r, w))

    def dv_shock(self, w):
        return self.ldv - ts_mean(self.ldv, w).shift(1)


# --------------------------------------------------------------------------
# Signal families
# --------------------------------------------------------------------------

REGISTRY: list[Signal] = []


def add(name: str, family: str, fn: Callable[[Context], pd.DataFrame]):
    REGISTRY.append(Signal(name, family, fn))


def _rsi(c: pd.DataFrame, w: int) -> pd.DataFrame:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / w, min_periods=w).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / w, min_periods=w).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _trend(ctx: Context, w: int, what: str) -> pd.DataFrame:
    y = np.log(ctx.c)
    t = pd.DataFrame(np.arange(len(y), dtype="float64")[:, None].repeat(y.shape[1], 1),
                     index=y.index, columns=y.columns).where(y.notna())
    cov = ts_cov(t, y, w)
    vt = ts_cov(t, t, w)
    vy = ts_cov(y, y, w)
    slope = cov / vt
    r2 = (cov * cov / (vt * vy)).clip(0, 1)
    if what == "r2":
        return r2
    n = w
    se = np.sqrt((1 - r2).clip(lower=1e-12) * vy / (vt * (n - 2)))
    return slope / se


def _streak(r: pd.DataFrame) -> pd.DataFrame:
    s = np.sign(r.to_numpy())
    out = np.zeros_like(s)
    for i in range(1, len(s)):
        prev = out[i - 1]
        cur = s[i]
        same = (np.sign(prev) == cur) & (cur != 0)
        out[i] = np.where(same, prev + cur, cur)
    out[np.isnan(s)] = np.nan
    return pd.DataFrame(out, index=r.index, columns=r.columns)


def _seasonal(ctx: Context, years: int, horizon: int) -> pd.DataFrame:
    """Average return over the next `horizon` days, as observed 1..years ago."""
    fwd = ctx.c.shift(-horizon) / ctx.c - 1.0
    total = pd.DataFrame(0.0, index=fwd.index, columns=fwd.columns)
    count = pd.DataFrame(0.0, index=fwd.index, columns=fwd.columns)
    for k in range(1, years + 1):
        lag = 252 * k - 1
        if lag < horizon:   # the window must end on or before today
            continue
        term = fwd.shift(lag)
        total += term.fillna(0.0)
        count += term.notna()
    return (total / count).where(count > 0)


def _register_all():
    # ---- Returns / momentum / reversal ---------------------------------
    for w in [1, 2, 3, 5, 10, 15, 21, 42, 63, 126, 189, 252]:
        add(f"ret_{w}", "momentum", lambda c, w=w: c.ret(w))
    for w, s in [(252, 21), (126, 21), (63, 21), (21, 5), (10, 1), (5, 1), (252, 5), (126, 5)]:
        add(f"ret_{w}_skip{s}", "momentum", lambda c, w=w, s=s: c.ret(w, s))
    for w in [1, 3, 5, 10, 21]:
        add(f"intraday_{w}", "momentum", lambda c, w=w: ts_mean(c.c / c.o - 1, w))
    for w in [1, 3, 5, 10, 21, 63]:
        add(f"overnight_{w}", "momentum", lambda c, w=w: ts_mean(c.o / c.c.shift(1) - 1, w))
    for w in [1, 3, 5, 10, 21, 63]:
        add(f"resret_{w}", "momentum", lambda c, w=w: ts_sum(c.resid(), w))
    for w in [1, 5, 21]:
        add(f"mktadj_{w}", "momentum", lambda c, w=w: ts_sum(c.r - c.market_frame(), w))
    for years, hz in [(1, 21), (3, 21), (5, 21), (5, 1), (10, 1)]:
        add(f"seasonal_{years}y_{hz}d", "seasonal", lambda c, y=years, h=hz: _seasonal(c, y, h))

    # ---- Volatility / risk ---------------------------------------------
    for w in [5, 10, 21, 63, 126, 252]:
        add(f"vol_{w}", "risk", lambda c, w=w: c.vol(w))
    for w in [21, 63]:
        add(f"downvol_{w}", "risk",
            lambda c, w=w: np.sqrt(ts_mean(c.r.clip(upper=0) ** 2, w)))
        add(f"updown_{w}", "risk",
            lambda c, w=w: np.sqrt(ts_mean(c.r.clip(lower=0) ** 2, w))
            / np.sqrt(ts_mean(c.r.clip(upper=0) ** 2, w)).replace(0, np.nan))
    for w in [1, 5, 21, 63]:
        add(f"parkinson_{w}", "risk", lambda c, w=w: np.sqrt(ts_mean(np.log(c.h / c.l) ** 2, w)))
    for a, b in [(5, 63), (21, 252), (10, 126)]:
        add(f"volchg_{a}_{b}", "risk", lambda c, a=a, b=b: c.vol(a) / c.vol(b))
    add("range_ratio_5_63", "risk",
        lambda c: ts_mean(np.log(c.h / c.l), 5) / ts_mean(np.log(c.h / c.l), 63))
    for w in [63, 126, 252]:
        add(f"beta_{w}", "risk", lambda c, w=w: c.beta(w))
    for w in [21, 63, 126]:
        add(f"idiovol_{w}", "risk", lambda c, w=w: ts_std(c.resid(), w))
    for w in [63, 252]:
        add(f"corrmkt_{w}", "risk", lambda c, w=w: ts_corr(c.r, c.market_frame(), w))
    for w in [5, 21, 63]:
        add(f"maxret_{w}", "risk", lambda c, w=w: ts_max(c.r, w))
        add(f"minret_{w}", "risk", lambda c, w=w: ts_min(c.r, w))
    for w in [21, 63, 126, 252]:
        add(f"skew_{w}", "risk", lambda c, w=w: c.r.rolling(w, min_periods=_mp(w)).skew())
    for w in [21, 63, 252]:
        add(f"kurt_{w}", "risk", lambda c, w=w: c.r.rolling(w, min_periods=_mp(w)).kurt())

    # ---- Price location / technical ------------------------------------
    for w in [5, 10, 21, 63, 126, 252]:
        add(f"dist_high_{w}", "technical", lambda c, w=w: c.c / ts_max(c.h, w) - 1)
        add(f"dist_low_{w}", "technical", lambda c, w=w: c.c / ts_min(c.l, w) - 1)
    for w in [3, 5, 10, 21, 50, 100, 200]:
        add(f"ma_ratio_{w}", "technical", lambda c, w=w: c.c / ts_mean(c.c, w) - 1)
    for s, l in [(5, 21), (10, 50), (21, 63), (50, 200), (20, 100)]:
        add(f"ma_cross_{s}_{l}", "technical", lambda c, s=s, l=l: ts_mean(c.c, s) / ts_mean(c.c, l) - 1)
    for w in [1, 3, 5, 10, 21]:
        add(f"clv_{w}", "technical",
            lambda c, w=w: ts_mean(((c.c - c.l) - (c.h - c.c)) / (c.h - c.l).replace(0, np.nan), w))
    for w in [5, 14, 21, 63]:
        add(f"stoch_{w}", "technical",
            lambda c, w=w: (c.c - ts_min(c.l, w)) / (ts_max(c.h, w) - ts_min(c.l, w)).replace(0, np.nan))
    for w in [2, 3, 5, 9, 14, 21]:
        add(f"rsi_{w}", "technical", lambda c, w=w: _rsi(c.c, w))
    for w in [10, 20, 50]:
        add(f"bollinger_{w}", "technical", lambda c, w=w: (c.c - ts_mean(c.c, w)) / ts_std(c.c, w))
    for w in [21, 63, 252]:
        add(f"days_since_high_{w}", "technical", lambda c, w=w: ts_argmax_age(c.h, w))
        add(f"days_since_low_{w}", "technical", lambda c, w=w: ts_argmax_age(c.l, w, use_min=True))
    for w in [10, 21, 63, 126, 252]:
        add(f"trend_t_{w}", "technical", lambda c, w=w: _trend(c, w, "t"))
    for w in [21, 63, 126]:
        add(f"trend_r2_{w}", "technical", lambda c, w=w: _trend(c, w, "r2"))
    add("streak", "technical", lambda c: _streak(c.r))
    for w in [5, 10, 21, 63]:
        add(f"up_frac_{w}", "technical", lambda c, w=w: ts_mean((c.r > 0).astype(float).where(c.r.notna()), w))

    # ---- Volume / liquidity --------------------------------------------
    for w in [5, 21, 63, 252]:
        add(f"log_dollar_vol_{w}", "liquidity", lambda c, w=w: np.log(ts_mean(c.dv, w)))
    for a, b in [(1, 21), (5, 21), (5, 63), (21, 126), (21, 252), (1, 5)]:
        add(f"dv_ratio_{a}_{b}", "liquidity", lambda c, a=a, b=b: ts_mean(c.dv, a) / ts_mean(c.dv, b))
    for w in [5, 21, 63, 252]:
        add(f"amihud_{w}", "liquidity", lambda c, w=w: np.log(ts_mean(c.r.abs() / c.dv, w) * 1e9 + 1e-6))
    for w in [21, 63]:
        add(f"dv_std_{w}", "liquidity", lambda c, w=w: ts_std(c.ldv, w))
    for w in [5, 10, 21, 63]:
        add(f"pv_corr_{w}", "liquidity", lambda c, w=w: ts_corr(c.c, c.dv, w))
    for w in [10, 21, 63]:
        add(f"rv_corr_{w}", "liquidity", lambda c, w=w: ts_corr(c.r, c.ldv.diff(), w))
    for w in [21, 63]:
        add(f"absr_dv_corr_{w}", "liquidity", lambda c, w=w: ts_corr(c.r.abs(), c.ldv, w))
    for w in [5, 10, 21]:
        add(f"vw_ret_{w}", "liquidity", lambda c, w=w: ts_sum(c.r * c.dv, w) / ts_sum(c.dv, w))
    for w in [5, 21, 63]:
        add(f"dv_shock_{w}", "liquidity", lambda c, w=w: c.dv_shock(w))

    # ---- WorldQuant-style formulaic alphas -----------------------------
    # Inspired by Kakushadze (2016), "101 Formulaic Alphas". Volume is dollar
    # volume and raw price differences are scaled by close so values compare
    # across stocks.
    def wq(n, f):
        add(f"wq_{n:03d}", "worldquant", f)

    R = lambda c, x: c.rank(x)  # noqa: E731
    wq(2, lambda c: -ts_corr(R(c, delta(c.ldv, 2)), R(c, (c.c - c.o) / c.o), 6))
    wq(3, lambda c: -ts_corr(R(c, c.o), R(c, c.dv), 10))
    wq(4, lambda c: -ts_rank(R(c, c.l), 9))
    wq(6, lambda c: -ts_corr(c.o, c.dv, 10))
    wq(12, lambda c: np.sign(delta(c.dv, 1)) * (-delta(c.c, 1) / c.c))
    wq(13, lambda c: -R(c, ts_cov(R(c, c.c), R(c, c.dv), 5)))
    wq(14, lambda c: -R(c, delta(c.r, 3)) * ts_corr(c.o, c.dv, 10))
    wq(15, lambda c: -ts_sum(R(c, ts_corr(R(c, c.h), R(c, c.dv), 3)), 3))
    wq(16, lambda c: -R(c, ts_cov(R(c, c.h), R(c, c.dv), 5)))
    wq(18, lambda c: -R(c, ts_std((c.c - c.o).abs() / c.c, 5) + (c.c - c.o) / c.c + ts_corr(c.c, c.o, 10)))
    wq(20, lambda c: -R(c, (c.o - c.h.shift(1)) / c.c) * R(c, (c.o - c.c.shift(1)) / c.c)
       * R(c, (c.o - c.l.shift(1)) / c.c))
    wq(22, lambda c: -delta(ts_corr(c.h, c.dv, 5), 5) * R(c, ts_std(c.r, 20)))
    wq(26, lambda c: -ts_max(ts_corr(ts_rank(c.dv, 5), ts_rank(c.h, 5), 5), 3))
    wq(33, lambda c: R(c, -(1 - c.o / c.c)))
    wq(34, lambda c: R(c, (1 - R(c, ts_std(c.r, 2) / ts_std(c.r, 5))) + (1 - R(c, delta(c.c, 1) / c.c))))
    wq(38, lambda c: -R(c, ts_rank(c.c, 10)) * R(c, c.c / c.o))
    wq(40, lambda c: -R(c, ts_std(c.h / c.c, 10)) * ts_corr(c.h, c.dv, 10))
    wq(41, lambda c: (np.sqrt(c.h * c.l) - c.vwap) / c.c)
    wq(42, lambda c: R(c, c.vwap - c.c) / R(c, c.vwap + c.c))
    wq(44, lambda c: -ts_corr(c.h, R(c, c.dv), 5))
    wq(53, lambda c: -delta(((c.c - c.l) - (c.h - c.c)) / (c.c - c.l).replace(0, np.nan), 9))
    wq(54, lambda c: -((c.l - c.c) * c.o ** 5) / ((c.l - c.h).replace(0, np.nan) * c.c ** 5))
    wq(55, lambda c: -ts_corr(R(c, (c.c - ts_min(c.l, 12)) / (ts_max(c.h, 12) - ts_min(c.l, 12)).replace(0, np.nan)),
                              R(c, c.dv), 6))
    wq(101, lambda c: (c.c - c.o) / ((c.h - c.l) + 0.001 * c.c))
    wq(9, lambda c: decay_linear(-delta(c.c, 1) / c.c, 5))
    wq(98, lambda c: R(c, decay_linear(ts_corr(c.vwap, ts_mean(c.dv, 5), 5), 7))
       - R(c, decay_linear(ts_rank(-ts_corr(R(c, c.o), R(c, ts_mean(c.dv, 15)), 21), 9), 7)))

    # ---- Interactions: product of centred cross-sectional ranks ----------
    base = {s.name: s for s in REGISTRY}
    pairs = [
        ("ret_1", "dv_shock_21"), ("ret_5", "dv_shock_21"), ("ret_1", "vol_21"),
        ("ret_5", "vol_21"), ("ret_21", "vol_63"), ("ret_252_skip21", "vol_63"),
        ("ret_1", "log_dollar_vol_21"), ("ret_5", "log_dollar_vol_21"),
        ("resret_1", "dv_shock_21"), ("resret_5", "dv_shock_21"),
        ("overnight_1", "dv_shock_5"), ("intraday_1", "dv_shock_5"),
        ("ret_21", "log_dollar_vol_63"), ("ret_252_skip21", "log_dollar_vol_63"),
        ("ret_5", "idiovol_63"), ("ret_1", "idiovol_21"), ("ret_21", "trend_r2_63"),
        ("dist_high_252", "vol_63"), ("ret_5", "beta_252"), ("ret_1", "beta_252"),
        ("clv_1", "dv_shock_5"), ("maxret_21", "log_dollar_vol_63"),
        ("ret_1", "amihud_21"), ("ret_5", "amihud_21"), ("overnight_5", "intraday_5"),
    ]
    for a, b in pairs:
        fa, fb = base[a].fn, base[b].fn
        add(f"x_{a}__{b}", "interaction",
            lambda c, fa=fa, fb=fb: (c.rank(fa(c)) - 0.5) * (c.rank(fb(c)) - 0.5))


_register_all()


def signal_names() -> list[str]:
    return [s.name for s in REGISTRY]
