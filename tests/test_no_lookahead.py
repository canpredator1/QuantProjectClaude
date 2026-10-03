"""Every signal must be computable from data available at the close of t.

We corrupt all data after a cut date and recompute. Any signal value on or
before the cut that changes would mean the signal peeks at the future.
"""
import numpy as np
import pytest

from conftest import make_panels, synthetic_prices
from alphafactory.signals import REGISTRY, Context

CUT = 300


@pytest.fixture(scope="module")
def two_worlds():
    base = synthetic_prices()
    rng = np.random.default_rng(1)
    corrupted = {}
    for k, df in base.items():
        df = df.copy()
        noise = np.exp(rng.normal(0, 0.3, df.iloc[CUT + 1:].shape))
        df.iloc[CUT + 1:] = df.iloc[CUT + 1:] * noise
        corrupted[k] = df
    # keep high/low consistent with open/close, identically in both worlds
    for d in (base, corrupted):
        d["high"] = np.maximum(d["high"], np.maximum(d["open"], d["close"]))
        d["low"] = np.minimum(d["low"], np.minimum(d["open"], d["close"]))
    return Context(make_panels(base)), Context(make_panels(corrupted))


@pytest.mark.parametrize("sig", REGISTRY, ids=lambda s: s.name)
def test_signal_uses_only_past_data(sig, two_worlds):
    a, b = two_worlds
    xa = sig.fn(a).iloc[: CUT + 1].to_numpy()
    xb = sig.fn(b).iloc[: CUT + 1].to_numpy()
    assert np.allclose(xa, xb, equal_nan=True, rtol=1e-9, atol=1e-12), sig.name


def test_registry_names_unique():
    names = [s.name for s in REGISTRY]
    assert len(names) == len(set(names))
    assert len(names) >= 200
