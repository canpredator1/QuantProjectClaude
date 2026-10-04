"""Offline tests for the prediction-market toolkit (no network, no API calls)."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from predmkt.calibration import calibration_table, contracts, strategy_test
from predmkt.config import PMConfig, fee_per_share
from predmkt.forecaster import ClaudeForecaster, Forecast, build_prompt, parse_forecast_json
from predmkt.history import price_at, snapshot_rows
from predmkt import paper
from predmkt.polymarket import parse_market
from predmkt.sizing import decide

NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)


def gamma_market(**kw):
    m = {
        "id": "123", "question": "Will X happen by Oct 31?", "description": "Resolves YES if X.",
        "endDate": "2026-10-31T23:59:00Z", "closed": False, "active": True,
        "outcomes": '["Yes", "No"]', "outcomePrices": '["0.30", "0.70"]',
        "clobTokenIds": '["111", "222"]', "volumeNum": 50000, "liquidityNum": 8000,
        "bestBid": 0.29, "bestAsk": 0.31, "negRisk": False,
        "events": [{"tags": [{"label": "Politics"}]}],
    }
    m.update(kw)
    return m


def test_parse_market_fields():
    m = parse_market(gamma_market())
    assert m.is_binary and m.yes_index == 0
    assert m.yes_price == pytest.approx(0.30)
    assert m.category == "politics"
    assert m.end == datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc)
    assert m.resolved_yes is None


def test_resolution_detection():
    assert parse_market(gamma_market(closed=True, outcomePrices='["1", "0"]')).resolved_yes is True
    assert parse_market(gamma_market(closed=True, outcomePrices='["0", "1"]')).resolved_yes is False
    assert parse_market(gamma_market(closed=True, outcomePrices='["0.5", "0.5"]')).resolved_yes is None
    # outcomes listed No-first are handled
    m = parse_market(gamma_market(outcomes='["No", "Yes"]', closed=True, outcomePrices='["0", "1"]'))
    assert m.resolved_yes is True


def test_fee_schedule_peaks_at_half():
    assert fee_per_share(0.5, "politics") == pytest.approx(0.01)
    assert fee_per_share(0.5, "crypto") == pytest.approx(0.0175)
    assert fee_per_share(0.5, "geopolitics") == 0
    assert fee_per_share(0.95, "politics") < fee_per_share(0.5, "politics")


def test_price_at_uses_only_past():
    hist = [(100, 0.2), (200, 0.4), (300, 0.9)]
    assert price_at(hist, 50) is None
    assert price_at(hist, 250) == 0.4
    assert price_at(hist, 300) == 0.9


def test_snapshot_rows():
    m = parse_market(gamma_market(closed=True, outcomePrices='["1", "0"]'))
    end_ts = int(m.end.timestamp())
    hist = [(end_ts - 40 * 86400, 0.10), (end_ts - 5 * 86400, 0.60), (end_ts - 3600, 0.97)]
    rows = snapshot_rows(m, hist, (1, 7, 30))
    got = {r["days_before"]: r["yes_price"] for r in rows}
    assert got == {1: 0.60, 7: 0.10, 30: 0.10}
    assert all(r["yes_won"] == 1 for r in rows)


def _snap(n=4000, seed=0, bias=0.0):
    """Synthetic resolved markets. bias>0 makes long shots win less than priced."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    true_p = np.clip(p + bias * (p - 0.5) * 0.2, 0.001, 0.999)
    won = (rng.uniform(size=n) < true_p).astype(int)
    ends = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"market_id": [str(i) for i in range(n)], "category": "politics",
                         "end": ends, "days_before": 7, "yes_price": p, "yes_won": won,
                         "question": "q", "volume": 1e5, "neg_risk": False})


def test_contracts_are_symmetric_and_fair_prices_lose_the_fee():
    c = contracts(_snap(20000, bias=0.0))
    assert len(c) == 40000
    t = calibration_table(c)
    # fair prices: win rate tracks price and returns are about minus the fee
    assert (t.gap.abs() < 0.05).all()
    assert c.ret.mean() == pytest.approx(-c.fee.mean() / c.price.mean(), abs=0.03)


def test_strategy_test_splits_by_time():
    res = strategy_test(_snap(4000), 0.9, 0.98, 7)
    assert set(res) == {"first half", "second half"}
    assert res["first half"]["bets"] > 0 and res["second half"]["bets"] > 0


def test_decide_buys_yes_when_model_higher():
    d = decide(0.60, best_bid=0.40, best_ask=0.42, category="politics", bankroll=10_000)
    assert d.side == "YES" and d.price == 0.42
    assert d.cost == pytest.approx(0.42 + 0.04 * 0.42 * 0.58)
    assert 0 < d.stake <= 200          # capped at 2% of bankroll


def test_decide_buys_no_when_model_lower():
    d = decide(0.10, best_bid=0.40, best_ask=0.42, category="politics", bankroll=10_000)
    assert d.side == "NO" and d.price == pytest.approx(0.60)


def test_decide_passes_on_small_edge_and_exposure_cap():
    assert decide(0.43, 0.40, 0.42, "politics", 10_000).side is None
    cfg = PMConfig(max_open_frac=0.30)
    d = decide(0.9, 0.40, 0.42, "politics", 10_000, open_exposure=3_000, cfg=cfg)
    assert d.side is None and "cap" in d.reason


def test_parse_forecast_json():
    text = 'Some research...\n```json\n{"probability": 0.37, "reasoning": "r", "key_facts": [], "rules_risk": "none"}\n```'
    assert parse_forecast_json(text)["probability"] == 0.37
    with pytest.raises(ValueError):
        parse_forecast_json("no json here")
    assert parse_forecast_json('```json\n{"probability": 0.999}\n```')["probability"] == 0.99


def test_prompt_never_contains_price():
    m = parse_market(gamma_market())
    p = build_prompt(m, NOW)
    assert "0.30" not in p and "0.31" not in p and "0.29" not in p
    assert m.question in p and "Resolves YES if X." in p


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def _resp(stop, blocks):
    return SimpleNamespace(stop_reason=stop, content=blocks, model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def test_forecaster_handles_pause_and_blocks_market_sites():
    final = 'Done.\n```json\n{"probability": 0.25, "reasoning": "base rate", "key_facts": ["a"], "rules_risk": "none"}\n```'
    fake = FakeMessages([
        _resp("pause_turn", [SimpleNamespace(type="server_tool_use")]),
        _resp("end_turn", [SimpleNamespace(type="text", text=final)]),
    ])
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    f = ClaudeForecaster(PMConfig(), client=client).forecast(parse_market(gamma_market()), NOW)
    assert f.probability == 0.25 and f.searches == 1 and len(fake.calls) == 2
    tool = fake.calls[0]["tools"][0]
    assert "polymarket.com" in tool["blocked_domains"]
    assert fake.calls[0]["fallbacks"] == "default"


def test_forecaster_refusal():
    fake = FakeMessages([_resp("refusal", [])])
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    f = ClaudeForecaster(PMConfig(), client=client).forecast(parse_market(gamma_market()), NOW)
    assert f.refused and np.isnan(f.probability)


class FakePM:
    def __init__(self, markets):
        self.store = {m["id"]: m for m in markets}

    def markets(self, **kw):
        return [parse_market(m) for m in self.store.values()]

    def market(self, mid):
        return parse_market(self.store[mid])


class FixedForecaster:
    def __init__(self, p):
        self.p = p

    def forecast(self, m, now):
        return Forecast(m.id, self.p, "r", [], "none", "claude-opus-5-5", now.isoformat(), 1, 1, 0)


def test_paper_cycle_end_to_end(tmp_path):
    cfg = PMConfig(paper_dir=tmp_path)
    pm = FakePM([gamma_market(id="1"),
                 gamma_market(id="2", question="BTC up or down today?"),      # skipped pattern
                 gamma_market(id="3", volumeNum=10)])                          # too little volume
    rows = paper.run_once(pm, FixedForecaster(0.60), cfg, now=NOW)
    assert [r["market_id"] for r in rows] == ["1"]
    assert rows[0]["side"] == "YES" and rows[0]["stake"] > 0
    # running again does not re-forecast an open market
    assert paper.run_once(pm, FixedForecaster(0.60), cfg, now=NOW) == []
    # market resolves YES -> profit
    pm.store["1"].update(closed=True, outcomePrices='["1", "0"]')
    assert paper.settle(pm, cfg, now=NOW + timedelta(days=30)) == 1
    r = paper.load_ledger(cfg)[0]
    assert r["status"] == "settled" and r["pnl"] > 0
    s = paper.score(cfg)
    assert s["settled"] == 1 and s["bets"] == 1
    assert s["brier_claude"] == pytest.approx((0.60 - 1) ** 2)
    assert s["brier_market"] == pytest.approx((0.30 - 1) ** 2)
