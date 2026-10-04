"""Offline tests for the Betfair Exchange pieces (fake API, synthetic files)."""
import bz2
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from predmkt import bf_paper
from predmkt.betfair import (BetfairClient, apply_book, parse_catalogue, read_stream_file,
                             stream_snapshot_rows)
from predmkt.bf_sizing import decide_market, settle_pnl
from predmkt.config import PMConfig
from predmkt.forecaster import ClaudeForecaster, parse_runner_json

NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)


def catalogue_entry(mid="1.234", close=NOW + timedelta(days=30)):
    return {
        "marketId": mid, "marketName": "Next Taoiseach", "totalMatched": 25000.0,
        "event": {"name": "Irish Politics"},
        "description": {"rules": "Settled on the person formally appointed.",
                        "marketTime": close.isoformat().replace("+00:00", "Z")},
        "runners": [{"selectionId": 11, "runnerName": "Candidate A"},
                    {"selectionId": 22, "runnerName": "Candidate B"},
                    {"selectionId": 33, "runnerName": "Candidate C"}],
    }


def book_entry(mid="1.234", status="OPEN", statuses=("ACTIVE",) * 3,
               prices=((1.8, 1.9), (3.0, 3.2), (8.0, 9.0))):
    runners = []
    for sid, st, (b, l) in zip((11, 22, 33), statuses, prices):
        runners.append({"selectionId": sid, "status": st, "lastPriceTraded": b,
                        "ex": {"availableToBack": [{"price": b, "size": 500}],
                               "availableToLay": [{"price": l, "size": 500}]}})
    return {"marketId": mid, "status": status, "runners": runners}


def test_parse_catalogue_and_book():
    m = apply_book(parse_catalogue(catalogue_entry()), book_entry())
    assert m.question == "Irish Politics: Next Taoiseach"
    assert [r.name for r in m.runners] == ["Candidate A", "Candidate B", "Candidate C"]
    assert m.runners[0].back == 1.8 and m.runners[0].lay == 1.9
    assert not m.settled
    m2 = apply_book(parse_catalogue(catalogue_entry()),
                    book_entry(status="CLOSED", statuses=("LOSER", "WINNER", "LOSER")))
    assert m2.settled and m2.winners() == {22}


def test_implied_probs_sum_to_one():
    m = apply_book(parse_catalogue(catalogue_entry()), book_entry())
    p = bf_paper.implied_probs(m)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p[11] > p[22] > p[33]


def test_back_when_model_more_bullish():
    m = apply_book(parse_catalogue(catalogue_entry()), book_entry())
    # market ~ 0.20 for B (odds 3.0-3.2); model says 0.45 -> back B
    d = decide_market(m, {11: 0.45, 22: 0.45, 33: 0.10}, bankroll=10_000, open_risk=0)
    assert d.side == "BACK" and d.selection_id == 22 and d.odds == 3.0
    assert d.ev_per_pound == pytest.approx(0.45 * 2.0 * 0.95 - 0.55)
    assert 0 < d.risk <= 200


def test_lay_when_model_more_bearish():
    m = apply_book(parse_catalogue(catalogue_entry()), book_entry())
    # favourite A priced ~0.54; model says 0.20 -> lay A at 1.9
    d = decide_market(m, {11: 0.20, 22: 0.30, 33: 0.11}, bankroll=10_000, open_risk=0)
    assert d.side == "LAY" and d.selection_id == 11 and d.odds == 1.9
    assert d.stake == pytest.approx(d.risk / 0.9)


def test_no_bet_when_agreeing_with_market():
    m = apply_book(parse_catalogue(catalogue_entry()), book_entry())
    p = bf_paper.implied_probs(m)
    assert decide_market(m, p, 10_000, 0).side is None


def test_settle_pnl_with_commission():
    assert settle_pnl("BACK", 3.0, 10, True) == pytest.approx(19.0)
    assert settle_pnl("BACK", 3.0, 10, False) == -10
    assert settle_pnl("LAY", 3.0, 10, True) == -20
    assert settle_pnl("LAY", 3.0, 10, False) == pytest.approx(9.5)


def test_parse_runner_json_matches_names_and_normalises():
    text = '```json\n{"probabilities": {"candidate a": 0.5, "Candidate B": 0.3, "Candidate C": 0.1}, "reasoning": "r"}\n```'
    d = parse_runner_json(text, ["Candidate A", "Candidate B", "Candidate C"])
    p = d["probabilities"]
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["Candidate A"] == pytest.approx(0.5 / 0.9)
    with pytest.raises(ValueError):
        parse_runner_json("nothing", ["A"])


def test_forecast_runners_never_sends_prices():
    final = '```json\n{"probabilities": {"Candidate A": 0.6, "Candidate B": 0.3, "Candidate C": 0.1}, "reasoning": "r", "key_facts": [], "rules_risk": "none"}\n```'
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                               content=[SimpleNamespace(type="text", text=final)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    f = ClaudeForecaster(PMConfig(), client=client).forecast_runners(
        "Irish Politics: Next Taoiseach", "rules", ["Candidate A", "Candidate B", "Candidate C"],
        NOW + timedelta(days=30), NOW)
    assert f["probabilities"]["Candidate A"] == pytest.approx(0.6)
    prompt = calls[0]["messages"][0]["content"]
    assert "1.8" not in prompt and "3.0" not in prompt and "Candidate B" in prompt


class FakeBF:
    def __init__(self):
        self.closed = False

    def catalogue(self, ids):
        return [parse_catalogue(catalogue_entry())]

    def books(self, markets):
        for m in markets:
            if self.closed:
                apply_book(m, book_entry(status="CLOSED", statuses=("LOSER", "WINNER", "LOSER")))
            else:
                apply_book(m, book_entry())
        return markets


class FixedRunnerForecaster:
    def forecast_runners(self, q, rules, runners, close, now):
        return {"refused": False, "probabilities": dict(zip(runners, [0.45, 0.45, 0.10])),
                "reasoning": "r", "model": "x", "created_at": now.isoformat()}


def test_betfair_paper_cycle(tmp_path):
    cfg = PMConfig(paper_dir=tmp_path)
    bf = FakeBF()
    rows = bf_paper.run_once(bf, FixedRunnerForecaster(), cfg, now=NOW)
    assert len(rows) == 1 and rows[0]["side"] == "BACK" and rows[0]["selection_id"] == 22
    assert bf_paper.run_once(bf, FixedRunnerForecaster(), cfg, now=NOW) == []   # no repeat
    bf.closed = True
    assert bf_paper.settle(bf, cfg, now=NOW + timedelta(days=31)) == 1
    s = bf_paper.score(cfg)
    assert s["settled"] == 1 and s["bets"] == 1 and s["pnl"] > 0
    assert s["brier_claude"] < s["brier_market"]       # model gave the winner more weight


def _stream_file(tmp_path, winner=22):
    end = NOW
    ms = lambda dt: int(dt.timestamp() * 1000)  # noqa: E731
    runners = [{"id": i, "name": n, "status": "ACTIVE"} for i, n in [(11, "A"), (22, "B")]]
    lines = [
        {"op": "mcm", "pt": ms(end - timedelta(days=40)), "mc": [{"id": "1.9", "marketDefinition": {
            "status": "OPEN", "marketTime": end.isoformat(), "eventTypeId": "2378961",
            "name": "Winner", "eventName": "Election", "runners": runners}}]},
        {"op": "mcm", "pt": ms(end - timedelta(days=20)), "mc": [{"id": "1.9", "rc": [
            {"id": 11, "ltp": 1.5}, {"id": 22, "ltp": 3.0}]}]},
        {"op": "mcm", "pt": ms(end - timedelta(hours=30)), "mc": [{"id": "1.9", "rc": [
            {"id": 11, "ltp": 2.5}, {"id": 22, "ltp": 1.7}]}]},
        {"op": "mcm", "pt": ms(end + timedelta(hours=2)), "mc": [{"id": "1.9", "marketDefinition": {
            "status": "CLOSED", "marketTime": end.isoformat(), "eventTypeId": "2378961",
            "name": "Winner", "eventName": "Election",
            "runners": [{"id": 11, "name": "A", "status": "WINNER" if winner == 11 else "LOSER"},
                        {"id": 22, "name": "B", "status": "WINNER" if winner == 22 else "LOSER"}]}}]},
    ]
    p = tmp_path / "1.9.bz2"
    with bz2.open(p, "wt") as f:
        f.write("\n".join(json.dumps(x) for x in lines))
    return p


def test_stream_file_snapshots_use_only_past_prices(tmp_path):
    rec = read_stream_file(_stream_file(tmp_path))
    rows = stream_snapshot_rows(rec, (1, 7, 30))
    got = {(r["days_before"], r["runner"]): (r["odds"], r["won"]) for r in rows}
    assert got[(1, "B")] == (1.7, 1) and got[(1, "A")] == (2.5, 0)
    assert got[(7, "B")] == (3.0, 1)
    assert (30, "A") not in got                 # no trade had happened 30 days out


def test_bf_calibration_table(tmp_path):
    for i, w in enumerate([22, 11, 22]):
        d = tmp_path / str(i)
        d.mkdir()
        _stream_file(d, winner=w)
    snap = bf_paper.build_bf_history(tmp_path)
    t = bf_paper.bf_calibration(snap[snap.days_before == 1])
    assert t.runners.sum() == 6
    assert set(t.columns) >= {"avg_implied", "win_rate", "back_return_per_£"}


def test_login_uses_app_key(monkeypatch):
    sent = {}

    class FakeSession:
        def post(self, url, data=None, headers=None, timeout=None, cert=None):
            sent.update(url=url, headers=headers)
            return SimpleNamespace(json=lambda: {"status": "SUCCESS", "token": "tok"})
    monkeypatch.delenv("BETFAIR_CERT_FILE", raising=False)
    c = BetfairClient(app_key="KEY", session=FakeSession())
    assert c.login("u", "p") == "tok"
    assert sent["headers"]["X-Application"] == "KEY"
    assert "identitysso.betfair.com" in sent["url"]
