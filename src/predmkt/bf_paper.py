"""Betfair paper trading and historical calibration.

Ledger: paper/betfair_ledger.jsonl (one line per forecast market, committed).
Flow per run: scan -> forecast every runner blind -> fetch book -> decide -> log.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .betfair import EVENT_TYPES, BFMarket, Runner, read_stream_file, stream_snapshot_rows
from .bf_sizing import decide_market, settle_pnl
from .calibration import BUCKETS
from .config import PM, PMConfig

COMMISSION = 0.05


def ledger_path(cfg: PMConfig = PM) -> Path:
    return cfg.paper_dir / "betfair_ledger.jsonl"


def load(cfg: PMConfig = PM) -> list[dict]:
    p = ledger_path(cfg)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def save(rows: list[dict], cfg: PMConfig = PM) -> None:
    cfg.paper_dir.mkdir(parents=True, exist_ok=True)
    ledger_path(cfg).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def implied_probs(m: BFMarket) -> dict[int, float]:
    """Market's own probabilities: mid of back/lay (or last trade), normalised."""
    raw = {}
    for r in m.runners:
        if r.status != "ACTIVE":
            continue
        if r.back and r.lay:
            raw[r.selection_id] = (1 / r.back + 1 / r.lay) / 2
        elif r.ltp:
            raw[r.selection_id] = 1 / r.ltp
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()} if total > 0 else {}


def eligible(m: BFMarket, now: datetime, cfg: PMConfig = PM, min_matched: float = 5_000,
             max_runners: int = 20, max_days: float = 120) -> bool:
    if m.close is None or not (2 <= len(m.runners) <= max_runners):
        return False
    days = (m.close - now).total_seconds() / 86400
    return cfg.min_days_to_end <= days <= max_days and m.total_matched >= min_matched


def run_once(client, forecaster, cfg: PMConfig = PM, now: datetime | None = None,
             event_types: tuple = ("politics", "special_bets")) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    rows = load(cfg)
    open_ids = {r["market_id"] for r in rows if r["status"] == "open"}
    open_risk = sum(r["risk"] for r in rows if r["status"] == "open")
    cats = client.catalogue([EVENT_TYPES[e] for e in event_types])
    picks = [m for m in cats if eligible(m, now, cfg) and m.market_id not in open_ids]
    picks = picks[: cfg.max_markets_per_run]
    new = []
    for m in picks:
        names = [r.name for r in m.runners]
        f = forecaster.forecast_runners(m.question, m.rules, names, m.close, now)  # blind
        client.books([m])                                                          # prices after
        if f.get("refused") or not f.get("probabilities"):
            probs, d = {}, None
        else:
            name_to_id = {r.name: r.selection_id for r in m.runners}
            probs = {name_to_id[n]: p for n, p in f["probabilities"].items() if n in name_to_id}
            d = decide_market(m, probs, cfg.bankroll, open_risk, COMMISSION, cfg.min_edge,
                              cfg.kelly_fraction, cfg.max_stake_frac, cfg.max_open_frac)
            open_risk += d.risk
        row = {
            "market_id": m.market_id, "question": m.question, "close": m.close,
            "total_matched": m.total_matched, "forecast": f,
            "claude_probs": {str(k): v for k, v in probs.items()},
            "market_probs": {str(k): v for k, v in implied_probs(m).items()},
            "runners": {str(r.selection_id): r.name for r in m.runners},
            "selection_id": d.selection_id if d else None, "side": d.side if d else None,
            "odds": d.odds if d else None, "ev": d.ev_per_pound if d else None,
            "risk": d.risk if d else 0.0, "stake": d.stake if d else 0.0,
            "decision": d.reason if d else "no forecast",
            "status": "open", "winners": None, "pnl": None, "settled_at": None,
        }
        new.append(row)
        rows.append(row)
    save(rows, cfg)
    return new


def settle(client, cfg: PMConfig = PM, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    rows = load(cfg)
    pending = [r for r in rows if r["status"] == "open"]
    if not pending:
        return 0
    markets = [BFMarket(r["market_id"], "", "", "", None, 0.0,
                        runners=[Runner(int(k), v) for k, v in r["runners"].items()])
               for r in pending]
    client.books(markets)
    n = 0
    for m, r in zip(markets, pending):
        if not m.settled:
            continue
        winners = m.winners()
        r["winners"] = sorted(winners)
        if r["side"]:
            won = r["selection_id"] in winners
            r["pnl"] = settle_pnl(r["side"], r["odds"], r["stake"], won, COMMISSION)
        else:
            r["pnl"] = 0.0
        r["status"], r["settled_at"] = "settled", now.isoformat()
        n += 1
    save(rows, cfg)
    return n


def score(cfg: PMConfig = PM) -> dict:
    """Multi-outcome Brier score and log score of Claude vs the market, plus P&L."""
    rows = [r for r in load(cfg) if r["status"] == "settled" and r["claude_probs"]]
    if not rows:
        return {"settled": 0}
    bc, bm, lc, lm = [], [], [], []
    for r in rows:
        ids = r["runners"].keys()
        win = {str(w) for w in r["winners"]}
        for probs, b_out, l_out in [(r["claude_probs"], bc, lc), (r["market_probs"], bm, lm)]:
            b_out.append(sum((probs.get(i, 0.0) - (i in win)) ** 2 for i in ids))
            pw = sum(probs.get(i, 0.0) for i in win)
            l_out.append(-math.log(max(pw, 1e-6)))
    bets = [r for r in rows if r["side"]]
    staked = sum(r["risk"] for r in bets)
    pnl = sum(r["pnl"] for r in bets)
    return {"settled": len(rows), "brier_claude": float(np.mean(bc)),
            "brier_market": float(np.mean(bm)), "logscore_claude": float(np.mean(lc)),
            "logscore_market": float(np.mean(lm)), "bets": len(bets), "risked": staked,
            "pnl": pnl, "roi": pnl / staked if staked else float("nan")}


# --------------------------------------------------------------------------
# Historical calibration from Betfair's downloadable stream files
# --------------------------------------------------------------------------

def build_bf_history(folder: Path, days: tuple = (1, 7, 30)) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and (p.suffix == ".bz2" or p.name.startswith("1.")):
            rows.extend(stream_snapshot_rows(read_stream_file(p), days))
    return pd.DataFrame(rows)


def bf_calibration(snap: pd.DataFrame, commission: float = COMMISSION) -> pd.DataFrame:
    """Backing every runner at its last traded price: does the implied
    probability match how often runners at that price win?"""
    c = snap.assign(price=1 / snap.odds)
    c["ret"] = np.where(c.won == 1, (c.odds - 1) * (1 - commission), -1.0)
    c["bucket"] = pd.cut(c.price, BUCKETS, include_lowest=True)
    g = c.groupby("bucket", observed=True)
    t = pd.DataFrame({"runners": g.size(), "avg_implied": g.price.mean(),
                      "win_rate": g.won.mean(), "back_return_per_£": g.ret.mean()})
    t["se"] = g.ret.std() / np.sqrt(t.runners)
    return t
