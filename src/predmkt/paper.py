"""Paper trading: scan -> blind forecast -> decision -> ledger -> settlement -> score.

The ledger (paper/ledger.jsonl) is append-only per forecast and rewritten only
to record settlements. It is committed to git so the record survives the
throwaway cloud container and every forecast's timestamp is auditable.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import PM, PMConfig
from .polymarket import Market, PolymarketClient
from .sizing import decide


def ledger_path(cfg: PMConfig = PM):
    return cfg.paper_dir / "ledger.jsonl"


def load_ledger(cfg: PMConfig = PM) -> list[dict]:
    p = ledger_path(cfg)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def save_ledger(rows: list[dict], cfg: PMConfig = PM) -> None:
    cfg.paper_dir.mkdir(parents=True, exist_ok=True)
    ledger_path(cfg).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def append_ledger(row: dict, cfg: PMConfig = PM) -> None:
    cfg.paper_dir.mkdir(parents=True, exist_ok=True)
    with ledger_path(cfg).open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def eligible(m: Market, now: datetime, cfg: PMConfig = PM) -> bool:
    if not (m.active and not m.closed and m.is_binary and m.end):
        return False
    days = (m.end - now).total_seconds() / 86400
    if not (cfg.min_days_to_end <= days <= cfg.max_days_to_end):
        return False
    if m.volume < cfg.min_volume_usd:
        return False
    q = m.question.lower()
    return not any(re.search(p, q) for p in cfg.skip_patterns)


def scan(client: PolymarketClient, now: datetime, already: set[str], cfg: PMConfig = PM) -> list[Market]:
    ms = client.markets(closed=False, active="true", order="volumeNum", ascending="false",
                        max_pages=10)
    picks = [m for m in ms if eligible(m, now, cfg) and m.id not in already]
    # Mid-volume markets first: liquid enough to trade, less crowded than the top.
    picks.sort(key=lambda m: abs(math.log10(max(m.volume, 1)) - 5.0))
    return picks[: cfg.max_markets_per_run]


def run_once(client: PolymarketClient, forecaster, cfg: PMConfig = PM,
             now: datetime | None = None, dry_run: bool = False) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    rows = load_ledger(cfg)
    open_ids = {r["market_id"] for r in rows if r["status"] == "open"}
    exposure = sum(r["stake"] for r in rows if r["status"] == "open")
    new = []
    for m in scan(client, now, open_ids, cfg):
        f = forecaster.forecast(m, now)                     # blind: no price shown
        fresh = client.market(m.id)                         # quote fetched AFTER the forecast
        bid, ask = fresh.best_bid, fresh.best_ask
        mid = (bid + ask) / 2 if bid is not None and ask is not None else fresh.yes_price
        if f.refused or bid is None or ask is None:
            d = None
        else:
            d = decide(f.probability, bid, ask, fresh.category, cfg.bankroll, exposure, cfg)
            exposure += d.stake
        row = {
            "market_id": m.id, "question": m.question, "category": fresh.category,
            "end": m.end.isoformat() if m.end else None, "volume": fresh.volume,
            "forecast": asdict(f), "bid": bid, "ask": ask, "mid": mid,
            "side": d.side if d else None, "price": d.price if d else None,
            "cost": d.cost if d else None, "edge": d.edge if d else None,
            "stake": d.stake if d and d.side else 0.0,
            "shares": d.shares if d and d.side else 0.0,
            "decision": d.reason if d else "no forecast or no quote",
            "status": "open", "yes_won": None, "pnl": None, "settled_at": None,
        }
        new.append(row)
        if not dry_run:
            append_ledger(row, cfg)
    return new


def settle(client: PolymarketClient, cfg: PMConfig = PM, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    rows = load_ledger(cfg)
    n = 0
    for r in rows:
        if r["status"] != "open":
            continue
        m = client.market(r["market_id"])
        outcome = m.resolved_yes
        if outcome is None:
            if m.closed:            # closed but not cleanly 1/0 (e.g. 50-50 resolution)
                r["status"] = "void"
                r["pnl"] = 0.0
                r["settled_at"] = now.isoformat()
                n += 1
            continue
        r["yes_won"] = int(outcome)
        if r["side"]:
            won = (r["side"] == "YES") == outcome
            r["pnl"] = r["shares"] * (1.0 if won else 0.0) - r["stake"]
        else:
            r["pnl"] = 0.0
        r["status"] = "settled"
        r["settled_at"] = now.isoformat()
        n += 1
    save_ledger(rows, cfg)
    return n


def score(cfg: PMConfig = PM) -> dict:
    """Accuracy of Claude vs the market on settled questions, and paper P&L."""
    rows = [r for r in load_ledger(cfg) if r["status"] == "settled"]
    if not rows:
        return {"settled": 0}
    df = pd.DataFrame({
        "q": [r["forecast"]["probability"] for r in rows],
        "mid": [r["mid"] for r in rows],
        "y": [r["yes_won"] for r in rows],
        "stake": [r["stake"] for r in rows],
        "pnl": [r["pnl"] for r in rows],
        "category": [r["category"] for r in rows],
    }).dropna(subset=["q", "mid", "y"])
    eps = 1e-6

    def logloss(p):
        p = p.clip(eps, 1 - eps)
        return float(-(df.y * np.log(p) + (1 - df.y) * np.log(1 - p)).mean())

    bets = df[df.stake > 0]
    return {
        "settled": len(df),
        "brier_claude": float(((df.q - df.y) ** 2).mean()),
        "brier_market": float(((df.mid - df.y) ** 2).mean()),
        "logloss_claude": logloss(df.q),
        "logloss_market": logloss(df.mid),
        "claude_beats_market_share": float((((df.q - df.y) ** 2) < ((df.mid - df.y) ** 2)).mean()),
        "bets": len(bets),
        "staked": float(bets.stake.sum()),
        "pnl": float(bets.pnl.sum()),
        "roi": float(bets.pnl.sum() / bets.stake.sum()) if len(bets) else float("nan"),
        "bet_hit_rate": float((bets.pnl > 0).mean()) if len(bets) else float("nan"),
    }
