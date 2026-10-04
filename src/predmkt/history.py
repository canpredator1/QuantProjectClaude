"""Phase 1 data: resolved binary markets with the price N days before the end.

One row per (market, snapshot horizon): the YES price at end - N days and the
final outcome. Snapshots use only the price history up to that moment.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .config import PM, PMConfig
from .polymarket import Market, PolymarketClient


def price_at(history: list[tuple[int, float]], ts: int) -> float | None:
    """Last traded price at or before ts (None if no trade yet)."""
    best = None
    for t, p in history:
        if t <= ts:
            best = p
        else:
            break
    return best


def snapshot_rows(m: Market, history: list[tuple[int, float]], days: tuple) -> list[dict]:
    outcome = m.resolved_yes
    if outcome is None or m.end is None or not history:
        return []
    history = sorted(history)
    rows = []
    for d in days:
        ts = int((m.end - timedelta(days=d)).timestamp())
        p = price_at(history, ts)
        if p is None or not (0.0 < p < 1.0):
            continue
        rows.append({
            "market_id": m.id, "question": m.question, "category": m.category,
            "end": m.end, "days_before": d, "yes_price": p, "yes_won": int(outcome),
            "volume": m.volume, "neg_risk": m.neg_risk,
        })
    return rows


def build_history(client: PolymarketClient, cfg: PMConfig = PM, start: str = "2023-01-01",
                  min_volume: float = 1_000, limit_markets: int | None = None,
                  verbose: bool = True) -> pd.DataFrame:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.data_dir / "resolved_snapshots.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    seen = set(done.market_id) if len(done) else set()

    markets = [m for m in client.markets(closed=True, end_date_min=start)
               if m.is_binary and m.volume >= min_volume and m.id not in seen
               and m.resolved_yes is not None]
    if limit_markets:
        markets = markets[:limit_markets]
    rows = []
    for i, m in enumerate(markets, 1):
        token = m.token_ids[m.yes_index]
        start_ts = m.end - timedelta(days=max(cfg.snapshot_days) + 2)
        hist = client.price_history(token, start=start_ts, end=m.end, fidelity_min=60)
        rows.extend(snapshot_rows(m, hist, cfg.snapshot_days))
        if verbose and i % 100 == 0:
            print(f"{i}/{len(markets)} markets, {len(rows)} snapshots", flush=True)
            pd.concat([done, pd.DataFrame(rows)]).to_parquet(out_path)   # checkpoint
    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(out_path)
    return df
