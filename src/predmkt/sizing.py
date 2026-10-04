"""Turn a probability and a price into a bet (or no bet).

A binary contract bought at all-in cost c pays 1 if it wins. With belief q the
expected profit per share is q - c and the Kelly fraction of bankroll is
(q - c) / (1 - c). We bet a fraction of Kelly and cap the stake.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import PM, PMConfig, fee_per_share


@dataclass
class Decision:
    side: str | None          # "YES", "NO" or None (no bet)
    price: float | None       # price paid per share, before fee
    cost: float | None        # all-in cost per share
    edge: float               # model prob - cost, per share
    stake: float              # dollars
    shares: float
    reason: str


def decide(q: float, best_bid: float, best_ask: float, category: str | None,
           bankroll: float, open_exposure: float = 0.0, cfg: PMConfig = PM) -> Decision:
    """q = model probability of YES. Buying YES pays the ask; buying NO costs
    1 - bid (the other side of the YES bid)."""
    if not (0 < best_ask <= 1 and 0 <= best_bid < 1 and best_bid <= best_ask):
        return Decision(None, None, None, 0.0, 0.0, 0.0, "no valid quote")
    options = []
    for side, prob, price in [("YES", q, best_ask), ("NO", 1 - q, 1 - best_bid)]:
        cost = price + fee_per_share(price, category, cfg)
        options.append((prob - cost, side, prob, price, cost))
    edge, side, prob, price, cost = max(options)
    if edge < cfg.min_edge:
        return Decision(None, None, None, edge, 0.0, 0.0,
                        f"edge {edge:+.3f} below {cfg.min_edge:.2f}")
    kelly = edge / (1 - cost)
    frac = min(cfg.kelly_fraction * kelly, cfg.max_stake_frac)
    room = max(0.0, cfg.max_open_frac * bankroll - open_exposure)
    stake = min(frac * bankroll, room)
    if stake <= 0:
        return Decision(None, None, None, edge, 0.0, 0.0, "exposure cap reached")
    return Decision(side, price, cost, edge, stake, stake / cost,
                    f"{side} edge {edge:+.3f}, kelly {kelly:.3f}")
