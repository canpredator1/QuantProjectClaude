"""Back/lay decisions on an exchange that charges commission on winnings.

Back at decimal odds o:  win (o-1)(1-c) per £1 staked, lose £1.
Lay at decimal odds o:   per £1 of liability, win (1-c)/(o-1), lose £1.
Both are bets with net payout multiple b; with win probability w the expected
profit per £1 risked is w*b - (1-w) and the Kelly fraction is that divided by b.
"""
from __future__ import annotations

from dataclasses import dataclass

from .betfair import BFMarket


@dataclass
class BFDecision:
    selection_id: int | None
    runner: str | None
    side: str | None            # "BACK", "LAY" or None
    odds: float | None
    ev_per_pound: float         # expected profit per £1 risked, after commission
    risk: float                 # £ at risk (stake for backs, liability for lays)
    stake: float                # backer's stake (for lays: risk / (odds - 1))
    reason: str


def _bet(win_prob: float, b: float):
    ev = win_prob * b - (1 - win_prob)
    return ev, (ev / b if b > 0 else -1.0)


def decide_market(m: BFMarket, probs: dict[int, float], bankroll: float, open_risk: float,
                  commission: float = 0.05, min_ev: float = 0.05, kelly_fraction: float = 0.25,
                  max_risk_frac: float = 0.02, max_open_frac: float = 0.30,
                  min_liquidity: float = 20.0) -> BFDecision:
    """Pick the single best back or lay in the market, or no bet."""
    best = None
    for r in m.runners:
        q = probs.get(r.selection_id)
        if q is None or r.status != "ACTIVE":
            continue
        if r.back and r.back > 1.0 and r.back_size >= min_liquidity:
            ev, k = _bet(q, (r.back - 1) * (1 - commission))
            cand = (ev, k, r, "BACK", r.back)
            best = cand if best is None or ev > best[0] else best
        if r.lay and r.lay > 1.0 and r.lay_size >= min_liquidity:
            ev, k = _bet(1 - q, (1 - commission) / (r.lay - 1))
            cand = (ev, k, r, "LAY", r.lay)
            best = cand if best is None or ev > best[0] else best
    if best is None:
        return BFDecision(None, None, None, None, 0.0, 0.0, 0.0, "no priced runners")
    ev, kelly, r, side, odds = best
    if ev < min_ev:
        return BFDecision(None, None, None, None, ev, 0.0, 0.0,
                          f"best EV {ev:+.3f}/£ below {min_ev:.2f}")
    risk = min(kelly_fraction * kelly, max_risk_frac) * bankroll
    risk = min(risk, max(0.0, max_open_frac * bankroll - open_risk))
    if risk <= 0:
        return BFDecision(None, None, None, None, ev, 0.0, 0.0, "exposure cap reached")
    stake = risk if side == "BACK" else risk / (odds - 1)
    return BFDecision(r.selection_id, r.name, side, odds, ev, risk, stake,
                      f"{side} {r.name} @ {odds} EV {ev:+.3f}/£")


def settle_pnl(side: str, odds: float, stake: float, won: bool, commission: float = 0.05) -> float:
    """P&L of one bet once the runner's result is known."""
    if side == "BACK":
        return stake * (odds - 1) * (1 - commission) if won else -stake
    # LAY: we keep the backer's stake if the runner loses
    return -stake * (odds - 1) if won else stake * (1 - commission)
