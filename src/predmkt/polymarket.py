"""Read-only client for Polymarket's public market data (no account, no trading).

Gamma API: market metadata, rules, current prices, resolution.
CLOB API:  price history per outcome token.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .config import PM, PMConfig


def _parse_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return []


def _parse_time(v) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class Market:
    id: str
    question: str
    rules: str
    end: datetime | None
    closed: bool
    active: bool
    outcomes: list
    prices: list                 # current (or final) price per outcome
    token_ids: list
    volume: float
    liquidity: float
    best_bid: float | None
    best_ask: float | None
    category: str | None
    tags: list
    neg_risk: bool
    raw: dict

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2 and len(self.token_ids) == 2

    @property
    def yes_index(self) -> int:
        low = [str(o).lower() for o in self.outcomes]
        return low.index("yes") if "yes" in low else 0

    @property
    def resolved_yes(self) -> bool | None:
        """True/False once settled to 1/0; None while open or ambiguous."""
        if not self.closed or len(self.prices) != 2:
            return None
        p = _num(self.prices[self.yes_index], -1)
        if p >= 0.99:
            return True
        if p <= 0.01:
            return False
        return None

    @property
    def yes_price(self) -> float | None:
        if len(self.prices) != 2:
            return None
        return _num(self.prices[self.yes_index], None)


def parse_market(m: dict) -> Market:
    tags = []
    for ev in m.get("events") or []:
        for t in ev.get("tags") or []:
            label = t.get("label") or t.get("slug")
            if label:
                tags.append(str(label).lower())
    category = (m.get("category") or (tags[0] if tags else None))
    return Market(
        id=str(m.get("id")),
        question=m.get("question") or "",
        rules=m.get("description") or "",
        end=_parse_time(m.get("endDate") or m.get("endDateIso")),
        closed=bool(m.get("closed")),
        active=bool(m.get("active")),
        outcomes=_parse_list(m.get("outcomes")),
        prices=[_num(x) for x in _parse_list(m.get("outcomePrices"))],
        token_ids=[str(x) for x in _parse_list(m.get("clobTokenIds"))],
        volume=_num(m.get("volumeNum", m.get("volume"))),
        liquidity=_num(m.get("liquidityNum", m.get("liquidity"))),
        best_bid=_num(m.get("bestBid"), None) if m.get("bestBid") is not None else None,
        best_ask=_num(m.get("bestAsk"), None) if m.get("bestAsk") is not None else None,
        category=category.lower() if isinstance(category, str) else None,
        tags=tags,
        neg_risk=bool(m.get("negRisk")),
        raw=m,
    )


class PolymarketClient:
    def __init__(self, cfg: PMConfig = PM, session: requests.Session | None = None):
        self.cfg = cfg
        self.http = session or requests.Session()

    def _get(self, base: str, path: str, params: dict | None = None):
        for attempt in range(4):
            r = self.http.get(f"{base}{path}", params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(self.cfg.request_pause_s)
            return r.json()
        r.raise_for_status()

    def markets(self, closed: bool | None = None, limit: int = 500, max_pages: int = 1000,
                **filters) -> list[Market]:
        """Page through Gamma /markets. Extra filters pass straight through
        (e.g. end_date_min="2025-01-01", order="volumeNum", ascending=False)."""
        out, offset = [], 0
        for _ in range(max_pages):
            params = {"limit": limit, "offset": offset, **filters}
            if closed is not None:
                params["closed"] = str(closed).lower()
            page = self._get(self.cfg.gamma_url, "/markets", params)
            if not page:
                break
            out.extend(parse_market(m) for m in page)
            if len(page) < limit:
                break
            offset += limit
        return out

    def market(self, market_id: str) -> Market:
        return parse_market(self._get(self.cfg.gamma_url, f"/markets/{market_id}"))

    def price_history(self, token_id: str, start: datetime | None = None,
                      end: datetime | None = None, fidelity_min: int = 60) -> list[tuple[int, float]]:
        params = {"market": token_id, "fidelity": fidelity_min}
        if start and end:
            params["startTs"] = int(start.timestamp())
            params["endTs"] = int(end.timestamp())
        else:
            params["interval"] = "max"
        data = self._get(self.cfg.clob_url, "/prices-history", params) or {}
        return [(int(h["t"]), float(h["p"])) for h in data.get("history", [])]
