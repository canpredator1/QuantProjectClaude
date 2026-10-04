"""Betfair Exchange: API client, market parsing and historical-data reader.

Read-only use needs a Betfair account and the free *delayed* application key.
Placing real bets would need the live key; this module never places bets.

Credentials come from the environment:
  BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY
  BETFAIR_CERT_FILE / BETFAIR_KEY_FILE (optional: non-interactive cert login)
"""
from __future__ import annotations

import bz2
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
LOGIN_URL = "https://identitysso.betfair.com/api/login"
CERT_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"

EVENT_TYPES = {"politics": "2378961", "special_bets": "10"}


def _t(v) -> datetime | None:
    if not v:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000, tz=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class Runner:
    selection_id: int
    name: str
    status: str = "ACTIVE"              # ACTIVE / WINNER / LOSER / REMOVED
    back: float | None = None           # best price available to back (decimal odds)
    lay: float | None = None            # best price available to lay
    back_size: float = 0.0
    lay_size: float = 0.0
    ltp: float | None = None            # last traded price


@dataclass
class BFMarket:
    market_id: str
    name: str
    event: str
    rules: str
    close: datetime | None
    total_matched: float
    status: str = "OPEN"                # OPEN / SUSPENDED / CLOSED
    runners: list[Runner] = field(default_factory=list)

    @property
    def question(self) -> str:
        return f"{self.event}: {self.name}" if self.event and self.event != self.name else self.name

    @property
    def settled(self) -> bool:
        return self.status == "CLOSED" and any(r.status == "WINNER" for r in self.runners)

    def winners(self) -> set[int]:
        return {r.selection_id for r in self.runners if r.status == "WINNER"}


def parse_catalogue(c: dict) -> BFMarket:
    desc = c.get("description") or {}
    return BFMarket(
        market_id=c["marketId"],
        name=c.get("marketName", ""),
        event=(c.get("event") or {}).get("name", ""),
        rules=desc.get("rules", "") or "",
        close=_t(desc.get("marketTime") or c.get("marketStartTime")),
        total_matched=float(c.get("totalMatched") or 0.0),
        runners=[Runner(r["selectionId"], r.get("runnerName", str(r["selectionId"])))
                 for r in c.get("runners", [])],
    )


def apply_book(m: BFMarket, book: dict) -> BFMarket:
    """Fill prices and statuses from a listMarketBook entry."""
    m.status = book.get("status", m.status)
    by_id = {r.selection_id: r for r in m.runners}
    for rb in book.get("runners", []):
        r = by_id.get(rb["selectionId"])
        if r is None:
            continue
        r.status = rb.get("status", r.status)
        r.ltp = rb.get("lastPriceTraded")
        ex = rb.get("ex") or {}
        backs, lays = ex.get("availableToBack") or [], ex.get("availableToLay") or []
        r.back, r.back_size = (backs[0]["price"], backs[0]["size"]) if backs else (None, 0.0)
        r.lay, r.lay_size = (lays[0]["price"], lays[0]["size"]) if lays else (None, 0.0)
    return m


class BetfairClient:
    def __init__(self, app_key: str | None = None, session: requests.Session | None = None):
        self.app_key = app_key or os.environ.get("BETFAIR_APP_KEY", "")
        self.http = session or requests.Session()
        self.token: str | None = None

    def login(self, username: str | None = None, password: str | None = None) -> str:
        username = username or os.environ["BETFAIR_USERNAME"]
        password = password or os.environ["BETFAIR_PASSWORD"]
        headers = {"X-Application": self.app_key, "Accept": "application/json"}
        cert, key = os.environ.get("BETFAIR_CERT_FILE"), os.environ.get("BETFAIR_KEY_FILE")
        if cert and key:
            r = self.http.post(CERT_LOGIN_URL, data={"username": username, "password": password},
                               headers=headers, cert=(cert, key), timeout=30)
            body = r.json()
            ok, token = body.get("loginStatus") == "SUCCESS", body.get("sessionToken")
        else:
            r = self.http.post(LOGIN_URL, data={"username": username, "password": password},
                               headers=headers, timeout=30)
            body = r.json()
            ok, token = body.get("status") == "SUCCESS", body.get("token")
        if not ok:
            raise RuntimeError(f"Betfair login failed: {body}")
        self.token = token
        return token

    def _call(self, method: str, params: dict):
        if not self.token:
            self.login()
        payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}",
                   "params": params, "id": 1}
        headers = {"X-Application": self.app_key, "X-Authentication": self.token,
                   "Content-Type": "application/json"}
        for attempt in range(4):
            r = self.http.post(BETTING_URL, data=json.dumps(payload), headers=headers, timeout=30)
            body = r.json()
            if "error" in body:
                if attempt < 3 and "TOO_MUCH_DATA" not in json.dumps(body["error"]):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Betfair {method} error: {body['error']}")
            return body["result"]

    def catalogue(self, event_type_ids: list[str], max_results: int = 200,
                  text_query: str | None = None) -> list[BFMarket]:
        flt = {"eventTypeIds": event_type_ids}
        if text_query:
            flt["textQuery"] = text_query
        res = self._call("listMarketCatalogue", {
            "filter": flt, "maxResults": max_results, "sort": "MAXIMUM_TRADED",
            "marketProjection": ["EVENT", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION",
                                 "MARKET_START_TIME"],
        })
        return [parse_catalogue(c) for c in res]

    def books(self, markets: list[BFMarket]) -> list[BFMarket]:
        by_id = {m.market_id: m for m in markets}
        ids = list(by_id)
        for i in range(0, len(ids), 10):          # stay under the request weight limit
            res = self._call("listMarketBook", {
                "marketIds": ids[i:i + 10],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
            })
            for b in res:
                apply_book(by_id[b["marketId"]], b)
        return markets


# --------------------------------------------------------------------------
# Historical data (historicdata.betfair.com, "stream" JSON-lines files)
# --------------------------------------------------------------------------

def read_stream_file(path: Path) -> dict:
    """Replay one market file. Returns the final definition plus a time series
    of last traded prices per runner: {"definition": {...}, "ltp": {id: [(ms, price)]}}."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    definition, ltp = None, {}
    with opener(path, "rt") as f:
        for line in f:
            msg = json.loads(line)
            pt = msg.get("pt")
            for mc in msg.get("mc", []):
                if "marketDefinition" in mc:
                    definition = mc["marketDefinition"]
                for rc in mc.get("rc", []):
                    if "ltp" in rc:
                        ltp.setdefault(rc["id"], []).append((pt, float(rc["ltp"])))
    return {"definition": definition or {}, "ltp": ltp, "market_id": path.name.split(".bz2")[0]}


def stream_snapshot_rows(rec: dict, days: tuple) -> list[dict]:
    """One row per (runner, horizon): last traded price N days before the
    market's scheduled time, and whether the runner won."""
    d = rec["definition"]
    if d.get("status") != "CLOSED":
        return []
    status = {r["id"]: r.get("status") for r in d.get("runners", [])}
    if not any(s == "WINNER" for s in status.values()):
        return []
    end = _t(d.get("marketTime")) or _t(d.get("suspendTime"))
    if end is None:
        return []
    names = {r["id"]: r.get("name", str(r["id"])) for r in d.get("runners", [])}
    rows = []
    for days_before in days:
        cutoff = int(end.timestamp() * 1000) - days_before * 86_400_000
        for rid, series in rec["ltp"].items():
            if status.get(rid) not in ("WINNER", "LOSER"):
                continue
            past = [p for t, p in series if t <= cutoff]
            if not past or past[-1] <= 1.0:
                continue
            rows.append({
                "market_id": rec["market_id"], "event": d.get("eventName", ""),
                "market": d.get("name", ""), "event_type": d.get("eventTypeId"),
                "runner": names.get(rid), "end": end, "days_before": days_before,
                "odds": past[-1], "won": int(status[rid] == "WINNER"),
            })
    return rows
