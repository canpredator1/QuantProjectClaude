"""Claude as a forecaster.

Claude sees the question, the resolution rules, the end date and today's date,
researches with web search, and returns a probability. It never sees the
market price, and prediction-market sites are blocked from its searches, so
the forecast is independent of the crowd it will be compared against.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic

from .config import PM, PMConfig
from .polymarket import Market

PREDICTION_MARKET_DOMAINS = [
    "polymarket.com", "kalshi.com", "manifold.markets", "metaculus.com",
    "predictit.org", "betfair.com", "smarkets.com", "oddschecker.com",
]

SYSTEM = """You are a careful, calibrated forecaster. You estimate the probability \
that a question resolves YES under its exact written rules.

How to work:
- Read the resolution rules closely. Resolution depends on what the rules say, \
not on what the headline suggests. Note edge cases (deadlines, time zones, \
which source decides, what happens if the event is ambiguous).
- Use web search to find the latest relevant facts as of today. Prefer primary \
and reputable sources. Do not look for betting odds or prediction-market prices.
- Start from a base rate for this kind of event, then adjust for the specific \
evidence. Consider how much time is left before the deadline.
- Be calibrated: 0.5 means genuinely unsure, 0.95 means you would be surprised \
to be wrong one time in twenty. Avoid 0 and 1.

Finish with a JSON object in a ```json code block and nothing after it:
{"probability": <number between 0.01 and 0.99>,
 "reasoning": "<3-6 sentences: base rate, key evidence, main uncertainty>",
 "key_facts": ["<fact with date and source>", ...],
 "rules_risk": "<anything in the rules that could make a correct-seeming answer resolve the other way, or 'none'>"}"""


@dataclass
class Forecast:
    market_id: str
    probability: float
    reasoning: str
    key_facts: list
    rules_risk: str
    model: str
    created_at: str
    input_tokens: int
    output_tokens: int
    searches: int
    refused: bool = False


def build_prompt(m: Market, now: datetime) -> str:
    end = m.end.strftime("%Y-%m-%d %H:%M UTC") if m.end else "unknown"
    rules = m.rules.strip() or "(no additional rules given)"
    return (f"Today is {now:%Y-%m-%d %H:%M} UTC.\n\n"
            f"Question: {m.question}\n"
            f"Outcomes: {', '.join(map(str, m.outcomes))} (estimate the probability of "
            f"'{m.outcomes[m.yes_index] if m.outcomes else 'Yes'}')\n"
            f"Market closes: {end}\n\n"
            f"Resolution rules:\n{rules}")


def parse_forecast_json(text: str) -> dict:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = blocks or re.findall(r"(\{[^{}]*\"probability\"[^{}]*\})", text, flags=re.S)
    for raw in reversed(candidates):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        p = float(d.get("probability"))
        if 0.0 < p < 1.0:
            d["probability"] = min(max(p, 0.01), 0.99)
            return d
    raise ValueError("no valid forecast JSON in response")


class ClaudeForecaster:
    def __init__(self, cfg: PMConfig = PM, client: anthropic.Anthropic | None = None):
        self.cfg = cfg
        self.client = client or anthropic.Anthropic()

    def _tools(self):
        if not self.cfg.use_web_search:
            return []
        return [{"type": "web_search_20260209", "name": "web_search",
                 "max_uses": self.cfg.max_searches,
                 "blocked_domains": PREDICTION_MARKET_DOMAINS}]

    def forecast(self, m: Market, now: datetime | None = None) -> Forecast:
        now = now or datetime.now(timezone.utc)
        messages = [{"role": "user", "content": build_prompt(m, now)}]
        in_tok = out_tok = searches = 0
        response = None
        for _ in range(5):                       # resume server-tool pauses
            response = self.client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=16000,
                system=SYSTEM,
                messages=messages,
                tools=self._tools(),
                output_config={"effort": self.cfg.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            in_tok += response.usage.input_tokens
            out_tok += response.usage.output_tokens
            searches += sum(1 for b in response.content if b.type == "server_tool_use")
            if response.stop_reason != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": response.content}]

        if response.stop_reason == "refusal":
            return Forecast(m.id, float("nan"), "refused", [], "", response.model,
                            now.isoformat(), in_tok, out_tok, searches, refused=True)
        text = "\n".join(b.text for b in response.content if b.type == "text")
        d = parse_forecast_json(text)
        return Forecast(
            market_id=m.id, probability=d["probability"],
            reasoning=str(d.get("reasoning", "")), key_facts=list(d.get("key_facts", [])),
            rules_risk=str(d.get("rules_risk", "")), model=response.model,
            created_at=now.isoformat(), input_tokens=in_tok, output_tokens=out_tok,
            searches=searches,
        )
