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


RUNNER_SYSTEM = """You are a careful, calibrated forecaster. You estimate, for every \
runner (possible outcome) in a market, the probability that it is the one that \
wins under the market's exact written rules.

How to work:
- Read the rules closely: what counts as winning, the deadline, which source \
decides, and what happens to the market if the event is delayed or void.
- Use web search to find the latest relevant facts as of today. Prefer primary \
and reputable sources. Do not look for betting odds or prediction-market prices.
- Start from base rates, then adjust for the specific evidence.
- Be calibrated. Probabilities across all runners must add up to 1. Give every \
runner a probability of at least 0.005; reserve tiny numbers for genuine long shots.

Finish with a JSON object in a ```json code block and nothing after it:
{"probabilities": {"<runner name exactly as listed>": <number>, ...},
 "reasoning": "<3-6 sentences: base rates, key evidence, main uncertainty>",
 "key_facts": ["<fact with date and source>", ...],
 "rules_risk": "<anything in the rules that could change who wins, or 'none'>"}"""


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


def parse_runner_json(text: str, runners: list[str]) -> dict:
    """Extract per-runner probabilities, match names, floor and renormalise."""
    blocks = re.findall(r"```json\s*(\{.*\})\s*```", text, flags=re.S)
    for raw in reversed(blocks):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        given = {str(k).strip().lower(): float(v) for k, v in (d.get("probabilities") or {}).items()}
        probs = {name: max(given.get(name.strip().lower(), 0.0), 0.005) for name in runners}
        total = sum(probs.values())
        if total <= 0 or sum(1 for n in runners if n.strip().lower() in given) < len(runners) / 2:
            continue
        d["probabilities"] = {k: v / total for k, v in probs.items()}
        return d
    raise ValueError("no valid runner-probability JSON in response")


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

    def _ask(self, system: str, prompt: str):
        """One forecasting conversation; resumes server-tool pauses.
        Returns (final response, input tokens, output tokens, searches)."""
        messages = [{"role": "user", "content": prompt}]
        in_tok = out_tok = searches = 0
        response = None
        for _ in range(5):
            response = self.client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=16000,
                system=system,
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
        return response, in_tok, out_tok, searches

    def forecast(self, m: Market, now: datetime | None = None) -> Forecast:
        now = now or datetime.now(timezone.utc)
        response, in_tok, out_tok, searches = self._ask(SYSTEM, build_prompt(m, now))
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

    def forecast_runners(self, question: str, rules: str, runners: list[str],
                         close: datetime | None, now: datetime | None = None) -> dict:
        """Probability for every runner of a multi-outcome market (prices never shown)."""
        now = now or datetime.now(timezone.utc)
        close_s = close.strftime("%Y-%m-%d %H:%M UTC") if close else "unknown"
        prompt = (f"Today is {now:%Y-%m-%d %H:%M} UTC.\n\nMarket: {question}\n"
                  f"Scheduled close: {close_s}\n\nRunners:\n" +
                  "\n".join(f"- {r}" for r in runners) +
                  f"\n\nRules:\n{rules.strip() or '(no rules given)'}")
        response, in_tok, out_tok, searches = self._ask(RUNNER_SYSTEM, prompt)
        base = {"model": response.model, "created_at": now.isoformat(),
                "input_tokens": in_tok, "output_tokens": out_tok, "searches": searches}
        if response.stop_reason == "refusal":
            return {**base, "refused": True, "probabilities": {}}
        text = "\n".join(b.text for b in response.content if b.type == "text")
        d = parse_runner_json(text, runners)
        return {**base, "refused": False, "probabilities": d["probabilities"],
                "reasoning": str(d.get("reasoning", "")), "key_facts": list(d.get("key_facts", [])),
                "rules_risk": str(d.get("rules_risk", ""))}
