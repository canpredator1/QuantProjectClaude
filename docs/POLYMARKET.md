# Prediction-market forecaster (research + paper trading)

**Status:** code complete and tested offline; waiting on network access to the
Polymarket API before the first real run.

**Legal note (Ireland):** since July 2026 Polymarket blocks Irish users from
opening accounts, depositing or opening positions after action by the
Gambling Regulatory Authority of Ireland. This project only *reads* public
market data and paper-trades. Do not use it to trade on Polymarket from
Ireland, and do not route around the geoblock. If the paper results justify
it, point the same engine at a venue licensed by the GRAI.

## What it does

```
scan open markets ──► Claude forecasts blind ──► fetch quote ──► decide ──► ledger
  (volume, 1–60 days      (rules + web search,       (after the      (edge > fee + 5¢,
   to close, no fast       prediction-market sites    forecast)       ¼ Kelly, ≤2% per
   sports/crypto)          blocked, never sees price)                 market, ≤30% open)
                                                                         │
                     settle when resolved ◄──────────────────────────────┘
                     score: Claude vs market (Brier, log loss) and paper P&L
```

## Phases

1. **Calibration study** (`pm.py history`, `pm.py calibrate`): are Polymarket
   prices honest probabilities? Tests the favourite-longshot bias on resolved
   markets at 1, 7 and 30 days before close, with fees, split by time.
2. **Paper forecasting** (`pm.py forecast`, `settle`, `score`): ~10–15
   forecasts a day. Pass bar after ~300 settled markets: Claude's Brier score
   below the market's **and** positive paper ROI after fees.
3. **Real money**: only if phase 2 passes, and only on a venue that is legal
   where you live.

## Running it

| Command | Needs |
|---|---|
| `python scripts/pm.py history --start 2023-01-01` | network: `gamma-api.polymarket.com`, `clob.polymarket.com` |
| `python scripts/pm.py calibrate` | the history download |
| `python scripts/pm.py forecast` | network + Anthropic API key (`ANTHROPIC_API_KEY`) |
| `python scripts/pm.py settle` / `score` | network |

Daily automation: `.github/workflows/polymarket-paper.yml` runs settle →
forecast → score at 07:17 UTC and commits `paper/ledger.jsonl`. It needs the
repository secret `ANTHROPIC_API_KEY` and runs from the default branch.

Cost: each forecast is one Claude Opus 5.5 call with up to 5 web searches,
roughly $0.10–0.40, so ~$2–5 a day at 15 markets.

## Design choices

- **Blind forecasts.** The price is fetched after the forecast and never shown
  to Claude; prediction-market and odds sites are blocked in its searches.
  Otherwise "beating the market" would just mean copying it.
- **Clean test.** Every forecast is about an event that has not happened yet,
  so it cannot come from memorised outcomes (the problem with backtesting an
  LLM on old news).
- **Mid-volume markets first.** Liquid enough to trade, less crowded than the
  headline markets.
- **Fees are modelled** as rate × p × (1 − p) by category (2026 schedule).
- **Auditable ledger.** Append-only JSONL committed to git with timestamps.
