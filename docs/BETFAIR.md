# Betfair Exchange forecaster (research + paper trading)

**Why Betfair:** it's licensed for Irish customers by the GRAI (check the register at
grai.ie), has politics and special-bets markets, and its official API allows
automated betting. This code only reads data and paper-trades; it never places bets.

## What it does

```
politics + special bets markets ─► Claude forecasts every runner blind ─► fetch prices ─► decide ─► ledger
  (≥ £5k matched, 1–120 days,        (rules + web search; betting and       (after the      (best back or lay
   2–20 runners)                      prediction-market sites blocked;       forecast)        with EV ≥ 5p per £
                                      probabilities sum to 1)                                 after 5% commission,
                                                                                              ¼ Kelly, ≤ 2% per market)
settle on WINNER/LOSER ─► score: multi-outcome Brier + log score vs the market's own odds, and paper P&L
```

- **Back** a runner when Claude thinks it's more likely than the odds imply; **lay**
  it when Claude thinks it's less likely. Both include commission on winnings.
- **Calibration study:** `bf.py calibrate <folder>` reads Betfair's downloadable
  historical files (historicdata.betfair.com) and checks whether runners at each
  implied probability win as often as the price says, i.e. whether there's a
  favourite-longshot bias to exploit.

## Setup (your side)

1. A verified Betfair account (Irish resident is fine).
2. Your **delayed application key** (free) from the Betfair developer portal. That's
   enough for research and paper trading. The live key (one-off fee, roughly
   £299–£499) is only needed to place real bets later.
3. Optional: historical data files from historicdata.betfair.com (there's a free basic tier), for the calibration study.
4. To run here: allow `api.betfair.com` and `identitysso.betfair.com` in the
   environment's network settings. For the daily GitHub Actions run, add the repository
   secrets `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, `BETFAIR_APP_KEY` and `ANTHROPIC_API_KEY`.

## Commands

| Command | What it does |
|---|---|
| `python scripts/bf.py forecast` | scan, forecast blind, paper-trade, write `paper/betfair_ledger.jsonl` |
| `python scripts/bf.py settle` | record winners and P&L for finished markets |
| `python scripts/bf.py score` | Claude vs market accuracy, paper ROI |
| `python scripts/bf.py calibrate data/betfair/` | historical calibration report |

## Pass bar before any real money

About 150 settled markets with Claude's Brier and log scores better than the market's
own odds **and** positive paper ROI after commission. Political markets settle
slowly, so this takes months. That's the honest price of a real test.
