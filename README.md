# QuantProjectClaude — Signal Factory

Generate hundreds of trading-signal ideas, test every one of them honestly,
keep the few that survive, combine them with LightGBM, and backtest the result
with trading costs on the point-in-time S&P 500.

**Results: [`reports/RESULTS.md`](reports/RESULTS.md)**

> Also in this repo: a prediction-market forecaster (Claude forecasts Polymarket
> questions blind, paper-trades, and is scored against the market). See
> [`docs/POLYMARKET.md`](docs/POLYMARKET.md). For Irish users the same engine runs on
> Betfair Exchange (licensed in Ireland): [`docs/BETFAIR.md`](docs/BETFAIR.md).

## What we found (October 2026 run)

1. **The signals are real.** 46 of 227 ideas cleared a strict t > 3 bar on
   2006–2015 data; on a shuffled (pure luck) target only 1 did. 41 of the 46
   still pointed the same way in 2016–2026, at about half their original
   strength. Winners are short-term reversal conditioned on volume, volatility
   and size, plus several WorldQuant-style price-volume formulas.
2. **Prediction stays real out of sample.** Every model's daily rank-IC
   against next-day returns was positive in almost every year, 2016–2026.
3. **But the edge is too small and too short-lived to pay trading costs.**
   Trading the next-day signal needs ~2.3x portfolio turnover per day; at
   5 bps per trade that is ~29% a year of costs against a gross edge far
   smaller than that. All 18 next-day configurations lost money net of costs in
   validation.
4. **Holding longer (5-day horizon) helped in validation, not in the holdout.**
   The frozen choice made +2.2%/yr net (Sharpe 0.24) in 2016–2019 and
   −3.8%/yr (Sharpe −0.20) on the untouched 2020–2026 holdout. The long-only
   version roughly matched the equal-weight index (+0.3%/yr). Deflated Sharpe
   ratio: 0.00, i.e. no evidence of skill after accounting for 36 trials.
5. **The hindsight trap, visible in the data.** Some configurations that were
   poor in validation did well in the holdout (`h5_lgbm_all|hl10|band20`:
   Sharpe −0.16 then +0.81). Picking them now would be choosing with
   knowledge of the answer — exactly what the locked holdout exists to stop.

Bottom line: price-and-volume signals on large US stocks predict next-day
returns slightly better than chance, which matches the published research,
and that is not enough to profit after realistic costs. A tradable edge needs
cheaper execution, slower signals, or information that isn't in the price
history (news, fundamentals, alternative data).

## How it works

```
prices (free, point-in-time S&P 500)
   │
   ├─ 1. signals.py   227 formulaic signals in 7 families
   │                  (momentum/reversal, risk, technical, liquidity,
   │                   seasonal, WorldQuant-style, interactions)
   ├─ 2. factory.py   daily rank-IC of every signal vs next-day return,
   │                  plus the same test against a shuffled target (pure luck)
   ├─ 3. combine.py   for each year Y, using only data before Y:
   │                    keep signals with Newey-West |t| > 3, drop duplicates,
   │                    combine by equal-weight vote or LightGBM
   ├─ 4. backtest.py  long top 10% / short bottom 10% (and long-only),
   │                  trade at the next open, 5 bps cost per trade
   └─ 5. run_pipeline.py
          --stage validation  choose a configuration on 2016-2019 only
          --stage final       report it on the untouched 2020-2026 holdout
```

### Guards against fooling ourselves

- **No look-ahead:** `tests/test_no_lookahead.py` corrupts all future data and
  checks every signal's past values are unchanged.
- **Execution delay:** signals use the close of day t; trades happen at the
  open of t+1 and are closed at the open of t+2.
- **Multiple testing:** a t-stat hurdle of 3 (not 2), Newey-West standard
  errors, a shuffled-target placebo run, and a deflated Sharpe ratio.
- **Walk-forward:** every prediction is made by a model that never saw that
  year. A 5-day embargo separates training and test data.
- **Locked holdout:** the configuration is frozen on 2016-2019 before
  2020-2026 is evaluated; the script refuses to run the final stage first.
- **Point-in-time universe:** only stocks that were in the S&P 500 on that day,
  including companies that later left the index (where free data exists).

## Run it

```bash
pip install -r requirements.txt
scripts/fetch_data.sh                          # ~160 MB from GitHub
python scripts/run_pipeline.py --stage validation
python scripts/run_pipeline.py --stage final
python -m pytest tests
```

The first run computes all signals (~20 minutes on 4 CPUs) and caches them in
`data/cache/`. Walk-forward model training takes roughly another 30 minutes.

## Data

[Johnbrick123/sp500-data](https://github.com/Johnbrick123/sp500-data): daily
OHLCV for S&P 500 members since 1995 with point-in-time membership, built from
Yahoo Finance and Tiingo. Free data is missing some companies that left the
index (about a quarter of members in 2006, almost none after 2020), so early
results are somewhat flattered. See the caveats in the results.

## Layout

```
src/alphafactory/   config, data, signals, evaluate, factory, combine, backtest
scripts/            fetch_data.sh, run_pipeline.py
tests/              no-lookahead and mechanics tests
reports/            RESULTS.md, CSVs, figures (generated)
```

Research code, not investment advice.
