# Reverse-engineering the biggest winners

Winner = top 1% forward return among S&P 500 members that day; loser = bottom 1%. Signals are measured at the close **before** the run starts. Signal values are cross-sectional ranks centred at 0 (range −0.5 … +0.5), so +0.10 means winners sat 10 percentile points above the typical stock.


## Horizon: 63 trading days (~3 months)

Discovery 2006–2015: 577 winner cases, 457 loser cases (sampled every 21 days).

### What winners looked like beforehand (vs all stocks)

| Signal | Family | Winners vs all | Losers vs all |
|---|---|---|---|
| `idiovol_126` | risk | +0.297 | +0.336 |
| `idiovol_63` | risk | +0.289 | +0.328 |
| `parkinson_63` | risk | +0.281 | +0.332 |
| `vol_126` | risk | +0.276 | +0.320 |
| `vol_252` | risk | +0.275 | +0.317 |
| `vol_63` | risk | +0.274 | +0.313 |
| `parkinson_21` | risk | +0.272 | +0.315 |
| `downvol_63` | risk | +0.257 | +0.298 |
| `idiovol_21` | risk | +0.256 | +0.291 |
| `vol_21` | risk | +0.249 | +0.280 |
| `parkinson_5` | risk | +0.248 | +0.299 |
| `maxret_63` | risk | +0.230 | +0.257 |

**Correlation between the winner profile and the loser profile across all 227 signals: +0.96.** 
Close to +1 means big winners and big losers looked the same before the move.

### What separates winners from losers

| Signal | Family | Winners − losers |
|---|---|---|
| `log_dollar_vol_63` | liquidity | -0.110 |
| `log_dollar_vol_5` | liquidity | -0.109 |
| `x_maxret_21__log_dollar_vol_63` | interaction | -0.109 |
| `wq_042` | worldquant | +0.105 |
| `log_dollar_vol_21` | liquidity | -0.103 |
| `x_ret_252_skip21__log_dollar_vol_63` | interaction | +0.101 |
| `log_dollar_vol_252` | liquidity | -0.099 |
| `amihud_252` | liquidity | +0.079 |
| `amihud_63` | liquidity | +0.072 |
| `amihud_21` | liquidity | +0.070 |
| `amihud_5` | liquidity | +0.064 |
| `x_ret_21__vol_63` | interaction | +0.057 |

### Biggest winners (examples)

| Period | Ticker | Start | Return |
|---|---|---|---|
| 2006–2015 | AIG | 2009-03-06 | +350% |
| 2006–2015 | HIG | 2009-03-06 | +330% |
| 2006–2015 | PFG | 2009-03-06 | +288% |
| 2006–2015 | THC | 2009-03-06 | +286% |
| 2006–2015 | TGNA | 2009-07-07 | +284% |
| 2006–2015 | LNC | 2009-03-06 | +277% |
| 2006–2015 | JNY-201404 | 2009-02-04 | +185% |
| 2006–2015 | BAC | 2009-02-04 | +179% |
| 2006–2015 | MBI | 2008-06-05 | +173% |
| 2006–2015 | CBRE | 2009-02-04 | +144% |
| 2016–2026 | SNDK | 2025-12-15 | +275% |
| 2016–2026 | DELL | 2026-03-18 | +187% |
| 2016–2026 | INTC | 2026-03-18 | +175% |
| 2016–2026 | STX | 2026-03-18 | +169% |
| 2016–2026 | AMD | 2026-03-18 | +163% |
| 2016–2026 | MRNA | 2026-06-17 | +147% |
| 2016–2026 | FANG | 2020-11-06 | +140% |
| 2016–2026 | PSKY | 2020-12-08 | +138% |
| 2016–2026 | BBWI | 2020-05-11 | +135% |
| 2016–2026 | DISCK | 2020-12-08 | +128% |

### Out-of-sample test (model trained on 2006–2015, scored 2016–2026)

Base rate: 1% of stocks are winners and 1% are losers by definition. Average 63-day return of all stocks: +3.4% (median +3.0%).

| Model's picks | Cases | Became winners | Became losers | Mean return | Median return |
|---|---|---|---|---|---|
| Top 1% of picks | 630 | 8.3% | 3.3% | +6.4% | +2.9% |
| Next 4% | 2,520 | 5.3% | 4.0% | +5.0% | +2.9% |
| Next 5% | 3,121 | 3.0% | 2.4% | +3.8% | +2.3% |
| Bottom 90% | 55,801 | 0.6% | 0.5% | +3.2% | +3.1% |

Most important signals in the model: `idiovol_126` (7%), `idiovol_63` (3%), `vol_252` (2%), `corrmkt_252` (2%), `pv_corr_63` (1%), `wq_042` (1%), `parkinson_63` (1%), `skew_252` (1%), `kurt_252` (1%), `x_ret_252_skip21__log_dollar_vol_63` (1%)


## Horizon: 252 trading days (~12 months)

Discovery 2006–2015: 577 winner cases, 457 loser cases (sampled every 21 days).

### What winners looked like beforehand (vs all stocks)

| Signal | Family | Winners vs all | Losers vs all |
|---|---|---|---|
| `idiovol_126` | risk | +0.300 | +0.315 |
| `idiovol_63` | risk | +0.285 | +0.311 |
| `parkinson_63` | risk | +0.281 | +0.314 |
| `vol_252` | risk | +0.277 | +0.292 |
| `vol_126` | risk | +0.277 | +0.300 |
| `parkinson_21` | risk | +0.270 | +0.304 |
| `vol_63` | risk | +0.269 | +0.295 |
| `downvol_63` | risk | +0.264 | +0.289 |
| `parkinson_5` | risk | +0.253 | +0.286 |
| `idiovol_21` | risk | +0.252 | +0.282 |
| `minret_63` | risk | -0.240 | -0.250 |
| `vol_21` | risk | +0.239 | +0.269 |

**Correlation between the winner profile and the loser profile across all 227 signals: +0.92.** 
Close to +1 means big winners and big losers looked the same before the move.

### What separates winners from losers

| Signal | Family | Winners − losers |
|---|---|---|
| `wq_042` | worldquant | +0.172 |
| `log_dollar_vol_63` | liquidity | -0.157 |
| `log_dollar_vol_21` | liquidity | -0.153 |
| `log_dollar_vol_252` | liquidity | -0.146 |
| `log_dollar_vol_5` | liquidity | -0.145 |
| `x_ret_252_skip21__log_dollar_vol_63` | interaction | +0.136 |
| `amihud_252` | liquidity | +0.135 |
| `amihud_63` | liquidity | +0.129 |
| `x_maxret_21__log_dollar_vol_63` | interaction | -0.126 |
| `amihud_21` | liquidity | +0.123 |
| `amihud_5` | liquidity | +0.104 |
| `ret_252` | momentum | +0.088 |

### Biggest winners (examples)

| Period | Ticker | Start | Return |
|---|---|---|---|
| 2006–2015 | HIG | 2009-03-06 | +681% |
| 2006–2015 | IP | 2009-03-06 | +528% |
| 2006–2015 | TXT | 2009-03-06 | +499% |
| 2006–2015 | THC | 2009-03-06 | +498% |
| 2006–2015 | JBL | 2009-03-06 | +422% |
| 2006–2015 | XL-201809 | 2009-01-05 | +420% |
| 2006–2015 | FITB | 2009-04-06 | +392% |
| 2006–2015 | FCX | 2008-12-03 | +368% |
| 2006–2015 | LNC | 2009-04-06 | +359% |
| 2006–2015 | WYND | 2009-04-06 | +356% |
| 2016–2026 | WDC | 2025-06-16 | +1129% |
| 2016–2026 | MU | 2025-06-16 | +774% |
| 2016–2026 | STX | 2025-06-16 | +728% |
| 2016–2026 | MRNA | 2025-09-16 | +536% |
| 2016–2026 | INTC | 2025-06-16 | +487% |
| 2016–2026 | BBWI | 2020-05-11 | +478% |
| 2016–2026 | TER | 2025-04-15 | +440% |
| 2016–2026 | FCX | 2020-03-11 | +423% |
| 2016–2026 | PSKY | 2020-03-11 | +394% |
| 2016–2026 | DELL | 2025-09-16 | +365% |

### Out-of-sample test (model trained on 2006–2015, scored 2016–2026)

Base rate: 1% of stocks are winners and 1% are losers by definition. Average 252-day return of all stocks: +13.3% (median +10.1%).

| Model's picks | Cases | Became winners | Became losers | Mean return | Median return |
|---|---|---|---|---|---|
| Top 1% of picks | 585 | 5.1% | 3.6% | +15.4% | +3.3% |
| Next 4% | 2,328 | 4.0% | 3.0% | +16.2% | +4.6% |
| Next 5% | 2,838 | 3.1% | 2.4% | +15.2% | +7.6% |
| Bottom 90% | 51,209 | 0.7% | 0.6% | +13.0% | +10.3% |

Most important signals in the model: `parkinson_63` (8%), `idiovol_126` (5%), `amihud_252` (2%), `wq_042` (2%), `corrmkt_252` (1%), `x_ret_252_skip21__log_dollar_vol_63` (1%), `dv_ratio_21_252` (1%), `days_since_low_252` (1%), `minret_63` (1%), `x_overnight_5__intraday_5` (1%)

