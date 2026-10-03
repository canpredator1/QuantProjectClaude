# FNSPID: notes from inspecting the dataset (October 2026)

Source: Dong, Fan & Peng, "FNSPID: A Comprehensive Financial News Dataset in
Time Series" (KDD 2024). Code: github.com/Zdong104/FNSPID_Financial_News_Dataset.
Data: huggingface.co/datasets/Zihan1004/FNSPID (not reachable from the cloud
sandbox; only the GitHub samples for AA and AAPL were inspected).

## What is in it

| Part | Content |
|---|---|
| News | ~15.7M records (HF viewer shows 13.1M–15.5M rows), Nasdaq.com articles, 1999–early 2024, sparse before ~2010 |
| News columns | Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa/Luhn/Textrank/Lexrank summaries |
| Prices | ~29.7M daily OHLCV rows for 4,775 symbols (`full_history.zip`, Yahoo-style, with Adj Close) |
| Sentiment | GPT-3.5 1–5 scores computed in the repo samples only, not for the full set |
| Size | 30+ GB |

## Problems found

1. **Timezone sign bug (lookahead).** `data_processor/preprocess.py` converts
   EST/EDT to UTC by *subtracting* 5/4 hours instead of adding them. Stored
   UTC times are 8–10 hours too early. Example: "October 13, 2021 — 03:32 am
   EDT" is stored as `2021-10-12 23:32 UTC` (correct: `2021-10-13 07:32 UTC`).
   `price_news_integrate.py` then normalises to the calendar date and joins
   on the same day's price row, so pre-market news lands on the previous
   trading day: the feature row for day t contains news from the morning of
   t+1. Any model trained on the integrated files can learn from the future.
   **Fix:** parse the raw ET string ourselves and never use the provided UTC
   column.
2. **Many articles have no time of day.** Older Nasdaq pages give only a date
   ("DEC 7, 2015"): 745 of 2,247 AA rows. Their release time relative to the
   close is unknown, so treat them as available only after that day's close
   (trade at the next open).
3. **Missing text.** `Mark = 0` rows have empty article text (scrape failed);
   in the AA sample that is the same 745 rows.
4. **Market-wide articles tagged to one stock.** e.g. "S&P futures slip ahead
   of retail sales data" is filed under AA because Alcoa is mentioned in a list.
   Relevance must be measured, not assumed.
5. **Low price alignment.** The authors report news covers only ~20% of
   stock-days; the integration fills gaps with decaying sentiment, which is a
   modelling choice, not data.
6. **LLM memorisation.** All of FNSPID predates Claude's training cutoff.
   Claude has likely seen many of these articles *and what the stock did
   afterwards*, so a Claude-scored backtest on FNSPID is not a fair test of
   option 2. Only news published after the model's cutoff (a live/paper
   forward test) gives a clean answer. FNSPID is still useful for training
   and testing non-LLM text models (TF-IDF, FinBERT, older embedding models)
   with the timing fixed.

## How it fits this repo

- Symbols are lowercase tickers; they can be joined to the point-in-time S&P
  500 universe in `data/raw/` (ticker renames and recycling need care).
- Correct timing rule for our backtest (decision at close t, trade at open
  t+1): use news with ET timestamp <= 16:00 on day t; news after 16:00 ET or
  before the next open belongs to t+1's pre-open information and can only be
  traded at the t+1 open if the decision is made pre-open.
