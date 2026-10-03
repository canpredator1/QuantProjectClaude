#!/usr/bin/env bash
# Download the free, point-in-time S&P 500 dataset published by
# github.com/Johnbrick123/sp500-data (refreshed nightly from Yahoo + Tiingo).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DIR"
BASE="https://github.com/Johnbrick123/sp500-data/releases/download/data"
for f in prices.parquet membership_intervals.parquet recycled_tickers.csv; do
  echo "downloading $f"
  curl -fSL --retry 4 -o "$DIR/$f" "$BASE/$f"
done
sha256sum "$DIR/prices.parquet"
