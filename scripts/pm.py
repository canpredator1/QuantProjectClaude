"""Prediction-market research CLI (read-only data + paper trading).

  python scripts/pm.py history  [--start 2023-01-01] [--limit N]   download resolved markets
  python scripts/pm.py calibrate                                    long-shot bias report
  python scripts/pm.py forecast [--dry-run]                         scan, forecast, paper-trade
  python scripts/pm.py settle                                       settle finished markets
  python scripts/pm.py score                                        Claude vs market scorecard

Needs network access to gamma-api.polymarket.com and clob.polymarket.com, and
Anthropic API credentials for `forecast`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predmkt.calibration import calibration_table, contracts, strategy_test  # noqa: E402
from predmkt.config import PM  # noqa: E402
from predmkt.history import build_history  # noqa: E402
from predmkt.paper import run_once, score, settle  # noqa: E402
from predmkt.polymarket import PolymarketClient  # noqa: E402


def cmd_history(a):
    df = build_history(PolymarketClient(), start=a.start, limit_markets=a.limit)
    print(f"{df.market_id.nunique():,} markets, {len(df):,} snapshots -> "
          f"{PM.data_dir / 'resolved_snapshots.parquet'}")


def cmd_calibrate(a):
    snap = pd.read_parquet(PM.data_dir / "resolved_snapshots.parquet")
    PM.report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Polymarket calibration (favourite-longshot test)\n",
             f"{snap.market_id.nunique():,} resolved binary markets, "
             f"{snap.end.min():%Y-%m-%d} to {snap.end.max():%Y-%m-%d}. Each market counts twice "
             "(YES at p and NO at 1-p). Returns include current taker fees.\n"]
    for d in PM.snapshot_days:
        t = calibration_table(contracts(snap[snap.days_before == d]))
        lines.append(f"\n## Price {d} day(s) before close\n")
        lines.append("| Price bucket | Contracts | Avg price | Win rate | Return per $1 | ± s.e. |")
        lines.append("|---|---|---|---|---|---|")
        for b, r in t.iterrows():
            lines.append(f"| {b} | {int(r.contracts):,} | {r.avg_price:.3f} | {r.win_rate:.3f} | "
                         f"{r['return_per_$']:+.3f} | "
                         f"{r.se:.3f} |")
    lines.append("\n## Rule test: buy favourites priced 0.90–0.98, 7 days out\n")
    res = strategy_test(snap, 0.90, 0.98, 7)
    lines.append("| Period | Bets | Win rate | Return per $1 | ± s.e. | Per day held |")
    lines.append("|---|---|---|---|---|---|")
    for k, r in res.items():
        lines.append(f"| {k} | {r['bets']:,} | {r['win_rate']:.3f} | {r['return_per_$']:+.4f} | "
                     f"{r['se']:.4f} | {r['return_per_$_per_day']:+.5f} |")
    out = PM.report_dir / "CALIBRATION.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


def cmd_forecast(a):
    from predmkt.forecaster import ClaudeForecaster
    rows = run_once(PolymarketClient(), ClaudeForecaster(), dry_run=a.dry_run)
    for r in rows:
        print(f"{r['forecast']['probability']:.2f} vs mid {r['mid']}  {r['decision']:<35s} "
              f"{r['question'][:70]}")


def cmd_settle(a):
    print(f"settled {settle(PolymarketClient())} positions")


def cmd_score(a):
    print(json.dumps(score(), indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history")
    h.add_argument("--start", default="2023-01-01")
    h.add_argument("--limit", type=int, default=None)
    sub.add_parser("calibrate")
    f = sub.add_parser("forecast")
    f.add_argument("--dry-run", action="store_true")
    sub.add_parser("settle")
    sub.add_parser("score")
    a = ap.parse_args()
    {"history": cmd_history, "calibrate": cmd_calibrate, "forecast": cmd_forecast,
     "settle": cmd_settle, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
