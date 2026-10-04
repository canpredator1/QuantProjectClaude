"""Betfair Exchange research CLI (read-only data + paper trading; never bets).

  python scripts/bf.py forecast                     scan politics/specials, forecast, paper-trade
  python scripts/bf.py settle                       settle finished markets
  python scripts/bf.py score                        Claude vs market scorecard
  python scripts/bf.py calibrate <folder>           calibration from historicdata.betfair.com files

Needs BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY (the free delayed key is enough),
network access to *.betfair.com, and Anthropic credentials for `forecast`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predmkt import bf_paper  # noqa: E402
from predmkt.config import PM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("forecast")
    sub.add_parser("settle")
    sub.add_parser("score")
    c = sub.add_parser("calibrate")
    c.add_argument("folder")
    a = ap.parse_args()

    if a.cmd == "calibrate":
        snap = bf_paper.build_bf_history(Path(a.folder))
        if snap.empty:
            sys.exit("no settled markets found in that folder")
        PM.report_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Betfair calibration (back every runner at its last traded price)\n",
                 f"{snap.market_id.nunique():,} settled markets. Returns after 5% commission.\n"]
        for d in sorted(snap.days_before.unique()):
            t = bf_paper.bf_calibration(snap[snap.days_before == d])
            lines += [f"\n## {d} day(s) before close\n",
                      "| Implied prob | Runners | Avg implied | Win rate | Back return per £1 | ± s.e. |",
                      "|---|---|---|---|---|---|"]
            for b, r in t.iterrows():
                lines.append(f"| {b} | {int(r.runners):,} | {r.avg_implied:.3f} | {r.win_rate:.3f} | "
                             f"{r['back_return_per_£']:+.3f} | {r.se:.3f} |")
        out = PM.report_dir.parent / "betfair" / "CALIBRATION.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(out.read_text())
        return

    from predmkt.betfair import BetfairClient
    client = BetfairClient()
    if a.cmd == "forecast":
        from predmkt.forecaster import ClaudeForecaster
        for r in bf_paper.run_once(client, ClaudeForecaster()):
            print(f"{r['decision']:<50s} {r['question'][:70]}")
    elif a.cmd == "settle":
        print(f"settled {bf_paper.settle(client)} markets")
    else:
        print(json.dumps(bf_paper.score(), indent=2))


if __name__ == "__main__":
    main()
