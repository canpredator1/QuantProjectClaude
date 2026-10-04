"""Reverse-engineer the biggest winners: what did they look like beforehand?

1. Label every stock-day by its forward return over H trading days
   (bought at the next open, valued at the close H days later).
   "Winner" = top 1% of the S&P 500 that day; "loser" = bottom 1%.
2. Describe: compare the 227 signal ranks of winners, losers and everyone,
   measured the day BEFORE the run starts (2006-2015).
3. Test: train a model on 2006-2015 to spot future winners, then check on
   2016-2026 whether its picks really are winners more often than chance,
   and what they return on average (hit rate alone hides the losers).

Usage: python scripts/winners_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphafactory.combine import Store  # noqa: E402
from alphafactory.config import CFG  # noqa: E402
from alphafactory.data import build_panels  # noqa: E402

HORIZONS = [63, 252]          # one quarter, one year
TAIL = 0.01                   # top / bottom 1%
SPLIT = pd.Timestamp("2016-01-01")
STRIDE = 21                   # sample every 21st day to limit overlap
OUT = CFG.report_dir / "winners"


def forward_returns(panels, store, h):
    o, c = panels.open, panels.close
    fwd = (c.shift(-h) / o.shift(-1) - 1.0)
    di = c.index.get_indexer(store.rows.date)
    ti = c.columns.get_indexer(store.rows.ticker)
    return fwd.to_numpy()[di, ti]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = Store(CFG)
    panels = build_panels(CFG)
    X = np.asarray(store.X)
    names = np.array(store.names)
    family = store.meta.set_index("signal").family
    rows = store.rows[["date", "ticker"]].copy()
    dates = np.sort(rows.date.unique())
    keep_days = set(dates[::STRIDE])
    sampled = rows.date.isin(keep_days).to_numpy()
    lines = ["# Reverse-engineering the biggest winners\n",
             "Winner = top 1% forward return among S&P 500 members that day; loser = bottom 1%. "
             "Signals are measured at the close **before** the run starts. Signal values are "
             "cross-sectional ranks centred at 0 (range −0.5 … +0.5), so +0.10 means winners sat "
             "10 percentile points above the typical stock.\n"]

    for h in HORIZONS:
        fwd = forward_returns(panels, store, h)
        rows["fwd"] = fwd
        pct = rows.groupby("date").fwd.rank(pct=True).to_numpy()
        valid = ~np.isnan(fwd) & sampled
        win = valid & (pct > 1 - TAIL)
        lose = valid & (pct <= TAIL)
        disc = valid & (rows.date < SPLIT).to_numpy()
        test = valid & (rows.date >= SPLIT).to_numpy()

        # ---------- 1. describe (discovery period only) ----------
        with np.errstate(invalid="ignore"):
            mu_all = np.nanmean(X[disc], axis=0)
            mu_win = np.nanmean(X[disc & win], axis=0)
            mu_lose = np.nanmean(X[disc & lose], axis=0)
        prof = pd.DataFrame({"family": family.reindex(names).to_numpy(),
                             "winners_vs_all": mu_win - mu_all,
                             "losers_vs_all": mu_lose - mu_all}, index=names)
        prof["winners_vs_losers"] = prof.winners_vs_all - prof.losers_vs_all
        prof.to_csv(OUT / f"profile_h{h}.csv")
        top_win = prof.reindex(prof.winners_vs_all.abs().sort_values(ascending=False).index).head(12)
        top_sep = prof.reindex(prof.winners_vs_losers.abs().sort_values(ascending=False).index).head(12)
        corr = np.corrcoef(prof.winners_vs_all, prof.losers_vs_all)[0, 1]

        ex = rows[win & disc].nlargest(400, "fwd").drop_duplicates("ticker").head(10)
        ex_t = rows[win & test].nlargest(400, "fwd").drop_duplicates("ticker").head(10)

        # ---------- 2. test: can the pattern pick winners? ----------
        y = win.astype(int)
        params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                      min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.7,
                      bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=7)
        # embargo: training labels must end before the test period starts
        train = disc & (rows.date < SPLIT - pd.Timedelta(days=int(h * 1.5))).to_numpy()
        model = lgb.train(params, lgb.Dataset(X[train], label=y[train]), num_boost_round=300)
        p = model.predict(X[test])
        t = rows[test].assign(p=p, win=win[test], lose=lose[test])
        t["bucket"] = t.groupby("date").p.rank(pct=True)
        bucket_stats = []
        for lo, hi, label in [(0.99, 1.01, "Top 1% of picks"), (0.95, 0.99, "Next 4%"),
                              (0.9, 0.95, "Next 5%"), (0.0, 0.9, "Bottom 90%")]:
            g = t[(t.bucket > lo) & (t.bucket <= hi)]
            bucket_stats.append({"bucket": label, "n": len(g),
                                 "winner_rate": g.win.mean(), "loser_rate": g.lose.mean(),
                                 "mean_fwd": g.fwd.mean(), "median_fwd": g.fwd.median()})
        bs = pd.DataFrame(bucket_stats)
        all_mean, all_med = t.fwd.mean(), t.fwd.median()
        imp = pd.Series(model.feature_importance("gain"), index=names).sort_values(ascending=False)
        imp = (imp / imp.sum()).head(10)

        # ---------- write ----------
        L = lines.append
        L(f"\n## Horizon: {h} trading days (~{h // 21} months)\n")
        L(f"Discovery 2006–2015: {int((disc & win).sum()):,} winner cases, "
          f"{int((disc & lose).sum()):,} loser cases (sampled every {STRIDE} days).\n")
        L("### What winners looked like beforehand (vs all stocks)\n")
        L("| Signal | Family | Winners vs all | Losers vs all |")
        L("|---|---|---|---|")
        for s, r in top_win.iterrows():
            L(f"| `{s}` | {r.family} | {r.winners_vs_all:+.3f} | {r.losers_vs_all:+.3f} |")
        L(f"\n**Correlation between the winner profile and the loser profile across all "
          f"227 signals: {corr:+.2f}.** ")
        L("Close to +1 means big winners and big losers looked the same before the move.\n")
        L("### What separates winners from losers\n")
        L("| Signal | Family | Winners − losers |")
        L("|---|---|---|")
        for s, r in top_sep.iterrows():
            L(f"| `{s}` | {r.family} | {r.winners_vs_losers:+.3f} |")
        L("\n### Biggest winners (examples)\n")
        L("| Period | Ticker | Start | Return |")
        L("|---|---|---|---|")
        for lab, e in [("2006–2015", ex), ("2016–2026", ex_t)]:
            for _, r in e.iterrows():
                L(f"| {lab} | {r.ticker} | {r.date.date()} | {r.fwd * 100:+.0f}% |")
        L("\n### Out-of-sample test (model trained on 2006–2015, scored 2016–2026)\n")
        L(f"Base rate: 1% of stocks are winners and 1% are losers by definition. "
          f"Average {h}-day return of all stocks: {all_mean * 100:+.1f}% "
          f"(median {all_med * 100:+.1f}%).\n")
        L("| Model's picks | Cases | Became winners | Became losers | Mean return | Median return |")
        L("|---|---|---|---|---|---|")
        for _, r in bs.iterrows():
            L(f"| {r.bucket} | {r.n:,} | {r.winner_rate * 100:.1f}% | {r.loser_rate * 100:.1f}% | "
              f"{r.mean_fwd * 100:+.1f}% | {r.median_fwd * 100:+.1f}% |")
        L("\nMost important signals in the model: " +
          ", ".join(f"`{k}` ({v * 100:.0f}%)" for k, v in imp.items()) + "\n")
        print(f"h={h} done: corr(win,lose profiles)={corr:+.2f}")
        print(bs.to_string(index=False))

    (OUT / "WINNERS.md").write_text("\n".join(lines) + "\n")
    print((OUT / "WINNERS.md").read_text())


if __name__ == "__main__":
    main()
