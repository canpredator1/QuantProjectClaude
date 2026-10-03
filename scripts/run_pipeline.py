"""End-to-end: signals -> walk-forward models -> portfolios -> report.

Two stages, run in this order and only once each:

  python scripts/run_pipeline.py --stage validation
      Scores every configuration on 2016-2019 only and freezes the best one
      into reports/chosen_config.json. Nothing from 2020+ is printed.

  python scripts/run_pipeline.py --stage final
      Reports the frozen configuration (and, for transparency, every other
      one) on the 2020+ holdout and writes reports/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphafactory.backtest import run_portfolio, summarize, to_wide  # noqa: E402
from alphafactory.combine import Store, signal_tstats, walk_forward  # noqa: E402
from alphafactory.config import CFG  # noqa: E402
from alphafactory.data import file_sha256  # noqa: E402
from alphafactory.evaluate import deflated_sharpe, perf_table, sharpe  # noqa: E402
from alphafactory.factory import run_factory  # noqa: E402

MODELS = ["simple", "lgbm_selected", "lgbm_all"]
HALFLIVES = [0, 3, 10]
BANDS = [None, 0.2]
VAL_END = CFG.holdout_start - 1


def configs():
    for m in MODELS:
        for h in HALFLIVES:
            for b in BANDS:
                yield {"model": m, "halflife": h, "exit_q": b}


def cfg_name(c):
    band = f"band{int(c['exit_q'] * 100)}" if c["exit_q"] else "noband"
    return f"{c['model']}|hl{c['halflife']}|{band}"


def get_predictions(store: Store) -> tuple[pd.DataFrame, dict]:
    path = CFG.cache_dir / "preds.parquet"
    sel_path = CFG.cache_dir / "selections.pkl"
    if path.exists() and sel_path.exists():
        return pd.read_parquet(path), pd.read_pickle(sel_path)
    preds, sels = walk_forward(store, CFG)
    preds.to_parquet(path)
    pd.to_pickle(sels, sel_path)
    return preds, sels


def evaluate_all(preds: pd.DataFrame, target: pd.DataFrame):
    out = {}
    for c in configs():
        score = to_wide(preds, c["model"])
        out[cfg_name(c)] = run_portfolio(score, target, CFG.quantile, CFG.cost_bps,
                                         halflife=c["halflife"], exit_q=c["exit_q"])
    return out


def fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def fmt(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def stage_validation(results):
    rows = []
    for name, res in results.items():
        s = summarize(res, CFG.first_test_year, VAL_END)
        rows.append({"config": name,
                     "ls_net_sharpe": s["long_short_net"]["sharpe"],
                     "ls_net_ann": s["long_short_net"]["ann_return"],
                     "ls_gross_sharpe": s["long_short_gross"]["sharpe"],
                     "turnover": s["long_short_net"]["turnover"],
                     "long_excess_sharpe": s["long_only_excess_vs_ew"]["sharpe"]})
    df = pd.DataFrame(rows).sort_values("ls_net_sharpe", ascending=False)
    best = df.iloc[0]["config"]
    chosen = [c for c in configs() if cfg_name(c) == best][0]
    CFG.report_dir.mkdir(exist_ok=True)
    (CFG.report_dir / "chosen_config.json").write_text(json.dumps(
        {"chosen": chosen, "name": best, "selected_on": f"{CFG.first_test_year}-{VAL_END}",
         "criterion": "long-short net Sharpe at 5 bps", "trials": len(df)}, indent=2))
    df.to_csv(CFG.report_dir / "validation_scores.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFrozen choice: {best}")


def spy_open_to_open(index: pd.DatetimeIndex) -> pd.Series:
    p = pd.read_parquet(CFG.raw_dir / "prices.parquet", filters=[("ticker", "==", "SPY")])
    p = p.set_index("date").sort_index()
    o = p.open * p.adj_close / p.close
    r = o.shift(-2) / o.shift(-1) - 1
    return r.reindex(index)


def stage_final(store, preds, sels, results, panels_target):
    chosen = json.loads((CFG.report_dir / "chosen_config.json").read_text())
    best = chosen["name"]
    R = CFG.report_dir
    figs = R / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    hs, ve = CFG.holdout_start, VAL_END

    # ---------------- signal-level evidence ----------------
    first = sels[CFG.first_test_year]
    disc_end = first.index  # noqa: F841 (documentation: selection table for the first window)
    meta = store.meta.set_index("signal")
    disc_dates = store.dates[store.dates < pd.Timestamp(f"{CFG.first_test_year}-01-01")]
    t_placebo = signal_tstats(store.ic_placebo.loc[disc_dates], CFG.nw_lags)
    n_sig = len(store.names)
    n_pass = int(first.passed.sum())
    n_pass_placebo = int((t_placebo.abs() > CFG.tstat_threshold).sum())
    n_pass_2 = int((first.t.abs() > 2).sum())
    n_pass_2_placebo = int((t_placebo.abs() > 2).sum())

    later = store.dates[store.dates >= pd.Timestamp(f"{CFG.first_test_year}-01-01")]
    later_ic = store.ic.loc[later].mean()
    later_t = signal_tstats(store.ic.loc[later], CFG.nw_lags)
    passed = first[first.passed].copy()
    passed["ic_after"] = later_ic.reindex(passed.index)
    passed["t_after"] = later_t.reindex(passed.index)
    passed["same_sign"] = np.sign(passed.ic_after) == passed.sign
    passed["retained"] = passed.ic_after * passed.sign / passed.ic.abs()
    passed["family"] = meta.family.reindex(passed.index)

    # long-short Sharpe of each signal before / after, sign fixed in-sample
    ls_disc = store.ls.loc[disc_dates]
    ls_after = store.ls.loc[later]
    to_mean = store.to.loc[disc_dates].mean()
    passed["sharpe_before"] = [sharpe(ls_disc[s] * passed.sign[s]) for s in passed.index]
    passed["sharpe_after_gross"] = [sharpe(ls_after[s] * passed.sign[s]) for s in passed.index]
    passed["sharpe_after_net"] = [sharpe(ls_after[s] * passed.sign[s]
                                         - store.to.loc[later, s] * CFG.cost_bps / 1e4)
                                  for s in passed.index]
    passed["turnover"] = to_mean.reindex(passed.index)
    passed = passed.sort_values("t", key=np.abs, ascending=False)
    passed.to_csv(R / "signals_discovery.csv")

    allsig = pd.DataFrame({"family": meta.family, "t_2006_2015": first.t, "ic_2006_2015": first.ic,
                           "t_placebo": t_placebo, "ic_2016_2026": later_ic, "t_2016_2026": later_t})
    allsig.sort_values("t_2006_2015", key=np.abs, ascending=False).to_csv(R / "signals_all.csv")

    fam = allsig.assign(passed=allsig.t_2006_2015.abs() > CFG.tstat_threshold).groupby("family") \
        .agg(tested=("passed", "size"), passed=("passed", "sum"))

    sel_rows = [{"year": y, "passed": int(s.passed.sum()), "kept": int(s.kept.sum()),
                 "top_kept": ", ".join(s[s.kept].t.abs().sort_values(ascending=False).index[:5])}
                for y, s in sels.items()]

    # ---------------- strategy-level evidence ----------------
    res = results[best]
    s_val = summarize(res, CFG.first_test_year, ve)
    s_hold = summarize(res, hs)
    spy = spy_open_to_open(res["ls_net"].index)

    cost_rows = []
    c0 = chosen["chosen"]
    score = to_wide(preds, c0["model"])
    for cb in CFG.cost_grid_bps:
        r = run_portfolio(score, panels_target, CFG.quantile, cb, c0["halflife"], c0["exit_q"])
        m = r["ls_net"].index.year >= hs
        cost_rows.append({"cost_bps": cb,
                          "ls_ann": r["ls_net"][m].mean() * 252, "ls_sharpe": sharpe(r["ls_net"][m]),
                          "long_excess_ann": (r["long_net"] - r["bench_ew"])[m].mean() * 252,
                          "long_excess_sharpe": sharpe((r["long_net"] - r["bench_ew"])[m])})

    all_cfg_rows = []
    for name, r in results.items():
        sv, sh = summarize(r, CFG.first_test_year, ve), summarize(r, hs)
        all_cfg_rows.append({"config": name, "val_sharpe": sv["long_short_net"]["sharpe"],
                             "hold_sharpe": sh["long_short_net"]["sharpe"],
                             "hold_ann": sh["long_short_net"]["ann_return"],
                             "hold_turnover": sh["long_short_net"]["turnover"]})
    all_cfg = pd.DataFrame(all_cfg_rows).sort_values("val_sharpe", ascending=False)
    all_cfg.to_csv(R / "all_configs.csv", index=False)

    trial_sr = [sharpe(r["ls_net"][r["ls_net"].index.year <= ve]) for r in results.values()]
    hold_net = res["ls_net"][res["ls_net"].index.year >= hs]
    dsr = deflated_sharpe(hold_net, trial_sr)

    yearly = pd.DataFrame({
        "long_short_net": res["ls_net"].groupby(res["ls_net"].index.year).sum(),
        "long_only_net": res["long_net"].groupby(res["long_net"].index.year).sum(),
        "equal_weight": res["bench_ew"].groupby(res["bench_ew"].index.year).sum(),
        "spy": spy.groupby(spy.index.year).sum(),
    })

    # model IC (prediction vs realized, cross-sectional) by year
    def daily_ic(col):
        g = preds.groupby("date")
        return g.apply(lambda d: d[col].rank().corr(d["target"].rank()))
    ic_models = pd.DataFrame({m: daily_ic(m) for m in MODELS})
    ic_year = ic_models.groupby(ic_models.index.year).mean()

    make_figures(res, spy, passed, figs, hs)

    write_results(R, chosen, best, n_sig, n_pass, n_pass_placebo, n_pass_2, n_pass_2_placebo,
                  passed, fam, sel_rows, s_val, s_hold, cost_rows, all_cfg, dsr, yearly,
                  ic_year, spy, res, hs, store)
    print((R / "RESULTS.md").read_text())


def make_figures(res, spy, passed, figs, hs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    s1, s2, s3 = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.titlecolor": ink,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    def curve(ax, r, color, label):
        w = (1 + r.fillna(0)).cumprod()
        ax.plot(w.index, w.values, color=color, lw=2, label=label)
        ax.annotate(label, (w.index[-1], w.values[-1]), xytext=(4, 0), textcoords="offset points",
                    color=ink2, fontsize=9, va="center")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    curve(ax, res["ls_net"], s1, "Long-short (net)")
    curve(ax, res["long_net"] - res["bench_ew"], s2, "Long-only minus equal-weight (net)")
    ax.axvline(pd.Timestamp(f"{hs}-01-01"), color=ink2, lw=1, ls="--")
    ax.text(pd.Timestamp(f"{hs}-01-01"), ax.get_ylim()[1], "  holdout starts", color=ink2,
            va="top", fontsize=9)
    ax.set_title("Growth of $1, net of 5 bps costs (2016–2026)", loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "equity_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    before = passed.ic.abs()
    after = passed.ic_after * passed.sign
    ax.scatter(before, after, s=36, color=s1, edgecolor="#fcfcfb", lw=1.5, label="signal")
    lim = max(before.max(), 0.001) * 1.1
    ax.plot([0, lim], [0, lim], color=ink2, lw=1, ls="--", label="no decay")
    ax.axhline(0, color=grid, lw=1)
    ax.set_xlabel("|IC| 2006–2015 (where it was discovered)")
    ax.set_ylabel("IC 2016–2026, same direction")
    ax.set_title("Signals weaken after discovery", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "signal_decay.png", dpi=130)
    plt.close(fig)


def write_results(R, chosen, best, n_sig, n_pass, n_pass_placebo, n_pass_2, n_pass_2_placebo,
                  passed, fam, sel_rows, s_val, s_hold, cost_rows, all_cfg, dsr, yearly,
                  ic_year, spy, res, hs, store):
    def perf_row(label, p):
        return (f"| {label} | {fmt_pct(p['ann_return'])} | {fmt_pct(p['ann_vol'])} | "
                f"{fmt(p['sharpe'])} | {fmt_pct(p['max_drawdown'])} | "
                f"{fmt(p.get('turnover', np.nan))} |")

    spy_hold = perf_table(spy[spy.index.year >= hs])
    spy_val = perf_table(spy[(spy.index.year >= CFG.first_test_year) & (spy.index.year < hs)])
    L = []
    a = L.append
    a("# Signal Factory: Results\n")
    a(f"_Generated by `scripts/run_pipeline.py --stage final`. Data snapshot: "
      f"`prices.parquet` sha256 `{file_sha256(CFG.raw_dir / 'prices.parquet')[:16]}…`, "
      f"{store.dates[0].date()} to {store.dates[-1].date()}._\n")
    a("## Setup\n")
    a(f"- **Universe:** point-in-time S&P 500 members, {len(store.rows):,} stock-days, "
      f"~{len(store.rows) / len(store.dates):.0f} stocks per day.")
    a("- **Timing:** signal from the close of day t, buy/sell at the open of t+1, "
      "close out at the open of t+2.")
    a(f"- **Costs:** {CFG.cost_bps:g} bps per unit of turnover (one-way), sensitivity below.")
    a(f"- **Signals tested:** {n_sig} formulaic ideas across "
      f"{len(fam)} families.")
    a(f"- **Walk-forward:** each year's models see only data before that year "
      f"(minus a {CFG.embargo_days}-day gap). 2016–{hs - 1} was used to choose the "
      f"configuration; **{hs}–2026 is the untouched holdout**.")
    a(f"- **Chosen configuration (frozen before the holdout was run):** `{best}` — picked from "
      f"{chosen['trials']} configurations by {chosen['criterion']}.\n")

    a("## 1. Signal discovery (2006–2015)\n")
    a(f"Hurdle: Newey-West |t| > {CFG.tstat_threshold:g} on the daily rank-IC.\n")
    a("| | Real target | Shuffled target (pure luck) |")
    a("|---|---|---|")
    a(f"| Signals with \\|t\\| > {CFG.tstat_threshold:g} | **{n_pass}** of {n_sig} | {n_pass_placebo} |")
    a(f"| Signals with \\|t\\| > 2 (the usual, too-easy bar) | {n_pass_2} | {n_pass_2_placebo} |\n")
    a(f"Of the {n_pass} discoveries, **{int(passed.same_sign.sum())}** kept the same direction in "
      f"2016–2026, and on average they kept **{passed.retained.median() * 100:.0f}%** "
      f"(median) of their strength.\n")
    a("### Strongest discoveries\n")
    a("| Signal | Family | IC 06–15 | t 06–15 | IC 16–26 (same dir.) | t 16–26 | "
      "L/S Sharpe 06–15 | L/S Sharpe 16–26 net | Turnover/day |")
    a("|---|---|---|---|---|---|---|---|---|")
    for s, r in passed.head(20).iterrows():
        a(f"| `{s}` | {r.family} | {r.ic:+.4f} | {r.t:+.1f} | {r.ic_after * r.sign:+.4f} | "
          f"{r.t_after * r.sign:+.1f} | {fmt(r.sharpe_before)} | {fmt(r.sharpe_after_net)} | "
          f"{fmt(r.turnover)} |")
    a("\n_Full lists: `reports/signals_discovery.csv`, `reports/signals_all.csv`._\n")
    a("### By family\n")
    a("| Family | Tested | Passed |")
    a("|---|---|---|")
    for f, r in fam.sort_values("passed", ascending=False).iterrows():
        a(f"| {f} | {r.tested} | {r.passed} |")
    a("\n![Signal decay](figures/signal_decay.png)\n")
    a("### Walk-forward selection per year\n")
    a("| Test year | Passed hurdle | Kept after de-duplication | Strongest kept |")
    a("|---|---|---|---|")
    for r in sel_rows:
        a(f"| {r['year']} | {r['passed']} | {r['kept']} | {r['top_kept']} |")

    a("\n## 2. Combined strategy\n")
    a("| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown | Turnover/day |")
    a("|---|---|---|---|---|---|")
    a(f"| **Validation {CFG.first_test_year}–{hs - 1}** | | | | | |")
    a(perf_row("Long-short, net", s_val["long_short_net"]))
    a(perf_row("Long-short, gross", s_val["long_short_gross"]))
    a(perf_row("Long-only top decile, net", s_val["long_only_net"]))
    a(perf_row("Long-only minus equal-weight, net", s_val["long_only_excess_vs_ew"]))
    a(perf_row("Equal-weight S&P 500 members", s_val["equal_weight_universe"]))
    a(perf_row("SPY (same open-to-open timing)", spy_val))
    a(f"| **Holdout {hs}–2026** | | | | | |")
    a(perf_row("Long-short, net", s_hold["long_short_net"]))
    a(perf_row("Long-short, gross", s_hold["long_short_gross"]))
    a(perf_row("Long-only top decile, net", s_hold["long_only_net"]))
    a(perf_row("Long-only minus equal-weight, net", s_hold["long_only_excess_vs_ew"]))
    a(perf_row("Equal-weight S&P 500 members", s_hold["equal_weight_universe"]))
    a(perf_row("SPY (same open-to-open timing)", spy_hold))
    a(f"\n**Deflated Sharpe ratio (holdout, {chosen['trials']} trials): {dsr:.2f}** — the "
      "probability that the holdout Sharpe is real skill rather than the luck of picking the "
      "best of many tries. Above 0.95 is convincing.\n")
    a("![Equity curves](figures/equity_curves.png)\n")
    a("### Cost sensitivity (holdout)\n")
    a("| Cost (bps) | Long-short ann. | Long-short Sharpe | Long-only excess ann. | Long-only excess Sharpe |")
    a("|---|---|---|---|---|")
    for r in cost_rows:
        a(f"| {r['cost_bps']:g} | {fmt_pct(r['ls_ann'])} | {fmt(r['ls_sharpe'])} | "
          f"{fmt_pct(r['long_excess_ann'])} | {fmt(r['long_excess_sharpe'])} |")
    a("\n### Year by year (sum of daily returns, net)\n")
    a("| Year | Long-short | Long-only | Equal-weight | SPY |")
    a("|---|---|---|---|---|")
    for y, r in yearly.iterrows():
        a(f"| {y} | {fmt_pct(r.long_short_net)} | {fmt_pct(r.long_only_net)} | "
          f"{fmt_pct(r.equal_weight)} | {fmt_pct(r.spy)} |")
    a("\n### Prediction quality by model (mean daily rank-IC)\n")
    a("| Year | " + " | ".join(ic_year.columns) + " |")
    a("|---|" + "---|" * len(ic_year.columns))
    for y, r in ic_year.iterrows():
        a(f"| {y} | " + " | ".join(f"{v:+.4f}" for v in r) + " |")
    a("\n### Every configuration (chosen on validation; holdout shown for transparency)\n")
    a("| Config | Validation Sharpe | Holdout Sharpe | Holdout ann. | Holdout turnover |")
    a("|---|---|---|---|---|")
    for _, r in all_cfg.iterrows():
        mark = " **(chosen)**" if r.config == best else ""
        a(f"| `{r.config}`{mark} | {fmt(r.val_sharpe)} | {fmt(r.hold_sharpe)} | "
          f"{fmt_pct(r.hold_ann)} | {fmt(r.hold_turnover)} |")
    a("\n## Caveats\n")
    a("- **Survivorship:** the free data is missing prices for some companies that left the "
      "index (about 25% of members in 2006, under 3% after 2020). Long-short ranking is "
      "less affected than long-only returns, but results before ~2012 are flattered.")
    a("- Stocks whose next two opens are missing (mostly takeovers) are left out of that "
      "day's portfolio — a small, unavoidable look-ahead.")
    a("- Costs are a flat per-trade charge; no market impact model, no borrow fees for shorts.")
    a("- Returns are summed daily returns of a $1 book, ignoring financing and margin.")
    (R / "RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validation", "final"], required=True)
    args = ap.parse_args()

    if not (CFG.cache_dir / "features.npy").exists():
        run_factory(CFG)
    store = Store(CFG)
    preds, sels = get_predictions(store)
    from alphafactory.data import build_panels
    target = build_panels(CFG).target
    if args.stage == "validation":
        preds_val = preds[preds.date.dt.year <= VAL_END]
        stage_validation(evaluate_all(preds_val, target))
    else:
        if not (CFG.report_dir / "chosen_config.json").exists():
            sys.exit("Run --stage validation first: the configuration must be frozen "
                     "before the holdout is looked at.")
        stage_final(store, preds, sels, evaluate_all(preds, target), target)


if __name__ == "__main__":
    main()
