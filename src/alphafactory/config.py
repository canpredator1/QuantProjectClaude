"""Central configuration. Every number that shapes a result lives here."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw_dir: Path = ROOT / "data" / "raw"
    cache_dir: Path = ROOT / "data" / "cache"
    report_dir: Path = ROOT / "reports"

    # Price history loaded (warm-up for 1-year lookbacks starts here).
    load_start: str = "2004-01-01"
    # First date a signal is evaluated or traded.
    eval_start: str = "2006-01-01"

    # Walk-forward: models for year Y are fit on data strictly before Y.
    first_test_year: int = 2016
    # Years >= holdout_start are the final, untouched test. Everything that
    # was tuned was tuned on first_test_year .. holdout_start-1 only.
    holdout_start: int = 2020
    # Trading days dropped between the end of a training window and the
    # first test day, so overlapping targets cannot leak.
    embargo_days: int = 5

    # Universe filters (applied on top of point-in-time S&P 500 membership).
    min_history_days: int = 126
    min_price: float = 3.0

    # Target: open(t+1) -> open(t+2) return, winsorised to this bound.
    target_clip: float = 0.5

    # Signal selection inside each training window.
    tstat_threshold: float = 3.0      # Harvey, Liu & Zhu (2016) hurdle
    nw_lags: int = 10                 # Newey-West lags for IC t-stats
    max_corr: float = 0.8             # de-duplication threshold
    min_stocks_per_day: int = 100

    # Portfolio.
    quantile: float = 0.1             # top/bottom decile
    cost_bps: float = 5.0             # one-way cost per unit of turnover
    cost_grid_bps: tuple = (0.0, 2.0, 5.0, 10.0, 20.0)

    lgbm_params: dict = field(default_factory=lambda: dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=2000,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l2=10.0,
        verbose=-1,
        num_threads=4,
        seed=7,
    ))
    lgbm_rounds: int = 300
    train_day_stride: int = 2         # use every 2nd training day (speed)


CFG = Config()
