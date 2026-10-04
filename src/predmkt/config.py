"""Settings for the prediction-market project. Every tunable number lives here."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PMConfig:
    data_dir: Path = ROOT / "data" / "polymarket"          # bulky downloads (git-ignored)
    paper_dir: Path = ROOT / "paper"                        # ledger, committed to git
    report_dir: Path = ROOT / "reports" / "polymarket"

    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    request_pause_s: float = 0.25                           # be polite to the public API

    # Taker fee per share = rate * p * (1 - p), by category (Polymarket 2026
    # published schedule; verify before relying on it). Unknown -> default.
    fee_rates: dict = field(default_factory=lambda: {
        "politics": 0.04, "sports": 0.05, "crypto": 0.07, "geopolitics": 0.0,
    })
    default_fee_rate: float = 0.05

    # Which open markets get forecast.
    min_volume_usd: float = 10_000
    min_days_to_end: float = 1.0
    max_days_to_end: float = 60.0
    skip_patterns: tuple = (r"\bup or down\b", r"\bvs\.?\b", r"\bo/u\b", r"\bspread\b")
    max_markets_per_run: int = 15

    # Forecaster.
    model: str = "claude-opus-5-5"
    effort: str = "high"
    use_web_search: bool = True
    max_searches: int = 5

    # Paper trading.
    bankroll: float = 10_000.0
    min_edge: float = 0.05            # model prob - all-in cost, per share
    kelly_fraction: float = 0.25
    max_stake_frac: float = 0.02      # of bankroll, per market
    max_open_frac: float = 0.30       # of bankroll, all open positions

    # Long-shot study: snapshot the price this many days before the end.
    snapshot_days: tuple = (1, 7, 30)


PM = PMConfig()


def fee_per_share(price: float, category: str | None, cfg: PMConfig = PM) -> float:
    rate = cfg.fee_rates.get((category or "").lower(), cfg.default_fee_rate)
    return rate * price * (1.0 - price)
