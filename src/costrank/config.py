"""Explicit, frozen experiment assumptions. All rates are decimal fractions."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


# Exchange and fee rules are reviewed through this day (reference/rules_2026_review.md): bounds for
# snapshots, fees and forward paper trading; research periods stay within 2025 (holdout_end check).
RULES_REVIEWED_THROUGH = "2026-12-31"


@dataclass(frozen=True)
class Config:
    dev_start: str = "2020-01-01"
    dev_end: str = "2024-12-31"
    holdout_start: str = "2025-01-01"
    holdout_end: str = "2025-12-31"
    train_days: int = 756
    valid_days: int = 126
    test_days: int = 126
    embargo_days: int = 1
    horizon: int = 5
    rebalance: int = 5
    universe_size: int = 500
    min_listing_days: int = 180
    min_adv: float = 10_000_000
    topk: int = 40
    # 0: candidates are the top `topk` names. N >= topk: a holding stays a candidate while it
    # ranks within the top N, and the remaining slots go to the best names not yet held.
    hold_rank: int = 0
    max_weight: float = 0.04
    # "grandfather" (v0.1): a drifted overweight may stay but not grow.
    # "enforce": trim it back to max_weight; the trim is allowed on top of turnover_cap.
    weight_cap_mode: str = "grandfather"
    invested: float = 0.95
    risk_budget: float = 0.03
    turnover_cap: float = 0.30
    cost_aversion: float = 100.0
    commission: float = 0.0003
    minimum_commission: float = 5.0
    assumed_slippage_bps: float = 5.0
    slippage_grid_bps: tuple = (0.0, 5.0, 10.0, 20.0)
    participation: float = 0.01
    initial_cash: float = 1_000_000
    ridge_alpha: float = 100.0
    trees: int = 200
    threads: int = 8
    seed: int = 42
    bootstrap_samples: int = 1000
    bootstrap_block: int = 20
    models: tuple = ("factor", "ridge", "lightgbm")
    policies: tuple = ("topk", "risk", "penalty", "turnover", "cost")
    # v0.4 (RESEARCH §9); the defaults keep v0.1-v0.3 behaviour.
    feature_set: str = "v02"  # "v04": the registered factor library, neutralized
    label_mode: str = "rank"  # "neutral_rank": rank of the industry- and size-neutral label
    fill_price: str = "open"  # "vwap30": VWAP of the execution session's first 30 minutes
    participation_basis: str = "adv20"  # "first30": trailing 20-session mean first-30-minute amount
    impact_coefficient: float = 0.0  # square-root impact k * sigma * sqrt(notional / ADV20) on fills
    dividend_tax: bool = False  # FIFO differentiated dividend tax where dividends are explicit
    selection_fdr: float = 0.10
    selection_coverage: float = 0.80
    selection_max_corr: float = 0.70
    risk_half_life: int = 90
    risk_min_history: int = 250
    specific_shrink: float = 0.30
    te_annual: float = 0.06
    industry_band: float = 0.03
    style_band: float = 0.30
    candidate_top: int = 300
    industry_top: int = 15
    alpha_ic_window: int = 252
    alpha_ic_min: int = 60
    kappa_scale: float = 1.0
    # RESEARCH section 9 gates: False drops provider valuation ratios from factors and risk styles;
    # "first" freezes industries at the first snapshot when the snapshots are not point in time.
    valuation_fields: bool = True
    industry_snapshot: str = "monthly"

    def __post_init__(self):
        from datetime import date
        dates = [date.fromisoformat(x) for x in
                 (self.dev_start, self.dev_end, self.holdout_start, self.holdout_end)]
        if not dates[0] < dates[1] < dates[2] <= dates[3]:
            raise ValueError("Require dev_start < dev_end < holdout_start <= holdout_end")
        if self.holdout_end > "2025-12-31":
            raise ValueError("v0.1 rules validated only through 2025; review 2026 rules first")
        for name in ("train_days", "valid_days", "test_days", "horizon", "rebalance",
                     "universe_size", "min_listing_days", "topk", "trees", "threads",
                     "bootstrap_samples", "bootstrap_block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embargo_days < 0 or self.valid_days <= self.horizon + self.embargo_days + 2:
            raise ValueError("Insufficient validation days / invalid embargo")
        for name in ("max_weight", "invested", "turnover_cap", "participation"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("min_adv", "cost_aversion", "commission", "minimum_commission",
                     "assumed_slippage_bps", "ridge_alpha"):
            if not 0 <= getattr(self, name) < float("inf"):
                raise ValueError(f"Invalid {name}")
        if not 0 < self.risk_budget < 1 or self.initial_cash <= 0:
            raise ValueError("Invalid risk budget / initial cash")
        for name in ("models", "policies", "slippage_grid_bps"):
            if len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f"Duplicate {name} values")
        if self.topk > self.universe_size:
            raise ValueError("topk exceeds universe_size")
        if not self.slippage_grid_bps or any(not 0 <= x <= 100 for x in self.slippage_grid_bps):
            raise ValueError("Slippage grid must be nonempty, between 0 and 100 bps")
        if not self.models or set(self.models) - {"factor", "ridge", "lightgbm", "ic_linear"}:
            raise ValueError("Unknown or empty model list")
        if not self.policies or set(self.policies) - {"topk", "risk", "penalty", "turnover", "cost", "mvo", "mvo_rank"}:
            raise ValueError("Unknown or empty policy list")
        for name, allowed in (("feature_set", {"v02", "v04"}), ("label_mode", {"rank", "neutral_rank"}),
                              ("fill_price", {"open", "vwap30"}), ("participation_basis", {"adv20", "first30"})):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if not isinstance(self.dividend_tax, bool) or not isinstance(self.valuation_fields, bool):
            raise ValueError("dividend_tax and valuation_fields must be true or false")
        if self.industry_snapshot not in {"monthly", "first"}:
            raise ValueError("industry_snapshot must be monthly or first")
        for name in ("risk_half_life", "risk_min_history", "candidate_top", "industry_top",
                     "alpha_ic_window", "alpha_ic_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("selection_fdr", "selection_coverage", "selection_max_corr", "te_annual", "industry_band"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("impact_coefficient", "specific_shrink", "style_band", "kappa_scale"):
            if not 0 <= getattr(self, name) < float("inf"):
                raise ValueError(f"Invalid {name}")
        if self.weight_cap_mode not in {"grandfather", "enforce"}:
            raise ValueError("weight_cap_mode must be grandfather or enforce")
        if isinstance(self.hold_rank, bool) or not isinstance(self.hold_rank, int) or \
                (self.hold_rank and not self.topk <= self.hold_rank <= self.universe_size):
            raise ValueError("hold_rank must be 0 or between topk and universe_size")

    def dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("models", "policies", "slippage_grid_bps"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)  # Unknown keys fail rather than being silently ignored.
