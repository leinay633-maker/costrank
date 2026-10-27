"""Forward-period protocol (docs/PAPER.md): which sessions may trade and on which scores. Hashed with
the rest of the protocol code at the freeze; only live.py (operations) and cli.py are not."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .backtest import limit_rate
from .config import RULES_REVIEWED_THROUGH, Config
from .data import STUDIED
from .experiment import MVO_POLICIES, config_hash, protocol_code_hash
from .features import prepare
from .models import Fold, predict_oos

ROW_CHANGE = 0.05  # largest session-to-session change in the number of studied rows
FRESH_DAYS = 10  # calendar days after listing: no bar guaranteed yet
NO_LIMIT_SESSIONS = 5  # a new listing trades without a price limit for its first five sessions


@dataclass(frozen=True)
class Protocol:
    """The forward protocol, written into RESEARCH before the first forward session."""
    forward_start: str
    model: str
    policy: str
    refit_every: int  # sessions per model; the first is trained on windows ending before forward_start
    slippage_bps: float = 10.0
    initial_cash: float = 1_000_000.0
    resolutions: str = "reference/delisting_resolutions.csv"
    reports_weekday: int = 4  # earnings forecasts and express reports refresh on this weekday (v0.4 only)
    reference_run: str = ""  # the development run whose distributions the weekly report compares against

    def __post_init__(self):
        if self.refit_every <= 0 or self.initial_cash <= 0 or not 0 <= self.slippage_bps <= 100:
            raise ValueError("Invalid forward protocol")
        if self.forward_start > RULES_REVIEWED_THROUGH:
            raise ValueError(f"Trading rules are reviewed only through {RULES_REVIEWED_THROUGH}")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def frozen(freeze_file):
    """The frozen Config. Refuses to run once protocol code differs from the freeze or the record's
    config no longer matches its own hash. A filesystem discipline, not a signature."""
    record = json.loads(Path(freeze_file).read_text(encoding="utf-8"))
    if record.get("protocol_code_hash") != protocol_code_hash():
        raise ValueError("Protocol code changed since the freeze: write a new forward protocol")
    values = dict(record["config"])
    for key in ("models", "policies", "slippage_grid_bps"):
        values[key] = tuple(values[key])
    config = Config(**values)
    if record.get("config_hash") != config_hash(config):
        raise ValueError("The freeze record's config does not match its hash")
    return config, record


def forward_folds(calendar, config, start, day, refit_every):
    """models.folds for the forward period: blocks of `refit_every` sessions from `start`, each
    model trained and validated on windows that end before its block, with the same embargo."""
    c = pd.DatetimeIndex(calendar)
    indices = np.flatnonzero((c >= pd.Timestamp(start)) & (c <= pd.Timestamp(day)))
    if len(indices) == 0:
        raise ValueError("No forward session yet")
    fmt = lambda j: c[j].strftime("%Y-%m-%d")
    out = []
    for number, offset in enumerate(range(0, len(indices), refit_every)):
        i = indices[offset]
        v = i - config.valid_days
        tr = v - config.train_days
        if tr < 0 or v - config.embargo_days < 0:
            raise ValueError("Insufficient history before the forward period")
        out.append(Fold(number, fmt(tr), fmt(v), fmt(v - config.embargo_days), fmt(i - config.embargo_days),
                        fmt(i), fmt(indices[min(offset + refit_every - 1, len(indices) - 1)])))
    return out


def gates(root, day, previous=None):
    """Checks of one session's market file before it may produce a signal, after the assembly's own
    provider repairs: files present, every listed code present, row count stable, OHLC consistent,
    suspended rows without trades, closes inside the board's price band."""
    from .acquire import master_frame
    from .snapshot import load_daily, repair_rows
    raw = Path(root) / "raw"
    fail = lambda name, why: {"passed": False, "checks": {name: False}, "examples": {name: [why]}, "rows": 0}
    if not (raw / "daily" / f"{day}.csv").exists():
        return fail("market_file", "missing")
    master = master_frame(root, day)
    master = master[master.code.str.fullmatch(STUDIED)]
    try:
        bars = load_daily(root, [day], master.code)
    except ValueError as exc:  # foreign sessions, duplicate codes, codes missing from the master
        return fail("market_file", str(exc))
    checks = {"market_file": len(bars) > 0, "adjust_file": (raw / "adjust" / f"{day}.csv").exists()}
    examples = {}
    if previous is not None and (raw / "daily" / f"{previous}.csv").exists():
        before = pd.read_csv(raw / "daily" / f"{previous}.csv", dtype=str, usecols=["code"]).code
        n0 = int(before.str.fullmatch(STUDIED).sum())
        checks["row_count"] = abs(len(bars) - n0) <= ROW_CHANGE * n0
    when = pd.Timestamp(day)
    listed = pd.to_datetime(master.ipoDate.where(master.ipoDate.ne("")), errors="coerce")
    gone = pd.to_datetime(master.outDate.where(master.outDate.ne("")), errors="coerce")
    expected = set(master.code[(listed <= when - pd.Timedelta(FRESH_DAYS, unit="D")) & (gone.isna() | (gone > when))])
    missing = sorted(expected - set(bars.code.astype(str)))
    checks["listed_codes_present"], examples["listed_codes_present"] = not missing, missing[:5]
    fixed, _ = repair_rows(bars)
    o, h, low, close, pre, vol = (fixed[c] for c in ("open", "high", "low", "close", "preclose", "volume"))
    traded = fixed.tradable.eq(1) & vol.gt(0)
    bad = traded & ~((low <= np.minimum(o, close)) & (np.maximum(o, close) <= h) & (low > 0))
    code = fixed.code.astype(str)
    checks["ohlc"], examples["ohlc"] = not bad.any(), code[bad].head(5).tolist()
    halted = fixed.tradable.eq(0) & vol.gt(0)
    checks["suspended_without_trades"], examples["suspended_without_trades"] = not halted.any(), code[halted].head(5).tolist()
    first = pd.to_datetime(code.map(dict(zip(master.code, master.ipoDate))), errors="coerce")
    # The session number of `day` in each listing's life (1 on the listing day), from the cached
    # calendars: a holiday can stretch the five sessions without a price limit past FRESH_DAYS.
    from .acquire import calendar_parts
    cal = pd.concat([f for *_, f in calendar_parts(root)], ignore_index=True).drop_duplicates("calendar_date")
    sessions = np.sort(pd.to_datetime(cal.loc[cal.is_trading_day.eq("1") & cal.calendar_date.le(day), "calendar_date"]).to_numpy())
    nth = len(sessions) - np.searchsorted(sessions, first.fillna(when).to_numpy(), side="left")
    fresh = first.isna() | (first > when - pd.Timedelta(FRESH_DAYS, unit="D")) | pd.Series(nth <= NO_LIMIT_SESSIONS, index=first.index)
    rate = np.array([float(limit_rate(c, day, bool(st))) for c, st in zip(code, fixed.is_st.fillna(0))])
    over = traded & ~fresh & pre.gt(0) & ((close / pre - 1).abs() > rate + 0.0101 / pre)
    checks["price_band"], examples["price_band"] = not over.any(), code[over].head(5).tolist()
    return {"passed": bool(all(checks.values())), "checks": checks, "examples": examples, "rows": int(len(bars))}


def forward_scores(bundle, config, protocol, day):
    """The protocol model's scores on every forward session through `day`, with the fold audits."""
    cfg = replace(config, models=(protocol.model,), policies=(protocol.policy,))
    v04 = cfg.feature_set == "v04"
    if v04 or protocol.policy in MVO_POLICIES:
        from .signals import prepare_v04
        panel, signals = prepare_v04(bundle, cfg, library=v04)
    else:
        panel, signals = prepare(bundle, cfg), None
    blocks = forward_folds(bundle.calendar, cfg, protocol.forward_start, day, protocol.refit_every)
    predictions, audits = predict_oos(panel, bundle.calendar, cfg, signals=signals if v04 else None, fold_list=blocks)
    scores = predictions[predictions.model == protocol.model][["date", "symbol", "score"]].reset_index(drop=True)
    return cfg, panel, scores, audits
