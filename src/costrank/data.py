"""Canonical data contract: raw prices, explicit calendar, historical lifetimes."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .config import RULES_REVIEWED_THROUGH

BARS = ["date", "symbol", "open", "high", "low", "close", "preclose", "volume",
        "amount", "tradable", "is_st"]
MASTER = ["symbol", "listed", "delisted"]
ACTIONS = ["date", "symbol", "split_ratio", "cash_per_share", "pay_date"]
# What holders received when a security left the market; absent symbols recover zero.
DELISTINGS = ["symbol", "date", "kind", "acquirer", "ratio", "cash_per_share", "settle_date"]
# hybrid: reconciled explicit events where the provider has them, the reference-price proxy for
# every other preclose jump (rights issues, resumption references, unmatched records).
MODES = {"reference_reinvestment_proxy", "explicit_events", "hybrid_explicit_proxy"}
CANONICAL = ("bars.csv.gz", "instruments.csv", "calendar.csv", "actions.csv")
# Schema 3 (full A share) adds per-share stock bonus subject to dividend tax at face value.
ACTIONS3 = ACTIONS + ["taxable_stock_per_share"]
# Optional trailing column (RESEARCH section 9 supplement 9): the day the shares a split adds can be
# sold (Shanghai: the session after the ex-date). Snapshots sealed without it read unchanged.
ACTIONS3_LISTING = ACTIONS3 + ["share_listing_date"]
ATTRIBUTES = ["date", "symbol", "turn", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
INDUSTRIES = ["date", "symbol", "industry", "update_date"]
CONSTITUENTS = ["date", "index", "symbol", "update_date"]
INDICES = ["date", "index", "open", "high", "low", "close", "preclose", "volume", "amount"]
INTRADAY = ["date", "symbol", "first_open", "first_close", "first_volume", "first_amount",
            "last_open", "last_close", "bars", "realized_var", "day_volume", "day_amount"]
FORECASTS = ["symbol", "pub_date", "stat_date", "kind", "chg_up", "chg_down"]
# available_date = the later of publication and last update: the provider keeps only the latest
# version of a revised express report, so its values are known from then on.
EXPRESS = ["symbol", "pub_date", "stat_date", "update_date", "available_date", "total_asset", "net_asset",
           "eps_chg_pct", "roe_wa", "eps_diluted", "gr_yoy", "op_yoy"]
RECONCILIATION = ["date", "symbol", "status", "prev_close", "preclose", "expected_preclose"]
# File -> (columns, date column that orders it in time; None for static tables).
TABLES3 = {"bars.parquet": (BARS, "date"), "actions.parquet": (ACTIONS3, "date"),
           "attributes.parquet": (ATTRIBUTES, "date"), "industries.parquet": (INDUSTRIES, "date"),
           "constituents.parquet": (CONSTITUENTS, "date"), "indices.parquet": (INDICES, "date"),
           "intraday.parquet": (INTRADAY, "date"), "forecasts.parquet": (FORECASTS, "pub_date"),
           "express.parquet": (EXPRESS, "available_date"), "reconciliation.parquet": (RECONCILIATION, "date")}
SCHEMA_FILES = {1: CANONICAL, 2: CANONICAL + ("delistings.csv",),
                3: ("instruments.csv", "calendar.csv", "delistings.csv") + tuple(TABLES3)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.12g", lineterminator="\n")
                          .encode("utf-8")).hexdigest()


def frame_hash(frame):
    """Storage-independent content hash: column names, value kinds and values in row order.
    Datetimes hash as int64 nanoseconds, numbers as float64, everything else as text."""
    h = hashlib.sha256()
    for name in frame.columns:
        col = frame[name]
        if pd.api.types.is_datetime64_any_dtype(col):
            kind, data = b"datetime", col.astype("datetime64[ns]").to_numpy().view("int64").tobytes()
        elif pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            kind, data = b"number", np.ascontiguousarray(col.to_numpy(dtype="float64")).tobytes()
        else:
            text = col.astype(object).where(col.notna(), "").astype(str)
            kind, data = b"text", "\x1f".join(text).encode("utf-8")
        h.update(str(name).encode("utf-8") + b"\0" + kind + b"\0" + str(len(col)).encode() + b"\0")
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


BOARDS = {"main": re.compile(r"(sh\.60|sz\.00)\d{4}"), "chinext": re.compile(r"sz\.30\d{4}"),
          "star": re.compile(r"sh\.688\d{3}")}
# One pattern for vectorized `Series.str.fullmatch` over all studied boards.
STUDIED = "|".join(f"(?:{p.pattern})" for p in BOARDS.values())


def board(symbol):
    """Board of an A-share code; None outside the studied boards (B shares, CDRs, BSE)."""
    for name, pattern in BOARDS.items():
        if pattern.fullmatch(str(symbol)):
            return name
    return None


def mainboard(symbol):
    return board(symbol) == "main"


def empty_delistings():
    return pd.DataFrame({"symbol": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                         "kind": pd.Series(dtype=str), "acquirer": pd.Series(dtype=str),
                         "ratio": pd.Series(dtype=float), "cash_per_share": pd.Series(dtype=float),
                         "settle_date": pd.Series(dtype="datetime64[ns]")})


@dataclass
class Bundle:
    bars: pd.DataFrame
    instruments: pd.DataFrame
    calendar: pd.DatetimeIndex
    metadata: dict
    actions: pd.DataFrame
    delistings: pd.DataFrame = field(default_factory=empty_delistings)
    # Schema 3 only, keyed by file stem: attributes, industries, constituents, indices, intraday,
    # forecasts, express, reconciliation.
    tables: dict = field(default_factory=dict)

    def prefix_hash(self, before):
        """Ignore future listings/delistings, but bind every historical observation."""
        date = pd.Timestamp(before)
        m = self.instruments[self.instruments.listed < date].copy()
        m.loc[m.delisted >= date, "delisted"] = pd.NaT
        schema = self.metadata.get("schema_version", 1)
        mode = self.metadata["corporate_actions"]
        if schema >= 3:
            frames = [("instruments", m[MASTER]), ("calendar", pd.DataFrame({"date": self.calendar[self.calendar < date]})),
                      ("delistings", self.delistings[self.delistings.date < date][DELISTINGS])]
            known = set(m.symbol)
            for name, (columns, when) in TABLES3.items():
                frame = {"bars.parquet": self.bars, "actions.parquet": self.actions}.get(name)
                frame = self.tables[name.split(".")[0]] if frame is None else frame
                if name == "actions.parquet":
                    columns = [c for c in ACTIONS3_LISTING if c in frame.columns]
                part = frame[frame[when] < date]
                # Records about a security listed later (prospectus forecasts of a new IPO) are a future
                # listing too: an extended snapshot adds them before the cutoff (RESEARCH 9, supplement 12).
                if "symbol" in part.columns:
                    part = part[part.symbol.isin(known)]
                frames.append((name, part[columns]))
            return hashlib.sha256((mode + "".join(n + frame_hash(f) for n, f in frames)).encode()).hexdigest()
        frames = [self.bars[self.bars.date < date][BARS], m[MASTER],
                  pd.DataFrame({"date": self.calendar[self.calendar < date]}),
                  self.actions[self.actions.date < date][ACTIONS]]
        if schema >= 2:
            frames.append(self.delistings[self.delistings.date < date][DELISTINGS])
        return hashlib.sha256((mode + "".join(canonical_hash(f) for f in frames)).encode()).hexdigest()


def read_bundle(path, *, allow_after=None):
    root = Path(path)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    # Check before reading any holdout market observations.
    if allow_after and meta["end"] >= allow_after:
        raise ValueError("Development refuses a snapshot containing the holdout; use a separate dataset")
    if meta.get("corporate_actions") not in MODES:
        raise ValueError("Declare corporate_actions mode explicitly")
    files = SCHEMA_FILES.get(meta.get("schema_version"))
    if not files or not meta.get("files"):
        raise ValueError("Dataset is not sealed; run import/collect first")
    if set(meta["files"]) != set(files):
        raise ValueError("Dataset manifest must bind every canonical file")
    for name, expected in meta["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    if meta["schema_version"] >= 3:
        return read_bundle3(root, meta)
    b = pd.read_csv(root / "bars.csv.gz", dtype={"symbol": str})[BARS]
    m = pd.read_csv(root / "instruments.csv", dtype=str, keep_default_na=False)[MASTER]
    c = pd.read_csv(root / "calendar.csv", dtype=str)["date"]
    a = pd.read_csv(root / "actions.csv", dtype={"symbol": str})[ACTIONS]
    d = (pd.read_csv(root / "delistings.csv", dtype={"symbol": str, "kind": str, "acquirer": str})[DELISTINGS]
         if "delistings.csv" in files else empty_delistings())
    for f, columns in ((b, ["date"]), (m, ["listed", "delisted"]), (a, ["date", "pay_date"]),
                       (d, ["date", "settle_date"])):
        for col in columns:
            f[col] = pd.to_datetime(f[col], format="%Y-%m-%d", errors="raise")
    calendar = pd.DatetimeIndex(pd.to_datetime(c, format="%Y-%m-%d"))
    bundle = Bundle(b, m, calendar, meta, a, d)
    validate(bundle)
    return bundle


def read_bundle3(root, meta):
    m = pd.read_csv(root / "instruments.csv", dtype=str, keep_default_na=False)[MASTER]
    c = pd.read_csv(root / "calendar.csv", dtype=str)["date"]
    d = pd.read_csv(root / "delistings.csv", dtype={"symbol": str, "kind": str, "acquirer": str})[DELISTINGS]
    for f, columns in ((m, ["listed", "delisted"]), (d, ["date", "settle_date"])):
        for col in columns:
            f[col] = pd.to_datetime(f[col], format="%Y-%m-%d", errors="raise")
    tables = {}
    for name, (columns, _) in TABLES3.items():
        frame = pd.read_parquet(root / name)
        for col in frame.columns[[pd.api.types.is_datetime64_any_dtype(t) for t in frame.dtypes]]:
            frame[col] = frame[col].astype("datetime64[ns]")
        allowed = [columns, ACTIONS3_LISTING] if name == "actions.parquet" else [columns]
        if list(frame.columns) not in allowed:
            raise ValueError(f"{name} columns differ from the schema-3 contract")
        tables[name.split(".")[0]] = frame
    bars, actions = tables.pop("bars"), tables.pop("actions")
    calendar = pd.DatetimeIndex(pd.to_datetime(c, format="%Y-%m-%d"))
    bundle = Bundle(bars, m, calendar, meta, actions, d, tables)
    validate(bundle)
    return bundle


def validate(bundle):
    b, m, c, a = bundle.bars, bundle.instruments, bundle.calendar, bundle.actions
    if len(c) < 2 or not c.is_unique or not c.is_monotonic_increasing:
        raise ValueError("Calendar must be unique, increasing and nonempty")
    if c[0] < pd.Timestamp("2015-08-01") or c[-1] > pd.Timestamp(RULES_REVIEWED_THROUGH):
        raise ValueError(f"Rules are reviewed for 2015-08 through {RULES_REVIEWED_THROUGH} only")
    if b.empty or m.empty or b.duplicated(["date", "symbol"]).any() or m.symbol.duplicated().any():
        raise ValueError("Empty dataset or duplicate bars/instruments")
    schema = bundle.metadata.get("schema_version", 1)
    supported = (lambda s: board(s) is not None) if schema >= 3 else mainboard
    if not m.symbol.map(supported).all() or not b.symbol.isin(m.symbol).all():
        raise ValueError("Unsupported board / unknown instrument")
    if m.listed.isna().any() or ((m.delisted <= m.listed) & m.delisted.notna()).any():
        raise ValueError("Invalid instrument lifetime")
    if not b.date.isin(c).all() or b.date.isna().any():
        raise ValueError("Off-calendar bar")
    b.sort_values(["symbol", "date"], inplace=True)
    b.reset_index(drop=True, inplace=True)
    m.sort_values("symbol", inplace=True)
    m.reset_index(drop=True, inplace=True)
    for col in BARS[2:]:
        b[col] = pd.to_numeric(b[col], errors="raise")
    if not b.tradable.isin([0, 1]).all() or not b.is_st.isin([0, 1]).all():
        raise ValueError("Trading/ST flags cannot be missing or inferred from today's name")
    if (b[["volume", "amount"]] < 0).any().any() or not np.isfinite(b[["volume", "amount"]]).all().all():
        raise ValueError("Invalid volume/amount")
    # Missing rows are NOT suspension. Only explicit suspended rows may carry prices.
    observed = {symbol: pd.DatetimeIndex(frame.date) for symbol, frame in b.groupby("symbol", sort=False)}
    for row in m.itertuples():
        expected = c[(c >= row.listed) & (c < row.delisted if pd.notna(row.delisted) else True)]
        actual = observed.get(row.symbol, pd.DatetimeIndex([]))
        if not expected.equals(actual):
            missing = expected.difference(actual)
            raise ValueError(f"Lifetime/calendar mismatch {row.symbol}; missing={list(missing[:5])}")
    suspended = b.tradable.eq(0)
    if (b.loc[suspended, ["volume", "amount"]] != 0).any().any():
        raise ValueError("Suspended row has nonzero trading volume/amount")
    carry = b.groupby("symbol", sort=False).close.transform(lambda x: x.where(x > 0).ffill())
    for col in ("open", "high", "low", "close", "preclose"):
        bad = ~np.isfinite(b[col]) | (b[col] <= 0)
        b.loc[bad & suspended, col] = carry[bad & suspended]
        if ((~np.isfinite(b[col]) | (b[col] <= 0)) & ~suspended).any():
            raise ValueError(f"Invalid {col}; cannot fabricate prices for active bars")
    if ((b.tradable == 1) & ((b.volume <= 0) | (b.amount <= 0))).any():
        raise ValueError("Active row has no trading volume/amount")
    if ((b.high < b[["open", "close"]].max(axis=1) - 1e-6) |
        (b.low > b[["open", "close"]].min(axis=1) + 1e-6) | (b.low > b.high)).any():
        raise ValueError("Invalid OHLC ordering")
    if a.duplicated(["date", "symbol"]).any() or not a.symbol.isin(m.symbol).all():
        raise ValueError("Duplicate/unknown corporate action")
    if not a.empty:
        amounts = ["split_ratio", "cash_per_share"] + (["taxable_stock_per_share"] if schema >= 3 else [])
        for col in amounts:
            a[col] = pd.to_numeric(a[col], errors="raise")
        if not np.isfinite(a[amounts]).all().all():
            raise ValueError("Nonfinite corporate action")
        if ((a.split_ratio <= 0) | (a.cash_per_share < 0) | (a.pay_date < a.date)).any():
            raise ValueError("Unsupported rights/invalid corporate action")
        if schema >= 3 and ((a.taxable_stock_per_share < 0) | (a.taxable_stock_per_share > a.split_ratio - 1 + 1e-9)).any():
            raise ValueError("Taxable stock bonus must lie within the share split")
        if not a.date.isin(c).all() or a.pay_date.isna().any():
            raise ValueError("Invalid corporate action date")
        if "share_listing_date" in a and (a.share_listing_date.isna() | (a.share_listing_date < a.date)).any():
            raise ValueError("Shares from a split cannot list before the ex-date")
    a.sort_values(["symbol", "date"], inplace=True)
    a.reset_index(drop=True, inplace=True)
    prev = b.groupby("symbol").close.shift()
    changed = prev.notna() & ((prev - b.preclose).abs() > 1e-6)
    if bundle.metadata["corporate_actions"] == "hybrid_explicit_proxy":
        # Every explicit event must reconcile; other preclose jumps fall back to the proxy.
        if not a.empty:
            keyed = b.assign(prev=prev).set_index(["date", "symbol"])[["prev", "preclose"]]
            j = a.join(keyed, on=["date", "symbol"])
            if j.preclose.isna().any() or j.prev.isna().any():
                raise ValueError("Corporate action has no matching bar or no pre-event close")
            if ((j.prev - j.cash_per_share) / j.split_ratio - j.preclose).abs().gt(0.02).any():
                raise ValueError("Action does not reconcile ex-reference price")
    elif bundle.metadata["corporate_actions"] == "explicit_events":
        lookup = a.set_index(["date", "symbol"])
        keys = pd.MultiIndex.from_frame(b[["date", "symbol"]])
        event_rows = keys.isin(lookup.index)
        if not lookup.index.isin(keys).all():
            raise ValueError("Corporate action has no matching bar")
        for i in b.index[changed | event_rows]:
            row = b.loc[i]
            key = (row.date, row.symbol)
            if key not in lookup.index:
                raise ValueError(f"Missing explicit corporate action: {key}")
            event = lookup.loc[key]
            if pd.isna(prev[i]):
                raise ValueError("Need a pre-event raw close to reconcile corporate action")
            expected_ref = (prev[i] - event.cash_per_share) / event.split_ratio
            if abs(expected_ref - row.preclose) > 0.02:
                raise ValueError(f"Action does not reconcile ex-reference price: {key}")
    elif not a.empty:
        raise ValueError("Proxy mode must not also apply explicit actions (double counting)")
    validate_delistings(bundle.delistings, m)
    if b.date.max().strftime("%Y-%m-%d") > bundle.metadata["end"]:
        raise ValueError("Metadata end does not cover all observations")
    if schema >= 3:
        validate_tables(bundle)


def validate_tables(bundle):
    """Schema-3 side tables: keyed like the bars they describe, on exchange sessions."""
    b, c, t = bundle.bars, bundle.calendar, bundle.tables
    keys = pd.MultiIndex.from_frame(b[["date", "symbol"]])
    attributes = t["attributes"].sort_values(["symbol", "date"]).reset_index(drop=True)
    if not pd.MultiIndex.from_frame(attributes[["date", "symbol"]]).equals(keys):
        raise ValueError("Attributes must have exactly one row per bar")
    t["attributes"] = attributes
    intraday = t["intraday"]
    if intraday.duplicated(["date", "symbol"]).any() or \
            not pd.MultiIndex.from_frame(intraday[["date", "symbol"]]).isin(keys).all():
        raise ValueError("Intraday aggregates need unique keys that match a bar")
    for name in ("industries", "constituents", "indices"):
        if not t[name].date.isin(c).all():
            raise ValueError(f"{name} dates must be exchange sessions")
    if t["indices"].duplicated(["date", "index"]).any() or t["constituents"].duplicated(["date", "index", "symbol"]).any():
        raise ValueError("Duplicate index rows")
    for name in ("forecasts", "express"):
        if t[name].pub_date.isna().any() or not t[name].symbol.isin(bundle.instruments.symbol).all():
            raise ValueError(f"{name} need a publication date and a known symbol")
    e = t["express"]
    if (e.available_date.isna() | (e.available_date < e.pub_date) | (e.update_date > e.available_date)).any():
        raise ValueError("Express availability must be the later of publication and update")
    if not t["reconciliation"].status.isin(["explicit", "proxy_no_event", "proxy_mismatch"]).all():
        raise ValueError("Unknown corporate-action reconciliation status")


def validate_delistings(d, m):
    if d.empty:
        return
    if d.symbol.duplicated().any() or not d.kind.isin(["share_exchange", "cash"]).all():
        raise ValueError("Delisting resolutions need one known kind per symbol")
    removal = d.symbol.map(m.set_index("symbol").delisted)
    if removal.isna().any() or not removal.eq(d.date).all():
        raise ValueError("Delisting resolution date must equal the instrument's removal date")
    if d.settle_date.isna().any() or (d.settle_date < d.date).any():
        raise ValueError("Delisting resolution settles before its removal date")
    swap = d[d.kind.eq("share_exchange")]
    if not (np.isfinite(swap.ratio) & (swap.ratio > 0)).all() or not swap.acquirer.isin(m.symbol).all():
        raise ValueError("Share exchange needs a positive ratio and a known acquirer")
    target = m.set_index("symbol").reindex(swap.acquirer)
    alive = (target.listed.to_numpy() <= swap.settle_date.to_numpy()) & \
            (target.delisted.isna().to_numpy() | (target.delisted.to_numpy() > swap.settle_date.to_numpy()))
    if not alive.all():
        raise ValueError("Share exchange acquirer is not listed on its settlement date")
    cash = d[d.kind.eq("cash")]
    if not (np.isfinite(cash.cash_per_share) & (cash.cash_per_share >= 0)).all():
        raise ValueError("Cash resolution needs a nonnegative cash amount")


def seal(path, metadata):
    root = Path(path)
    target = root / "metadata.json"
    if target.exists():
        raise FileExistsError("Snapshots are immutable; choose a new output directory")
    schema = 3 if (root / "bars.parquet").exists() else 2 if (root / "delistings.csv").exists() else 1
    meta = dict(metadata, schema_version=schema)
    meta["files"] = {x: sha256(root / x) for x in SCHEMA_FILES[schema]}
    target.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        return read_bundle(root)
    except Exception:
        # Keep the evidence, but do not leave a directory that claims to be a sealed snapshot.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target.rename(root / f"metadata.invalid-{stamp}.json")
        raise


def synthetic(path, *, days=1000, symbols=60, seed=7):
    """Deterministic accounting fixture, NEVER market evidence."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=days)
    if dates[-1] > pd.Timestamp("2025-12-31"):
        raise ValueError("Synthetic calendar exceeds supported rules")
    frames, master = [], []
    market = rng.normal(0, 0.007, days)
    for i in range(symbols):
        symbol = f"sh.{600000 + i:06d}"
        idio = np.random.default_rng(np.random.SeedSequence([seed, 1, i]))
        vol_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, i]))
        ret = np.clip(market + idio.normal(0, 0.012, days), -0.07, 0.07)
        close = np.round((10 + i) * np.cumprod(1 + ret), 2)
        ref = np.r_[10 + i, close[:-1]]
        op = np.round(ref * (1 + np.clip(ret / 3, -0.02, 0.02)), 2)
        volume = vol_rng.integers(2_000_000, 5_000_000, days)
        frames.append(pd.DataFrame(dict(date=dates, symbol=symbol, open=op,
            high=np.maximum(op, close) + 0.05, low=np.minimum(op, close) - 0.05,
            close=close, preclose=ref, volume=volume, amount=volume * close,
            tradable=1, is_st=0)))
        master.append(dict(symbol=symbol, listed="2010-01-01", delisted=""))
    pd.concat(frames).to_csv(root / "bars.csv.gz", index=False)
    pd.DataFrame(master).to_csv(root / "instruments.csv", index=False)
    pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
    pd.DataFrame(columns=ACTIONS).to_csv(root / "actions.csv", index=False)
    return seal(root, {"source": "SYNTHETIC", "evidence": "synthetic_only",
        "corporate_actions": "explicit_events", "coverage": "synthetic",
        "end": dates[-1].strftime("%Y-%m-%d"), "seed": seed,
        "limitations": ["Business-day fixture, not the Chinese exchange calendar"]})
