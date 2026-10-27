"""Offline assembly of a schema-3 (full A share) snapshot from an acquire() raw cache.

Types the provider's records, removes alias codes and provider artefacts by the same rules as
the main-board collector, reconciles corporate actions for the hybrid mode, aggregates 30-minute
bars into per-session execution fields and seals. No network access.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acquire import (INDICES as INDEX_CODES, dividend_files, dividend_items, first_sessions, master_frame,
                      min30_items, read_raw, report_items, segment_index, trading_days)
from .data import (ACTIONS3_LISTING, ATTRIBUTES, BARS, CONSTITUENTS, DELISTINGS, EXPRESS, FORECASTS, INDICES, INDUSTRIES,
                   INTRADAY, MASTER, RECONCILIATION, STUDIED, seal)
from .providers import curated_resolutions, find_aliases

PRICES = ["open", "high", "low", "close", "preclose"]
ATTRIBUTE_FIELDS = {"turn": "turn", "peTTM": "pe_ttm", "pbMRQ": "pb_mrq", "psTTM": "ps_ttm", "pcfNcfTTM": "pcf_ttm"}
FLAGS = {"1": 1.0, "0": 0.0}
TOLERANCE = 0.02  # yuan, as data.validate reconciles explicit events


def number(series):
    return pd.to_numeric(series.where(series.ne("")), errors="raise").astype(float)


def day(series):
    return pd.to_datetime(series.where(series.ne("")), format="%Y-%m-%d", errors="raise").astype("datetime64[ns]")


def session(text):
    return np.datetime64(text, "ns")


def load_daily(root, days, codes):
    """Studied-board rows of every per-session market file; codes become one shared category."""
    dtype = pd.CategoricalDtype(sorted(codes))
    frames = []
    for day_text in days:
        f = pd.read_csv(Path(root) / "raw" / "daily" / f"{day_text}.csv", dtype=str, keep_default_na=False)
        if not f.date.eq(day_text).all() or f.code.duplicated().any():
            raise ValueError(f"Market file {day_text} has foreign sessions or duplicate codes")
        f = f[f.code.str.fullmatch(STUDIED)]
        unknown = set(f.code) - set(dtype.categories)
        if unknown:
            raise ValueError(f"Codes missing from the security master on {day_text}: {sorted(unknown)[:5]}; run the master stage")
        out = pd.DataFrame({"date": session(day_text), "code": f.code.astype(dtype)})
        for col in PRICES + ["volume", "amount"] + list(ATTRIBUTE_FIELDS):
            out[col] = number(f[col]).to_numpy()
        out["tradable"] = f.tradestatus.map(FLAGS).to_numpy()
        out["is_st"] = f.isST.map(FLAGS).to_numpy()
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


class CodeFrames:
    """find_aliases' view of one code: raw rows with text dates and status flags."""

    def __init__(self, daily):
        self.daily, self.rows = daily, daily.groupby("code", observed=True).indices

    def __getitem__(self, code):
        f = self.daily.iloc[self.rows.get(code, [])]
        return pd.DataFrame({"date": f.date.dt.strftime("%Y-%m-%d").to_numpy(),
                             "tradestatus": f.tradable.map({1.0: "1", 0.0: "0"}).fillna("").to_numpy(),
                             **{c: f[c].to_numpy() for c in ("open", "high", "low", "close", "volume")}})


def repair_rows(bars):
    """Typed twin of providers.repair_baostock_rows (empty fields are NaN here)."""
    flat = pd.concat([(bars[c] - bars.preclose).abs().le(1e-6) for c in ("open", "high", "low", "close")],
                     axis=1).all(axis=1)
    none = {c: bars[c].isna() | bars[c].eq(0) for c in ("volume", "amount")}
    active, halted = bars.tradable.eq(1), bars.tradable.eq(0)
    vwap = bars.amount / bars.volume
    rules = {
        "active_flag_without_trades": active & none["volume"] & none["amount"] & flat,
        "stale_volume_on_suspended_row": halted & bars.volume.gt(0) & none["amount"] & flat,
        "suspended_flag_with_trades": halted & bars.volume.gt(0) & bars.amount.gt(0) & ~flat
            & vwap.ge(bars.low - 0.01) & vwap.le(bars.high + 0.01),
    }
    repairs = [dict(rule=rule, symbol=str(r.code), date=r.date.strftime("%Y-%m-%d"), tradestatus=r.tradable,
                    volume=r.volume, amount=r.amount)
               for rule, mask in rules.items() for r in bars.loc[mask].itertuples()]
    bars = bars.copy()
    bars.loc[rules["active_flag_without_trades"], ["tradable", "volume", "amount"]] = 0.0
    bars.loc[rules["stale_volume_on_suspended_row"], "volume"] = 0.0
    bars.loc[rules["suspended_flag_with_trades"], "tradable"] = 1.0
    return bars, sorted(repairs, key=lambda x: (x["date"], x["symbol"]))


def corporate_actions(root, bars, start, end):
    """Explicit events that reconcile the ex-date reference price, plus a row for every preclose
    jump or provider event saying how the ledger treats it (explicit, or proxy with the reason)."""
    frames = []
    for path in sorted(dividend_files(root, end).values()):
        try:
            frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
        except pd.errors.EmptyDataError:
            continue
    div = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    b = bars[["date", "symbol", "close", "preclose"]].copy()
    b["prev"] = b.groupby("symbol").close.shift()
    keyed = b.set_index(["date", "symbol"])[["prev", "preclose"]]
    if len(div):
        div = div[div.dividOperateDate.between(start, end) & div.code.isin(set(bars.symbol))]
    if len(div):
        ev = pd.DataFrame({"date": day(div.dividOperateDate), "symbol": div.code.to_numpy(),
                           "cash": number(div.dividCashPsBeforeTax).fillna(0).to_numpy(),
                           "bonus": number(div.dividStocksPs).fillna(0).to_numpy(),
                           "conversion": number(div.dividReserveToStockPs).fillna(0).to_numpy(),
                           "pay": day(div.dividPayDate).to_numpy(),
                           "listing": day(div.dividStockMarketDate).to_numpy()})
        # Rare same-day double records (e.g. a special dividend) are booked as one event.
        ev = ev.groupby(["date", "symbol"], as_index=False).agg(
            cash=("cash", "sum"), bonus=("bonus", "sum"), conversion=("conversion", "sum"), pay=("pay", "max"),
            listing=("listing", "max"))
    else:
        ev = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "symbol": pd.Series(dtype=str),
                           "cash": [], "bonus": [], "conversion": [], "pay": pd.Series(dtype="datetime64[ns]"),
                           "listing": pd.Series(dtype="datetime64[ns]")})
    ev = ev.join(keyed, on=["date", "symbol"])
    ev["split"] = 1 + ev.bonus + ev.conversion
    ev["expected"] = (ev.prev - ev.cash) / ev.split
    ev["ok"] = ev.prev.notna() & ev.preclose.notna() & (ev.expected - ev.preclose).abs().le(TOLERANCE)
    good = ev[ev.ok]
    # New shares from a split sell from the provider's listing date (missing: the ex-date itself).
    listing = good.listing.fillna(good.date).where(good.split > 1, good.date)
    actions = pd.DataFrame({"date": good.date, "symbol": good.symbol, "split_ratio": good.split,
                            "cash_per_share": good.cash,
                            "pay_date": np.maximum(good.pay.fillna(good.date), good.date),
                            "taxable_stock_per_share": good.bonus,
                            "share_listing_date": listing.where(listing >= good.date, good.date)})[ACTIONS3_LISTING]
    jumps = b[b.prev.notna() & (b.prev - b.preclose).abs().gt(1e-6)]
    rec = jumps[["date", "symbol", "prev", "preclose"]].merge(
        ev[["date", "symbol", "prev", "preclose", "expected", "ok"]], on=["date", "symbol"], how="outer",
        suffixes=("", "_event"))
    rec["prev"] = rec.prev.fillna(rec.prev_event)
    rec["preclose"] = rec.preclose.fillna(rec.preclose_event)
    rec["status"] = np.where(rec.ok.isna(), "proxy_no_event", np.where(rec.ok.eq(True), "explicit", "proxy_mismatch"))
    rec = rec.rename(columns={"prev": "prev_close", "expected": "expected_preclose"})[RECONCILIATION]
    return (actions.sort_values(["symbol", "date"]).reset_index(drop=True),
            rec.sort_values(["symbol", "date"]).reset_index(drop=True))


def snapshots(root, folder, symbols, columns, start, end):
    frames = []
    for path in sorted((Path(root) / "raw" / folder).glob("*.csv")):
        if not start <= path.stem <= end:
            continue
        f = pd.read_csv(path, dtype=str, keep_default_na=False)
        f = f[f.code.isin(symbols)]
        frames.append(pd.DataFrame({"date": session(path.stem), "symbol": f.code.to_numpy(),
                                    **{k: f[v].to_numpy() for k, v in columns.items()}}))
    return pd.concat(frames, ignore_index=True)


def intraday(root, keys, start, end, read=True):
    """Per-session execution fields from 30-minute bars (bar end times 10:00 ... 15:00); with
    read=False the table is empty whatever minute files the cache holds."""
    rows = []
    for path in sorted((Path(root) / "raw" / "min30").glob("*.csv")) if read else []:
        if not start[:4] <= path.stem[-4:] <= end[:4]:
            continue
        try:
            f = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        if f.empty:
            continue
        f = f[f.date.between(start, end)]
        if f.empty:
            continue
        f = pd.DataFrame({"date": day(f.date), "symbol": f.code, "end": f.time.str[8:12],
                          **{c: number(f[c]) for c in ("open", "high", "low", "close", "volume", "amount")}})
        f = f.sort_values(["date", "end"])
        g = f.groupby("date")
        ret = np.log(f.close / g.close.shift().fillna(f.open))
        first, last = f[f.end.eq("1000")].set_index("date"), f[f.end.eq("1500")].set_index("date")
        out = pd.DataFrame({"bars": g.size(), "realized_var": (ret ** 2).groupby(f.date).sum(),
                            "day_volume": g.volume.sum(), "day_amount": g.amount.sum()})
        out = out.join(first[["open", "close", "volume", "amount"]].add_prefix("first_"))
        out = out.join(last[["open", "close"]].add_prefix("last_"))
        out["symbol"] = f.symbol.iloc[0]
        rows.append(out.reset_index())
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else (object if c == "symbol" else float))
                             for c in INTRADAY}), 0
    frame = pd.concat(rows, ignore_index=True)[INTRADAY]
    inside = pd.MultiIndex.from_frame(frame[["date", "symbol"]]).isin(keys)
    return frame[inside].sort_values(["symbol", "date"]).reset_index(drop=True), int((~inside).sum())


def covering(parts, start, end):
    """The segments overlapping [start, end] if together they cover it without a gap, else None;
    overlapping segments are an error (a record could come from two fetches)."""
    parts = [x for x in parts if x[0] <= end and x[1] >= start]
    if not parts or parts[0][0] > start:
        return None
    for (_, last, _), (first, _, _) in zip(parts, parts[1:]):
        following = (pd.Timestamp(last) + pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d")
        if first < following:
            raise ValueError(f"Overlapping cache segments: {[str(x[2]) for x in parts]}")
        if first > following:
            return None
    return parts if parts[-1][1] >= end else None


def segments_for(parts, start, end, complete=True):
    """covering(), or with complete=False every segment overlapping [start, end] (overlaps still fail)."""
    if complete:
        return covering(parts, start, end)
    picked = [x for x in parts if x[0] <= end and x[1] >= start]
    for (_, last, _), (first, _, _) in zip(picked, picked[1:]):
        if first <= last:
            raise ValueError(f"Overlapping cache segments: {[str(x[2]) for x in picked]}")
    return picked


def read_segments(parts, when):
    """Rows of each segment file whose `when` date lies inside that file's own segment: a record
    is read once, from the fetch whose range contains it, so later fetches never rewrite history."""
    frames = []
    for first, last, path in parts or []:
        try:
            f = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        if len(f):
            d = when(f)
            frames.append(f[(d >= first) & (d <= last)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


REPORT_DATES = {
    "forecast": lambda f: f.profitForcastExpPubDate,
    # The provider keeps the latest version of a revised express report: known from its last update.
    "express": lambda f: f.performanceExpUpdateDate.where(f.performanceExpUpdateDate > f.performanceExpPubDate,
                                                          f.performanceExpPubDate),
}


def empty_table(columns, text):
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c.endswith("date") else (object if c in text else float))
                         for c in columns})


def reports(root, symbols, start, end, complete=True):
    raw = {}
    for kind, when in REPORT_DATES.items():
        index = segment_index(root, kind)
        frames = [read_segments(segments_for(index.get(code, []), start, end, complete), when) for code in sorted(symbols)]
        frame = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else None
        raw[kind] = None if frame is None else frame[when(frame).le(end)].drop_duplicates()
    fc, ex = raw["forecast"], raw["express"]
    if fc is None:
        forecasts = empty_table(FORECASTS, ("symbol", "kind"))
    else:
        forecasts = pd.DataFrame({"symbol": fc.code, "pub_date": day(fc.profitForcastExpPubDate),
            "stat_date": day(fc.profitForcastExpStatDate), "kind": fc.profitForcastType,
            "chg_up": number(fc.profitForcastChgPctUp), "chg_down": number(fc.profitForcastChgPctDwn)})[FORECASTS]
    if ex is None:
        express = empty_table(EXPRESS, ("symbol",))
    else:
        express = pd.DataFrame({"symbol": ex.code, "pub_date": day(ex.performanceExpPubDate),
            "stat_date": day(ex.performanceExpStatDate), "update_date": day(ex.performanceExpUpdateDate),
            "available_date": day(REPORT_DATES["express"](ex)),
            **{k: number(ex[v]) for k, v in {"total_asset": "performanceExpressTotalAsset",
                "net_asset": "performanceExpressNetAsset", "eps_chg_pct": "performanceExpressEPSChgPct",
                "roe_wa": "performanceExpressROEWa", "eps_diluted": "performanceExpressEPSDiluted",
                "gr_yoy": "performanceExpressGRYOY", "op_yoy": "performanceExpressOPYOY"}.items()}})[EXPRESS]
    dropped = {"forecasts_without_pub_date": int(forecasts.pub_date.isna().sum()),
               "express_without_pub_date": int(express.pub_date.isna().sum())}
    forecasts, express = forecasts[forecasts.pub_date.notna()], express[express.pub_date.notna()]
    order = lambda f: f.sort_values(list(f.columns), kind="stable").reset_index(drop=True)
    return order(forecasts), order(express), dropped


def assemble(output, start, end, *, resolutions=None, min30_top=2500, target=None, forward=False, partial=False,
             minutes=True):
    """Seal data/<snapshot> from its raw cache once every acquisition stage is complete; with
    `target`, the sealed tables go to that new directory and the raw cache stays reusable.

    forward: a forward paper-trading session, whose report segments may end before the session
    (refreshed weekly; records still enter only after publication) and which has no minute bars.
    partial: an engineering preview before every per-code file exists (RESEARCH section 9: 2019
    and earlier only): events without dividend details use the proxy, report and minute tables may
    be empty. Marked in the metadata; experiment.run refuses it beyond 2019 and freeze refuses it.
    minutes=False: no 30-minute table at all (RESEARCH section 9 supplement 11): the snapshot of the
    decision path (open fills), whose freeze and 2025 extension are assembled the same way; runs
    that fill at VWAP or cap participation by first-30-minute amount refuse it."""
    root = Path(output)
    out = Path(target) if target is not None else root
    if (out / "metadata.json").exists():
        raise FileExistsError("Snapshots are immutable; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "collection-plan.json").read_text(encoding="utf-8"))
    # Into a separate target, a prefix of a growing (forward) cache may be sealed: every source is
    # filtered to `end`.
    prefix = target is not None and plan.get("start") == start and end <= plan.get("end", "")
    if plan != {"provider": "baostock", "schema": 3, "start": start, "end": end} and not prefix:
        raise ValueError(f"Assembly plan differs from the acquisition plan {plan}")
    raw = root / "raw"
    days = trading_days(root, start, end)
    months = first_sessions(days, "M")
    expected = ([f"daily/{d}" for d in days] + [f"adjust/{d}" for d in days] + [f"industry/{d}" for d in months]
                + [f"{i}/{d}" for i in ("hs300", "zz500") for d in months])
    missing = [x for x in expected if not (raw / f"{x}.csv").exists()]
    index_parts = segment_index(root, "index")
    missing += [] if forward else [f"index/{c} for {start}..{end}" for c in INDEX_CODES
                                   if covering(index_parts.get(c, []), start, end) is None]
    if missing:
        raise ValueError(f"{len(missing)} acquisition files missing, e.g. {missing[:3]}")
    master = master_frame(root, end)
    master = master[master.code.str.fullmatch(STUDIED)]
    daily = load_daily(root, days, master.code)
    seen = sorted(daily.code.unique().astype(str))
    parts = {k: segment_index(root, k) for k in REPORT_DATES}
    uncovered = [] if forward or partial else [f"{k}/{c} for {start}..{end}"
                                               for k, c in report_items(pd.DataFrame({"code": seen}))
                                               if covering(parts[k].get(c, []), start, end) is None]
    have = dividend_files(root, end)
    # Only events up to `end`: a prefix of a grown forward cache must not need a later ex-date's version.
    names = [] if partial else [f"dividend/{c}-{y}" for c, y in dividend_items(root, through=end) if f"{c}-{y}" not in have]
    names += [] if forward or partial or not minutes else [f"min30/{c}-{y}" for c, y, _, _ in min30_items(
        pd.DataFrame({"date": daily.date.dt.strftime("%Y-%m-%d"), "code": daily.code.astype(str),
                      "amount": daily.amount.fillna(0).astype(str)}), start, end, min30_top)]
    missing = uncovered + [x for x in names if not (raw / f"{x}.csv").exists()]
    if missing:
        raise ValueError(f"{len(missing)} per-code acquisition files missing, e.g. {missing[:3]}")
    basic = master[master.code.isin(seen)].copy()
    if basic.ipoDate.eq("").any() or len(basic) != len(seen):
        raise ValueError("Every code in the market files needs a master row with a listing date")
    if ((basic.ipoDate > end) | (basic.outDate.ne("") & (basic.outDate <= start))).any():
        raise ValueError("Master lifetime excludes a code that has market rows in range")
    aliases = find_aliases(basic, CodeFrames(daily))
    dropped = {x["symbol"] for x in aliases}
    basic = basic[~basic.code.isin(dropped)]
    daily = daily[~daily.code.isin(dropped)]
    daily["code"] = daily.code.cat.remove_unused_categories()
    removal = day(daily.code.astype(str).map(basic.set_index("code").outDate))
    after = removal.notna() & (daily.date >= removal)
    if daily.loc[after, "tradable"].ne(0).any():
        raise ValueError("Active bar on or after outDate; cannot infer the delisting day")
    daily, repairs = repair_rows(daily[~after])
    suspended = daily.tradable.eq(0)
    daily.loc[suspended, ["volume", "amount"]] = daily.loc[suspended, ["volume", "amount"]].fillna(0)
    flags = daily[["tradable", "is_st"]]
    if flags.isna().any().any():
        bad = daily[flags.isna().any(axis=1)].head(3)
        raise ValueError(f"Missing trading/ST flags, e.g. {bad[['date', 'code']].to_dict('records')}")
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    symbol = daily.code.astype(str)
    bars = pd.DataFrame({"date": daily.date, "symbol": symbol, **{c: daily[c] for c in PRICES + ["volume", "amount"]},
                         "tradable": daily.tradable.astype("int64"), "is_st": daily.is_st.astype("int64")})[BARS]
    attributes = pd.DataFrame({"date": daily.date, "symbol": symbol,
                               **{v: daily[k] for k, v in ATTRIBUTE_FIELDS.items()}})[ATTRIBUTES]
    actions, reconciliation = corporate_actions(root, bars, start, end)
    symbols = set(basic.code)
    industries = snapshots(root, "industry", symbols, {"industry": "industry", "update_date": "updateDate"}, start, end)
    industries["update_date"] = day(industries.update_date)
    constituents = []
    for index in ("hs300", "zz500"):
        f = snapshots(root, index, set(read_raw(root, index).code), {"update_date": "updateDate"}, start, end)
        constituents.append(f.assign(index=index))
    constituents = pd.concat(constituents, ignore_index=True)
    constituents["update_date"] = day(constituents.update_date)
    constituents = constituents[CONSTITUENTS].sort_values(["date", "index", "symbol"]).reset_index(drop=True)
    industries = industries[INDUSTRIES].sort_values(["date", "symbol"]).reset_index(drop=True)
    indices = []
    for code in INDEX_CODES:
        f = read_segments(segments_for(index_parts.get(code, []), start, end, not forward), lambda x: x.date)
        f = f[f.date.between(start, end)].drop_duplicates("date") if len(f) else pd.DataFrame(columns=["date"] + INDICES[2:])
        indices.append(pd.DataFrame({"date": day(f.date), "index": code,
                                     **{c: number(f[c]) for c in INDICES[2:]}}))
    indices = pd.concat(indices, ignore_index=True)[INDICES].sort_values(["index", "date"]).reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(bars[["date", "symbol"]])
    intraday30, outside = intraday(root, keys, start, end, read=minutes)
    forecasts, express, undated = reports(root, symbols, start, end, complete=not (forward or partial))
    master_out = basic.rename(columns={"code": "symbol", "ipoDate": "listed", "outDate": "delisted"})[MASTER]
    master_out.sort_values("symbol").to_csv(out / "instruments.csv", index=False)
    pd.DataFrame({"date": days}).to_csv(out / "calendar.csv", index=False)
    delistings, curated = (curated_resolutions(resolutions, basic, end) if resolutions
                           else (pd.DataFrame(columns=DELISTINGS), None))
    delistings.to_csv(out / "delistings.csv", index=False)
    for name, frame in (("bars", bars), ("attributes", attributes), ("actions", actions),
                        ("industries", industries), ("constituents", constituents), ("indices", indices),
                        ("intraday", intraday30), ("forecasts", forecasts), ("express", express),
                        ("reconciliation", reconciliation)):
        frame.to_parquet(out / f"{name}.parquet", index=False)
    ledger = json.loads((root / "request-ledger.json").read_text(encoding="utf-8"))
    counts = reconciliation.status.value_counts().to_dict()
    volume_gap = intraday30.merge(bars[["date", "symbol", "volume"]], on=["date", "symbol"])
    volume_gap = (volume_gap.day_volume - volume_gap.volume).abs() > 0.001 * volume_gap.volume.clip(lower=1)
    return seal(out, {"source": "BaoStock", "evidence": "free_data_exploratory",
        "corporate_actions": "hybrid_explicit_proxy", "coverage": "unverified",
        "boards": ["main", "chinext", "star"], "start": start, "end": end, "partial_assembly": bool(partial),
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "requests_by_beijing_day": ledger, "provider_repairs": repairs, "dropped_aliases": aliases,
        "delisting_resolutions": curated,
        "corporate_action_reconciliation": {k: int(v) for k, v in counts.items()},
        "intraday": {"omitted": not minutes,
                     "first_session": str(intraday30.date.min().date()) if len(intraday30) else None,
                     "symbols": int(intraday30.symbol.nunique()), "rows": len(intraday30),
                     "rows_outside_bars": outside, "volume_mismatch_rows": int(volume_gap.sum())},
        "reports_dropped": undated,
        "limitations": ["Delisted coverage has not been independently verified",
            "Corporate events reconcile only where provider dividend records match the ex-date reference price; "
            "other preclose jumps use the reference-price reinvestment proxy",
            "Daily ST/status fields lack independently verified publication timestamps",
            "Valuation ratios are provider values; their report publication lags are not verified",
            "Industry is the provider's CSRC classification sampled monthly",
            "30-minute bars begin 2020-01-02 and cover only codes ranked by trailing amount",
            "Open-auction depth and fills are not observed",
            "Share-exchange and cash delisting outcomes are hand-curated; other removals assume zero recovery",
            "Provider universe, not independently certified"]})
