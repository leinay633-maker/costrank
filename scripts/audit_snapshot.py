"""Audit a sealed BaoStock snapshot: provider universe versus the master's lifetimes,
delisting coverage, suspension/ST rows, ex-reference days, and a seeded reconciliation
of bar ex-dates against BaoStock's separate dividend endpoint. Only the reconciliation
uses the network (one serial session); --dividend-symbols 0 keeps the audit offline.

usage: python scripts/audit_snapshot.py <snapshot_dir> <out.json> [--dividend-symbols 8] [--seed 20261012]
"""
import argparse
import json
import random
import socket
import time
from pathlib import Path
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from costrank.config import Config
from costrank.data import board as board_of, mainboard, read_bundle


def universe(raw, studied=mainboard):
    basic = pd.read_csv(raw / "master.csv", dtype=str, keep_default_na=False)
    board = basic[basic.code.map(studied)]
    snapshots = {}
    # v0.1-v0.3 collectors wrote raw/universe-<day>.csv; the v0.4 acquirer writes raw/universe/<day>.csv.
    paths = sorted(raw.glob("universe-*.csv")) + sorted((raw / "universe").glob("*.csv"))
    for path in paths:
        day = path.stem[len("universe-"):] if path.stem.startswith("universe-") else path.stem
        listed = pd.read_csv(path, dtype=str, keep_default_na=False)
        names = set(listed.code[listed.code.map(studied)])
        alive = set(board.code[(board.ipoDate <= day) & ((board.outDate == "") | (board.outDate > day))])
        snapshots[day] = {"snapshot": len(names), "master_alive": len(alive),
                          "snapshot_only": sorted(names - alive)[:20], "master_only": sorted(alive - names)[:20],
                          "snapshot_only_count": len(names - alive), "master_only_count": len(alive - names)}
    return {"master_rows": len(basic), "master_mainboard": len(board),
            "mainboard_type_counts": board["type"].value_counts().to_dict(),
            "mainboard_status_counts": board["status"].value_counts().to_dict(),
            "mainboard_with_outDate": int(board.outDate.ne("").sum()),
            "mainboard_outDate_in_2016_2024": int(board.outDate.between("2016-01-01", "2024-12-31").sum()),
            "absent_from_master_fetched_individually": sorted([p.stem[len("master-"):] for p in raw.glob("master-*.csv")]
                                                               + [p.stem for p in (raw / "master").glob("*.csv")]),
            "snapshots_vs_master_alive": snapshots}


def reconcile(bars, symbols, years, events, count, seed):
    import baostock as bs
    from costrank.providers import guard_closed_socket
    guard_closed_socket()
    socket.setdefaulttimeout(30)
    pick = sorted(random.Random(seed).sample(sorted(symbols), min(len(symbols), count)))
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed {login.error_code} {login.error_msg}")
    records, requests = [], 0
    try:
        for code in pick:
            for year in years:
                rs = bs.query_dividend_data(code=code, year=str(year), yearType="operate")
                requests += 1
                if rs.error_code != "0":
                    raise RuntimeError(f"{code} {year}: {rs.error_code} {rs.error_msg}")
                while rs.next():
                    records.append(dict(zip(rs.fields, rs.get_row_data())))
                time.sleep(0.2)
    finally:
        bs.logout()
    prev = bars.groupby("symbol").close.shift()
    row = pd.Series(bars.index, index=pd.MultiIndex.from_arrays([bars.symbol, bars.date]))
    checks = []
    for r in records:
        if not r["dividOperateDate"]:
            continue
        cash = float(r["dividCashPsBeforeTax"] or 0)
        shares = float(r["dividStocksPs"] or 0) + float(r["dividReserveToStockPs"] or 0)
        item = {"symbol": r["code"], "ex_date": r["dividOperateDate"], "cash_ps": cash, "share_ps": shares}
        i = row.get((r["code"], pd.Timestamp(r["dividOperateDate"])))
        if i is None:
            item["status"] = "no_bar_on_ex_date"
        else:
            item.update(prev_close=None if pd.isna(prev[i]) else float(prev[i]),
                        preclose=float(bars.preclose[i]), tradable=int(bars.tradable[i]))
            if pd.isna(prev[i]):
                item["status"] = "no_prior_close"
            else:
                expected = (prev[i] - cash) / (1 + shares)
                item["expected_ref"] = round(expected, 4)
                item["status"] = "match" if abs(expected - item["preclose"]) <= 0.02 else "mismatch"
        checks.append(item)
    covered = {(c["symbol"], c["ex_date"]) for c in checks}
    picked = sorted(k for k in zip(events.symbol, events.date.dt.strftime("%Y-%m-%d")) if k[0] in pick)
    return {"seed": seed, "symbols": pick, "requests": requests, "dividend_records": len(records),
            "status_counts": pd.Series([c["status"] for c in checks]).value_counts().to_dict() if checks else {},
            "bar_events_without_dividend_record": [k for k in picked if k not in covered], "checks": checks}


def band_violations(bars):
    """Active sessions whose open or close lies outside the board's price band (DATA.md), by board.
    These are the special sessions without limits (listing days, resumptions) the ledger refuses."""
    rows = bars[bars.tradable.eq(1)]
    twenty = rows.symbol.str.startswith("sh.688") | (rows.symbol.str.startswith("sz.30") & (rows.date >= "2020-08-24"))
    rate = np.where(twenty, 0.20, np.where(rows.is_st.eq(1), 0.05, 0.10))
    lo = np.round(rows.preclose.to_numpy() * (1 - rate) + 1e-9, 2)
    hi = np.round(rows.preclose.to_numpy() * (1 + rate) + 1e-9, 2)
    out = (rows.open.to_numpy() < lo - 0.011) | (rows.open.to_numpy() > hi + 0.011) | \
          (rows.close.to_numpy() < lo - 0.011) | (rows.close.to_numpy() > hi + 0.011)
    kinds = rows.symbol.map(board_of)
    return {k: {"active_rows": int((kinds == k).sum()), "outside_band": int((out & (kinds == k).to_numpy()).sum())}
            for k in ("main", "chinext", "star")}


def schema3(bundle):
    """Side-table checks for full-A snapshots: board coverage, bands, float shares, valuation gaps,
    corporate-action reconciliation, intraday aggregates, industries and index constituents."""
    b, m, t = bundle.bars, bundle.instruments, bundle.tables
    kinds = m.symbol.map(board_of)
    report = {"boards": {k: {"instruments": int((kinds == k).sum()),
                             "delisted_in_range": int(((kinds == k) & m.delisted.notna()).sum())}
                         for k in ("main", "chinext", "star")},
              "price_band": band_violations(b)}
    a = t["attributes"].merge(b[["date", "symbol", "volume", "close", "preclose", "tradable"]], on=["date", "symbol"])
    traded = a.tradable.eq(1) & a.volume.gt(0) & a.turn.ge(0.01)
    shares = (a.volume / (a.turn / 100)).where(traded)
    a["shares"] = shares.groupby(a.symbol).ffill()
    prev = a.groupby("symbol").shares.shift()
    jump = (a.shares / prev - 1).abs()
    ex = (a.groupby("symbol").close.shift() - a.preclose).abs() > 1e-6
    report["float_shares"] = {"days_with_change_over_1pct": int((jump > 0.01).sum()),
                              "of_which_on_ex_dates": int(((jump > 0.01) & ex).sum()),
                              "symbols": int(a.symbol[jump > 0.01].nunique()),
                              "change_quantiles": jump[jump > 0.01].quantile([0.5, 0.9, 0.99]).round(4).to_dict()}
    years = a.date.dt.year
    report["valuation_missing_share_by_year"] = {
        col: a[col].isna().groupby(years).mean().round(4).to_dict() for col in ("pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm")}
    rec = t["reconciliation"]
    report["corporate_actions"] = {"explicit_actions": len(bundle.actions),
                                   "reconciliation": rec.status.value_counts().to_dict(),
                                   "explicit_share_of_events": float(rec.status.eq("explicit").mean()) if len(rec) else None}
    intra = t["intraday"]
    later = b[b.date >= "2020-01-02"]
    report["intraday"] = {"rows": len(intra), "symbols": int(intra.symbol.nunique()),
                          "share_of_active_bars_since_2020": float(len(intra) / max(1, int(later.tradable.eq(1).sum()))),
                          "sessions_not_8_bars": int(intra.bars.ne(8).sum()),
                          "first_bar_missing": int(intra.first_volume.isna().sum())}
    ind = t["industries"]
    report["industries"] = {"snapshots": int(ind.date.nunique()), "rows": len(ind),
                            "empty_industry_share": float(ind.industry.eq("").mean())}
    con = t["constituents"]
    report["constituents"] = con.groupby("index").date.nunique().to_dict() | {
        "names_per_snapshot": con.groupby(["index", "date"]).size().groupby(level=0).describe()[["min", "max"]].to_dict("index")}
    return report


def audit_snapshot(root, config, *, dividend_symbols=8, seed=20261012):
    root = Path(root)
    bundle = read_bundle(root, allow_after=config.holdout_start)
    bars, master, cal = bundle.bars, bundle.instruments, bundle.calendar
    report = {"snapshot": str(root.resolve()), "metadata": {k: v for k, v in bundle.metadata.items() if k != "files"},
              "files": bundle.metadata["files"],
              "validate": {"bars": len(bars), "instruments": len(master), "sessions": len(cal),
                           "first_session": str(cal[0].date()), "last_session": str(cal[-1].date()),
                           "prefix_hash": bundle.prefix_hash(config.holdout_start)},
              "provider_universe": universe(root / "raw", (lambda c: board_of(c) is not None)
                                            if bundle.metadata["schema_version"] >= 3 else mainboard)}
    if bundle.metadata["schema_version"] >= 3:
        report["schema3"] = schema3(bundle)
    gone = master[master.delisted.notna()]
    report["sample_coverage"] = {"symbols": len(master), "sh": int(master.symbol.str.startswith("sh").sum()),
        "sz": int(master.symbol.str.startswith("sz").sum()),
        "listed_on_or_after_first_session": int((master.listed >= cal[0]).sum()),
        "delisted_within_range": int((gone.delisted <= cal[-1]).sum()),
        "delisted_list": [f"{r.symbol} {r.listed.date()}→{r.delisted.date()}" for r in gone.itertuples()]}
    suspended = bars.tradable.eq(0)
    streak = bars.assign(s=suspended).groupby("symbol").s.apply(lambda x: x.groupby((~x).cumsum()).sum().max())
    report["suspension_st"] = {"suspended_rows": int(suspended.sum()),
        "symbols_with_suspension": int(bars.symbol[suspended].nunique()),
        "longest_suspension_sessions": int(streak.max()),
        "longest_suspension_symbol": streak.idxmax() if streak.max() > 0 else None,
        "st_rows": int(bars.is_st.eq(1).sum()), "symbols_ever_st": int(bars.symbol[bars.is_st.eq(1)].nunique())}
    prev = bars.groupby("symbol").close.shift()
    events = bars[prev.notna() & ((prev - bars.preclose).abs() > 1e-6)].copy()
    ratio = events.preclose / prev[events.index]
    report["ex_reference_days"] = {"rows": len(events), "symbols": int(events.symbol.nunique()),
        "ratio_quantiles": ratio.quantile([0, 0.05, 0.5, 0.95, 1]).round(4).to_dict(),
        "ratio_above_1": int((ratio > 1).sum())}
    if dividend_symbols > 0:
        report["dividend_reconciliation"] = reconcile(bars, master.symbol, range(cal[0].year, cal[-1].year + 1),
                                                      events, dividend_symbols, seed)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--config", default="configs/v01.json")
    parser.add_argument("--dividend-symbols", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    report = audit_snapshot(args.snapshot, Config.load(args.config),
                            dividend_symbols=args.dividend_symbols, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    brief = {k: v for k, v in report.items() if k not in ("files", "metadata")}
    if "dividend_reconciliation" in brief:
        brief["dividend_reconciliation"] = {k: v for k, v in brief["dividend_reconciliation"].items() if k != "checks"}
    print(json.dumps(brief, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
