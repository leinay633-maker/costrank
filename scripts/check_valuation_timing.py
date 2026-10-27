"""Gate 2 of RESEARCH section 9: do the provider's valuation ratios change only after the report
that justifies them is public? Fetches annual-report publication dates (BaoStock profit data,
cached in raw/profit/, counted against the daily request budget), then reads the per-session files.

usage: python scripts/check_valuation_timing.py <acquire_dir> --output report.json [--pairs 30]

For 30 company-years (main-board codes in code order, annual reports for 2020-2023 published after
15 April of the next year): book value per share (close / pbMRQ) and TTM earnings per share
(close / peTTM) are step functions. A step between the report period end and the publication
day that no earlier flash report (express) explains, and that is not an ex-date, means the field
used unpublished data. The gate passes with no such step and with at least half the company-years
informative for each field (the value moved between period end and three sessions after
publication), since a series that never moves cannot reveal a leak.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.acquire import Fetcher, Ledger, meter, segment
from costrank.providers import guard_closed_socket

STEP = 0.003  # relative change that counts as a step rather than ratio rounding


def steps(series):
    rel = (series / series.shift() - 1).abs()
    return list(series.index[rel > STEP])


def evaluate(root, days, pairs, wanted_pairs):
    """Offline part of the gate: steps of book value and TTM earnings per share before publication
    that neither an earlier flash report nor an ex-date explains."""
    root = Path(root)
    wanted = {c for c, *_ in pairs}
    frames = []
    for day in days:
        f = pd.read_csv(root / "raw" / "daily" / f"{day}.csv", dtype=str, keep_default_na=False,
                        usecols=["date", "code", "close", "preclose", "pbMRQ", "peTTM"])
        frames.append(f[f.code.isin(wanted)])
    panel = pd.concat(frames, ignore_index=True)
    for col in ("close", "preclose", "pbMRQ", "peTTM"):
        panel[col] = pd.to_numeric(panel[col].where(panel[col].ne("")), errors="coerce")
    rows, leaks = [], 0
    for code, year, pub, stat in pairs:
        g = panel[panel.code.eq(code)].set_index("date").sort_index()
        # Ex-dates move book value per share legitimately; steps on them are not evidence.
        ex_dates = set(g.index[(g.close.shift() - g.preclose).abs() > 1e-6])
        window = g[(g.index > stat) & (g.index <= pub)]
        express = []
        for express_path in sorted((root / "raw" / "express").glob(f"{code}__*__*.csv")):
            try:
                ex = pd.read_csv(express_path, dtype=str, keep_default_na=False)
            except pd.errors.EmptyDataError:
                continue
            if len(ex):
                express += list(ex.loc[ex.performanceExpStatDate.eq(stat), "performanceExpPubDate"])
        express = sorted(set(express))
        allowed_from = min([pub] + express)
        found, informative = {}, {}
        around = g[(g.index >= stat) & (g.index > pub)].head(4)
        for name, ratio in (("bvps", g.close / g.pbMRQ.where(g.pbMRQ > 0)),
                            ("eps_ttm", g.close / g.peTTM.where(g.peTTM.abs() > 0))):
            series = ratio.loc[window.index].dropna()
            found[name] = [d for d in steps(series) if d < allowed_from and d not in ex_dates]
            # A company-year tells something only if the per-share value moved between the period
            # end and three sessions after publication; a flat series cannot show a leak.
            before = ratio[(ratio.index <= stat)].dropna().tail(1)
            after = ratio.loc[around.index].dropna().tail(1)
            informative[name] = bool(len(before) and len(after)
                                     and abs(after.iloc[0] / before.iloc[0] - 1) > STEP)
        leak = any(found.values())
        leaks += leak
        rows.append(dict(code=code, year=year, published=pub, period_end=stat, express=express,
                         early_steps=found, informative=informative, leak=leak))
    counts = {k: sum(r["informative"][k] for r in rows) for k in ("bvps", "eps_ttm")}
    return {"pairs": len(rows), "leaks": leaks, "informative": counts,
            "passed": len(rows) == wanted_pairs and leaks == 0 and min(counts.values()) >= wanted_pairs // 2,
            "details": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("acquire_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=45_000)
    args = parser.parse_args()
    root = Path(args.acquire_dir)
    days = sorted(p.stem for p in (root / "raw" / "daily").glob("*.csv") if p.stem >= "2020-01-01")
    first = pd.read_csv(root / "raw" / "daily" / f"{days[0]}.csv", dtype=str, usecols=["code"]).code
    last = pd.read_csv(root / "raw" / "daily" / f"{days[-1]}.csv", dtype=str, usecols=["code"]).code
    codes = sorted(set(first[first.str.fullmatch(r"(sh\.60|sz\.00)\d{4}")]) & set(last))
    import baostock as bs
    from baostock.util import socketutil
    guard_closed_socket()
    meter(socketutil, Ledger(root / "request-ledger.json", args.budget))
    if bs.login().error_code != "0":
        raise RuntimeError("BaoStock login failed")
    fetch = Fetcher(bs, root, pause=0.2, retries=3)
    plan = json.loads((root / "collection-plan.json").read_text(encoding="utf-8"))
    pairs = []
    try:
        for code in codes:
            for year in range(2020, 2024):
                profit = fetch.get(f"profit/{code}-{year}", lambda c=code, y=year: bs.query_profit_data(code=c, year=y, quarter=4))
                if profit.empty or not profit.pubDate.iloc[0]:
                    continue
                pub = profit.pubDate.iloc[0]
                if pub[5:] > "04-15" and pub <= days[-1]:
                    pairs.append((code, year, pub, profit.statDate.iloc[0]))
            if len(pairs) >= args.pairs:
                break
        # Flash reports of these codes: the same plan-range request (and cache file) as the
        # acquisition's reports stage, which may not have run yet.
        for code in sorted({c for c, *_ in pairs[:args.pairs]}):
            fetch.get(f"express/{code}{segment(plan['start'], plan['end'])}", lambda c=code: (
                bs.query_performance_express_report(c, start_date=plan["start"], end_date=plan["end"])))
    finally:
        bs.logout()
    report = evaluate(root, days, pairs[:args.pairs], args.pairs)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pairs", "leaks", "informative", "passed")}))


if __name__ == "__main__":
    main()
