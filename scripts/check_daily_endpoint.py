"""Gate 1 of RESEARCH section 9: may the per-session full-market endpoint replace per-symbol
history? Offline over two raw caches.

usage: python scripts/check_daily_endpoint.py <acquire_dir> <dev4_dir> --delisted codes.csv
           [--chinext sz.300750 --chinext-file per_symbol_unadjusted.csv] --output report.json

Passes only if (1) every (date, code) row of the per-symbol main-board cache appears in the
per-session files with identical open/high/low/close/preclose/volume/amount/tradestatus/isST
(set equality on the overlap of sessions, not an inner join); (2) every delisted code is present
20 sessions before its removal; (3) a ChiNext stock's per-session rows equal its per-symbol
unadjusted history. Extra main-board rows in the per-session files are listed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.acquire import trading_days

FIELDS = ["open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"]


def per_session(root, days, pattern):
    frames = []
    for day in days:
        f = pd.read_csv(Path(root) / "raw" / "daily" / f"{day}.csv", dtype=str, keep_default_na=False)
        frames.append(f.loc[f.code.str.fullmatch(pattern), ["date", "code"] + FIELDS])
    return pd.concat(frames, ignore_index=True)


def same(a, b):
    """Field-wise equality: numbers compared as parsed floats (empty equals empty), flags as text."""
    out = {}
    for col in FIELDS:
        x, y = a[col].astype(str), b[col].astype(str)
        if col in ("tradestatus", "isST"):
            out[col] = x.eq(y)
            continue
        nx, ny = pd.to_numeric(x.where(x.ne("")), errors="coerce"), pd.to_numeric(y.where(y.ne("")), errors="coerce")
        out[col] = (nx == ny) | (nx.isna() & ny.isna())
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("acquire_dir")
    parser.add_argument("dev4_dir")
    parser.add_argument("--delisted", required=True, help="CSV with code,outDate of codes that must be present")
    parser.add_argument("--chinext")
    parser.add_argument("--chinext-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.acquire_dir)
    days = sorted(p.stem for p in (root / "raw" / "daily").glob("*.csv"))
    plan = json.loads((root / "collection-plan.json").read_text(encoding="utf-8"))
    planned = trading_days(root, plan["start"], plan["end"])
    report = {"sessions_planned": len(planned), "sessions_checked": len(days), "first": days[0], "last": days[-1]}
    sessions = per_session(root, days, r"(sh\.60|sz\.00)\d{4}")
    keyed = sessions.set_index(["date", "code"])
    missing, mismatched, compared = [], {f: 0 for f in FIELDS}, 0
    reference_keys = set()
    for path in sorted((Path(args.dev4_dir) / "raw").glob("bars-*.csv")):
        ref = pd.read_csv(path, dtype=str, keep_default_na=False)
        ref = ref[ref.date.isin(days)]
        if ref.empty:
            continue
        keys = pd.MultiIndex.from_frame(ref[["date", "code"]])
        reference_keys.update(keys)
        present = keys.isin(keyed.index)
        missing += [f"{d} {c}" for d, c in keys[~present]]
        if present.any():
            mine = keyed.loc[keys[present]].reset_index()
            ok = same(ref[present].reset_index(drop=True), mine)
            compared += len(ok)
            for col in FIELDS:
                mismatched[col] += int((~ok[col]).sum())
    extra = [f"{d} {c}" for d, c in keyed.index if (d, c) not in reference_keys]
    report["main_board"] = {"reference_rows": len(reference_keys), "missing_in_sessions": len(missing),
                            "missing_examples": missing[:20], "compared_rows": compared, "mismatches": mismatched,
                            "extra_rows_in_sessions": len(extra), "extra_examples": extra[:20]}
    delisted = pd.read_csv(args.delisted, dtype=str)
    calendar = pd.Index(days)
    absent, checked, unchecked = [], 0, []
    for r in delisted.itertuples():
        pos = calendar.searchsorted(r.outDate) - 20
        if pos < 0 or r.outDate > days[-1]:
            unchecked.append(f"{r.code} {r.outDate}")
            continue
        checked += 1
        day = days[pos]
        f = pd.read_csv(root / "raw" / "daily" / f"{day}.csv", dtype=str, keep_default_na=False, usecols=["code"])
        if r.code not in set(f.code):
            absent.append(f"{r.code} {day}")
    report["delisted_presence"] = {"listed": int(len(delisted)), "checked": checked, "absent": absent,
                                   "unchecked": unchecked}
    if args.chinext and args.chinext_file:
        ref = pd.read_csv(args.chinext_file, dtype=str, keep_default_na=False)
        ref = ref[ref.date.isin(days)].reset_index(drop=True)
        if ref.empty:
            raise ValueError("The ChiNext reference shares no session with the collected files")
        mine = per_session(root, sorted(ref.date), r"sz\.30\d{4}")
        mine = mine[mine.code.eq(args.chinext)].set_index("date").reindex(ref.date).reset_index()
        ok = same(ref.reset_index(drop=True), mine) if len(mine) == len(ref) else None
        report["chinext_unadjusted"] = {"code": args.chinext, "rows": len(ref),
                                        "equal_rows": int(ok.all(axis=1).sum()) if ok is not None else 0,
                                        "row_count_matches": len(mine) == len(ref)}
    cx = report.get("chinext_unadjusted")
    report["checks_passed"] = bool(not missing and not extra and not any(mismatched.values()) and not absent
                                   and (cx is None or (cx["row_count_matches"] and cx["equal_rows"] == cx["rows"])))
    # The gate needs every planned session, every listed delisting and the ChiNext comparison.
    report["complete"] = bool(days == planned and not unchecked and cx is not None)
    report["passed"] = report["checks_passed"] and report["complete"]
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "main_board"} | {"main_board": {
        k: v for k, v in report["main_board"].items() if not k.endswith("examples")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
