"""Gate 1 helper (RESEARCH section 9): one code's per-symbol unadjusted daily history from
BaoStock, cached as raw/gate1/<code>-<start>-<end>.csv in an acquisition directory and charged to
its daily request ledger. Do not run it while `costrank acquire` writes the same ledger.

usage: python scripts/fetch_symbol_daily.py data/bao-alla-dev-1 sz.300750 [--start 2016-01-01] [--end 2024-12-31]
"""
import argparse
import json
from pathlib import Path
from costrank.acquire import Fetcher, Ledger, meter

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST"


def fetch(root, code, start, end, budget=45_000):
    import baostock as bs
    from baostock.util import socketutil
    root = Path(root)
    ledger = Ledger(root / "request-ledger.json", budget)
    meter(socketutil, ledger)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code}: {login.error_msg}")
    try:
        frame = Fetcher(bs, root, pause=0.2, retries=3).get(
            f"gate1/{code}-{start}-{end}",
            lambda: bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end, frequency="d",
                                                 adjustflag="3"),
            allow_empty=False)
    finally:
        bs.logout()
    return {"path": str(root / "raw" / "gate1" / f"{code}-{start}-{end}.csv"), "rows": len(frame),
            "requests_this_run": ledger.sent}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("acquire_dir")
    parser.add_argument("code")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    print(json.dumps(fetch(args.acquire_dir, args.code, args.start, args.end), indent=2))


if __name__ == "__main__":
    main()
