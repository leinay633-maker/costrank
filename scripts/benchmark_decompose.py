"""Where the eligible-pool benchmark's return comes from (report only, written after the capacity
grid's results). Rebuilds the benchmark from a snapshot with a configuration (experiment.
benchmark_returns), checks it against runs' benchmark.csv, and splits the equal-weight close-to-
close return by the previous close's ADV20 rank bucket and board, beside wider equal-weight pools
and the CSI 300/500/1000 price indices, in total and by year.

usage: python scripts/benchmark_decompose.py <snapshot> --config <config.json> --output report.json
           [--start 2020-01-01] [--end 2024-12-31] [--check <run>/benchmark.csv ...]

Wider pools use the same previous-close membership and total-return closes as the benchmark but
leave out delisting outcomes (the benchmark's figures do not move at four decimals without them).
Indices are price indices; the pools reinvest dividends.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import board, read_bundle
from costrank.experiment import benchmark_returns
from costrank.features import prepare

BUCKETS = ["1-300", "301-800", "801-1800", "1801+"]
INDICES = {"sh.000300": "CSI 300", "sh.000905": "CSI 500", "sh.000852": "CSI 1000"}


def stats(daily):
    daily = daily.dropna()
    years = len(daily) / 252
    return {"arithmetic": float(daily.mean() * 252), "geometric": float((1 + daily).prod() ** (1 / years) - 1),
            "volatility": float(daily.std() * np.sqrt(252)), "sessions": int(len(daily))}


def yearly(daily):
    return {int(y): float((1 + x.dropna()).prod() - 1) for y, x in daily.groupby(daily.index.year)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--check", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = read_bundle(args.snapshot)
    config = Config.load(args.config)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    panel = prepare(bundle, config)
    bench = benchmark_returns(panel, start, end, bundle).benchmark_return
    checks = {}
    for f in args.check:
        other = pd.read_csv(f, parse_dates=["date"]).set_index("date").benchmark_return
        checks[f] = {"max_abs_diff": float((other - bench.reindex(other.index)).abs().max()), "sessions": int(len(other))}

    p = panel.sort_values(["symbol", "date"]).copy()
    g = p.groupby("symbol", sort=False)
    p["r"] = g.tr_close.pct_change()
    listed = p.symbol.map(bundle.instruments.set_index("symbol").listed)
    # The eligibility filters of features.prepare without the ADV floor and rank cut, then with the floor.
    p["wide"] = ((p.date - listed).dt.days >= config.min_listing_days) & p.tradable.eq(1) & p.is_st.eq(0)
    p["base"] = p.wide & p.adv20.ge(config.min_adv)
    rank = p.adv20.where(p.base).groupby(p.date).rank(method="first", ascending=False)
    p["bucket"] = pd.cut(rank, [0, 300, 800, 1800, np.inf], labels=BUCKETS)
    p["board"] = p.symbol.map(board)
    for col in ("eligible", "base", "wide"):
        p["was_" + col] = g[col].shift(fill_value=False).astype(bool)
    p["was_bucket"] = g.bucket.shift()
    p["was_board"] = g.board.shift()
    p = p[(p.date >= start) & (p.date <= end) & p.r.notna()]
    ew = lambda mask: p[mask].groupby("date").r.mean()

    series = {"benchmark": bench, "pool without delisting outcomes": ew(p.was_eligible)}
    for b in BUCKETS:
        series[f"ADV20 rank {b}"] = ew(p.was_base & p.was_bucket.eq(b))
    series["listed, tradable, non-ST, ADV20 floor"] = ew(p.was_base)
    series["listed, tradable, non-ST"] = ew(p.was_wide)
    for b in sorted(p.was_board.dropna().unique()):
        series[f"pool, board {b}"] = ew(p.was_eligible & p.was_board.eq(b))
    idx = bundle.tables.get("indices")
    if idx is None:
        idx = pd.read_parquet(Path(args.snapshot) / "indices.parquet")
    idx = idx.assign(date=pd.to_datetime(idx.date))
    for code, name in INDICES.items():
        close = idx[idx["index"] == code].set_index("date").close.sort_index()
        series[f"{name} price index"] = close.pct_change().reindex(bench.index)

    share = p[p.was_eligible].groupby("was_bucket", observed=True).size() / p.was_eligible.sum()
    report = {"snapshot": str(Path(args.snapshot).resolve()), "config": args.config, "start": args.start, "end": args.end,
              "checks": checks, "pools": {k: stats(v) for k, v in series.items()},
              "pool_share_by_bucket": {str(k): float(v) for k, v in share.items()},
              "by_year": {k: yearly(series[k]) for k in ("benchmark", "listed, tradable, non-ST", "CSI 1000 price index",
                                                         "ADV20 rank 1-300", "ADV20 rank 801-1800")}}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    for f, c in checks.items():
        print(f"{f}: max |diff| {c['max_abs_diff']:.2e}, sessions {c['sessions']}")
    print(pd.DataFrame(report["pools"]).T.round(4).to_string())
    print("pool share by ADV20 bucket:", {k: round(v, 3) for k, v in report["pool_share_by_bucket"].items()})
    print(pd.DataFrame(report["by_year"]).round(4).to_string())


if __name__ == "__main__":
    main()
