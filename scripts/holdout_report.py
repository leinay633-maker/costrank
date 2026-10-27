"""The 2025 report exactly as declared in RESEARCH section 9 (supplement six). Offline.

usage: python scripts/holdout_report.py <holdout_run> [--development-run RUN] --output report.json

For LightGBM (the frozen model), ridge and factor under the run's single policy at slip_10bps:
the annualized net information ratio against 0.95 x the eligible-pool equal-weight benchmark with
a 95% circular block bootstrap interval (block 20, 10,000 samples, seed 42) of the daily active
returns, annualized return, tracking error, maximum drawdown, mean two-way turnover and cost drag;
LightGBM against ridge by the Ledoit-Wolf test (reported, not decided); and, with the development
run of the same configuration, LightGBM's information ratio by development year beside 2025's.
The minimum detectable information ratio is printed with it: one year cannot confirm or refute
ratios of 0.5 to 1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.metrics import minimum_detectable, sharpe_difference_test

PRIMARY = "slip_10bps"
COSTS = ["commission", "stamp", "transfer", "slippage", "dividend_tax"]


def series(run, model):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    policies = manifest["config"]["policies"]
    if len(policies) != 1:
        raise ValueError("A frozen v0.4 run has exactly one policy")
    daily = pd.read_csv(run / f"{model}__{policies[0]}__{PRIMARY}" / "daily.csv.gz", parse_dates=["date"]).set_index("date")
    bench = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    active = (daily.net_return - manifest["config"]["invested"] * bench.reindex(daily.index)).dropna()
    return active, daily


def information_ratio(a):
    sd = a.std(ddof=1)
    return float(a.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan


def block_interval(a, *, block=20, samples=10_000, seed=42):
    x = np.asarray(a, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(samples)
    for i in range(samples):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        s = x[idx]
        out[i] = s.mean() / s.std(ddof=1) * np.sqrt(252) if s.std(ddof=1) > 0 else np.nan
    lo, hi = np.nanquantile(out, [0.025, 0.975])
    return [float(lo), float(hi)]


def describe(active, daily, samples):
    nav = daily.nav
    drag = daily[[c for c in COSTS if c in daily]].sum(axis=1) / nav.shift().fillna(nav.iloc[0])
    return {"sessions": int(len(active)), "information_ratio": information_ratio(active),
            "ir_interval_95": block_interval(active, samples=samples),
            "active_return_annual": float(active.mean() * 252),
            "tracking_error": float(active.std(ddof=1) * np.sqrt(252)),
            "annual_return": float((nav.iloc[-1] / nav.iloc[0]) ** (252 / max(len(nav) - 1, 1)) - 1),
            "max_drawdown": float((nav / nav.cummax() - 1).min()),
            "mean_two_way_turnover": float(daily.turnover_two_way.mean()),
            "annual_cost_drag": float(drag.mean() * 252)}


def report(run, development=None, samples=10_000):
    out = {"run": str(Path(run).resolve()), "scenario": PRIMARY, "models": {}}
    curves = {}
    for model in ("lightgbm", "ridge", "factor"):
        active, daily = series(run, model)
        curves[model] = active
        out["models"][model] = describe(active, daily, samples)
    common = curves["lightgbm"].index.intersection(curves["ridge"].index)
    test = sharpe_difference_test(curves["lightgbm"].loc[common].to_numpy(), curves["ridge"].loc[common].to_numpy(),
                                  block=20, samples=samples, seed=42, bandwidth=20)
    out["lightgbm_minus_ridge"] = test
    # Standard error of one annualized ratio over these sessions, and what a two-sided 5% test with
    # 80% power could detect.
    se = float(np.sqrt(252 / max(len(curves["lightgbm"]), 1)))
    out["single_ratio_se"], out["minimum_detectable_ratio"] = se, minimum_detectable(se)
    if development:
        active, _ = series(development, "lightgbm")
        out["development_lightgbm_ir_by_year"] = {int(y): information_ratio(p) for y, p in active.groupby(active.index.year)}
        out["development_lightgbm_ir"] = information_ratio(active)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--development-run")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = report(args.run, args.development_run, args.samples)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({m: {k: v for k, v in r.items() if k in ("information_ratio", "ir_interval_95", "tracking_error")}
                      for m, r in result["models"].items()}, ensure_ascii=False, indent=2))
    print("minimum detectable ratio over these sessions:", round(result["minimum_detectable_ratio"], 2))


if __name__ == "__main__":
    main()
