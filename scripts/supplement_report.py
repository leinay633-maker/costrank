"""Report of a supplementary run as declared in RESEARCH section 11. Offline, descriptive.

usage: python scripts/supplement_report.py <supplement_run> --development-run RUN --output report.json

For the run's single policy at slip_10bps: the section 9 figures (holdout_report.report: information
ratio against 0.95 x the eligible-pool equal weight over all sessions, interval, tracking error,
drawdown, turnover, cost drag, LightGBM against ridge, development years); the same ratios in the
forward evaluation window (evaluation_window.window: without the first, all-cash session, the build
session's benchmark open to close); in that window the revision-two comparisons of section 10 (wide
equal weight, CSI 1000, CSI 500); each model's beta to the pool and absolute return (summary.csv);
and the daily rank IC (mean, sd, positive share, Newey-West t with lag horizon - 1).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import read_bundle
from costrank.experiment import clean
from costrank.live import Protocol, evaluation_benchmark
from evaluation_window import window
from forward_report import against, comparison_returns, panel_of
from holdout_report import PRIMARY, report as section9
from neutral_ic import summary as ic_summary


def report(run, development, samples=10_000):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["phase"] != "supplement" or manifest["status"] != "complete":
        raise ValueError("Needs a completed supplementary run")
    values = dict(manifest["config"])
    for key in ("models", "policies", "slippage_grid_bps"):
        values[key] = tuple(values[key])
    cfg = Config(**values)
    out = {"run": str(run.resolve()), "period": manifest["period"], "scenario": PRIMARY,
           "section9": section9(run, development, samples), "forward_window": window(run, samples=samples)}
    bundle = read_bundle(manifest["data_path"])
    panel = panel_of(bundle, cfg)
    reported = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    first, last = reported.index.min(), reported.index.max()
    protocol = Protocol(forward_start=str(first.date()), model=cfg.models[0], policy=cfg.policies[0], refit_every=126)
    bench = evaluation_benchmark(panel, bundle, protocol, str(last.date()))
    build = bench.index.min()
    series = {"eligible-pool benchmark": bench, **dict(comparison_returns(panel, bundle, cfg, build, bench.index).items())}
    summary = pd.read_csv(run / "summary.csv")
    ic = pd.read_csv(run / "rank_ic.csv", parse_dates=["date"])
    out["models"] = {}
    for model in cfg.models:
        daily = pd.read_csv(run / f"{model}__{cfg.policies[0]}__{PRIMARY}" / "daily.csv.gz", parse_dates=["date"]).set_index("date")
        net = daily.net_return.reindex(bench.index)
        row = summary[(summary.model == model) & (summary.scenario == PRIMARY)].iloc[0]
        out["models"][model] = {
            "total_return": float(row.total_return), "annualized_return": float(row.annualized_return),
            "beta_to_pool": float(row.beta), "max_drawdown": float(row.max_drawdown),
            "window_net_total_return": float((1 + net.dropna()).prod() - 1),
            "comparisons": {k: against(net, v, cfg.invested) for k, v in series.items()},
            "rank_ic": ic_summary(ic[ic.model == model].rank_ic, cfg.horizon - 1)}
    out["pool_total_return"] = float((1 + reported).prod() - 1)
    out["window"] = {"first_session": str(first.date()), "build_session": str(build.date()), "last_session": str(last.date()),
                     "sessions": int(len(bench))}
    return clean(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--development-run", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = report(args.run, args.development_run, args.samples)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for model, r in result["forward_window"]["models"].items():
        w = r["window"]
        print(model, "forward window IR", round(w["information_ratio"], 3), [round(x, 2) for x in w["ir_interval_95"]],
              "| section 9 IR", round(result["section9"]["models"][model]["information_ratio"], 3),
              "| mean IC", round(result["models"][model]["rank_ic"]["mean"], 4))
    print("minimum detectable ratio:", round(result["section9"]["minimum_detectable_ratio"], 2))


if __name__ == "__main__":
    main()
