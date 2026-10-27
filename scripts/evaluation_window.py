"""A run's net information ratios under the forward evaluation window of RESEARCH section 10, beside
the figures over all sessions. Offline and descriptive; reads the run and the snapshot it used.

usage: python scripts/evaluation_window.py <run> [<run> ...] --output window.json

Every run starts in cash: its first session's active return is -0.95 x the pool's return before the
first decision, and its second session buys at the open while the close-to-close benchmark also
carries the overnight move. The window leaves out the first session and takes the second session's
benchmark open to close (live.evaluation_benchmark). For each model of the run's single policy at
slip_10bps: the information ratio over all sessions (as reported), without the first session, and
in the window with its 95% circular block bootstrap interval (block 20, 10,000 samples, seed 42).
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
from forward_report import panel_of
from holdout_report import PRIMARY, block_interval, information_ratio, series


def window(run, *, samples=10_000):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    values = dict(manifest["config"])
    for key in ("models", "policies", "slippage_grid_bps"):
        values[key] = tuple(values[key])
    cfg = Config(**values)
    # A supplementary run (RESEARCH section 11) reads a snapshot past the holdout, as the holdout run does.
    holdout = manifest["phase"] in ("holdout", "supplement")
    bundle = read_bundle(manifest["data_path"], allow_after=None if holdout else cfg.holdout_start)
    panel = panel_of(bundle, cfg)
    reported = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    first, last = reported.index.min(), reported.index.max()
    protocol = Protocol(forward_start=str(first.date()), model=cfg.models[0], policy=cfg.policies[0], refit_every=126)
    bench = evaluation_benchmark(panel, bundle, protocol, str(last.date()))
    build = bench.index.min()
    later = bench.index > build
    if not np.allclose(bench[later], reported.reindex(bench.index[later]), rtol=0, atol=1e-12):
        raise ValueError("The close-to-close benchmark differs from the run's benchmark.csv")
    out = {"run": str(run), "phase": manifest["phase"], "policy": cfg.policies[0], "scenario": PRIMARY,
           "first_session": {"date": str(first.date()), "benchmark_close_to_close": float(reported.loc[first])},
           "build_session": {"date": str(build.date()), "benchmark_close_to_close": float(reported.loc[build]),
                             "benchmark_open_to_close": float(bench.loc[build])}, "models": {}}
    for model in cfg.models:
        active, daily = series(run, model)
        windowed = (daily.net_return - cfg.invested * bench.reindex(daily.index)).dropna()
        out["models"][model] = {
            "all_sessions": {"sessions": int(len(active)), "information_ratio": information_ratio(active),
                             "active_first": float(active.loc[first]), "active_build": float(active.loc[build])},
            "without_first": {"sessions": int(len(active) - 1), "information_ratio": information_ratio(active.drop(first))},
            "window": {"sessions": int(len(windowed)), "information_ratio": information_ratio(windowed),
                       "ir_interval_95": block_interval(windowed, samples=samples),
                       "active_return_arith_annual": float(windowed.mean() * 252),
                       "tracking_error_annual": float(windowed.std(ddof=1) * np.sqrt(252)),
                       "active_build": float(windowed.loc[build])}}
    return clean(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = [window(r, samples=args.samples) for r in args.runs]
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
