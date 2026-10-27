"""Signal diagnostics for a completed run: Rank IC by holding horizon (decay), by year and
by liquidity tercile, and the rank autocorrelation of each model's scores at the
rebalance lag (signal turnover). Offline; reads the run's predictions and the snapshot it
recorded. Forward returns use the same open-to-open causal index as the labels; removals
inside a horizon are left out here rather than booked at their outcome.

usage: python scripts/signal_report.py <run_dir> <out.json> [--horizons 1 2 5 10 20]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import read_bundle
from costrank.features import prepare


def daily_rank_corr(frame, x, y):
    """Spearman correlation of x and y within each date, vectorized over dates."""
    f = frame[["date", x, y]].dropna()
    g = f.groupby("date")
    rx = g[x].rank() - g[x].transform("count").add(1) / 2
    ry = g[y].rank() - g[y].transform("count").add(1) / 2
    parts = pd.DataFrame({"date": f.date, "xy": rx * ry, "xx": rx * rx, "yy": ry * ry}).groupby("date").sum()
    counts = g.size()
    ic = parts.xy / np.sqrt(parts.xx * parts.yy)
    return ic[counts >= 5].dropna()


def summarize(ic):
    return {"days": int(len(ic)), "mean": float(ic.mean()), "std": float(ic.std()),
            "ir_daily": float(ic.mean() / ic.std()), "share_positive": float((ic > 0).mean())}


def signal_report(run, horizons=(1, 2, 5, 10, 20)):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in manifest["config"].items()})
    holdout = manifest["phase"] == "holdout"
    bundle = read_bundle(manifest["data_path"], allow_after=None if holdout else config.holdout_start)
    panel = prepare(bundle, config).sort_values(["symbol", "date"])
    g = panel.groupby("symbol").tr_open
    entry = g.shift(-1)
    fwd = panel[["date", "symbol", "adv20"]].copy()
    for h in horizons:
        fwd[f"fwd{h}"] = g.shift(-(h + 1)) / entry - 1
    predictions = pd.read_csv(run / "predictions.csv.gz", parse_dates=["date"])
    report = {"run": str(run.resolve()), "horizons": list(horizons), "models": {}}
    for model, p in predictions.groupby("model"):
        x = p.merge(fwd, on=["date", "symbol"], how="left")
        decay = {f"h{h}": summarize(daily_rank_corr(x, "score", f"fwd{h}")) for h in horizons}
        base = daily_rank_corr(x, "score", f"fwd{config.horizon}")
        by_year = {str(y): float(v.mean()) for y, v in base.groupby(base.index.year)}
        x["liquidity"] = x.groupby("date").adv20.transform(
            lambda a: pd.qcut(a.rank(method="first"), 3, labels=["low", "mid", "high"]))
        by_liquidity = {str(k): summarize(daily_rank_corr(part, "score", f"fwd{config.horizon}"))["mean"]
                        for k, part in x.groupby("liquidity", observed=True)}
        wide = p.pivot(index="date", columns="symbol", values="score")
        lagged = {}
        for lag in (1, config.rebalance):
            pairs = pd.DataFrame({"date": wide.index.repeat(wide.shape[1]),
                                  "now": wide.to_numpy().ravel(), "before": wide.shift(lag).to_numpy().ravel()})
            lagged[f"lag{lag}"] = float(daily_rank_corr(pairs, "now", "before").mean())
        report["models"][model] = {"ic_by_horizon": decay, "ic_by_year": by_year,
                                   "ic_by_liquidity_tercile": by_liquidity, "score_rank_autocorrelation": lagged}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("output")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    args = parser.parse_args()
    report = signal_report(args.run, tuple(args.horizons))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for model, r in report["models"].items():
        decay = "  ".join(f"{k}={v['mean']:.4f}" for k, v in r["ic_by_horizon"].items())
        print(f"{model}: IC {decay} | by liquidity {r['ic_by_liquidity_tercile']} | autocorr {r['score_rank_autocorrelation']}")


if __name__ == "__main__":
    main()
