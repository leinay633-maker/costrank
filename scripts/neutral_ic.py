"""Rank IC before and after neutralizing the scores against the risk model's exposures (RESEARCH
section 11). Offline and descriptive.

usage: python scripts/neutral_ic.py <run> <schema-3 snapshot> --output neutral-ic.json

For every session of the run's evaluation period, on that session's eligible pool (the risk model's
names): each model's scores are clipped at the median +- 5 x 1.4826 x MAD (as the risk model's
styles), regressed by equal-weight least squares on the session's exposures, and the residual's rank
correlation with the label is taken. "industry" uses the industry dummies; "industry_style" adds
every style with any variation (value and earnings yield are zero without valuation fields). "raw" is
the scores' own rank correlation, i.e. the run's rank_ic.csv. Also the mean cross-sectional R^2 of
the clipped scores on all exposures. Labels run open(t+1) to open(t+h+1), so neighbouring sessions
share h-1 daily returns: the Newey-West t uses lag h-1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import read_bundle
from costrank.experiment import clean
from costrank.factors import market
from costrank.risk import STYLES, RiskModel
from costrank.signals import MAD_SCALE
from forward_report import panel_of

VARIANTS = ("raw", "industry", "industry_style")


def newey_west_t(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def summary(x, lag):
    x = pd.Series(x, dtype=float).dropna()
    if x.empty:
        return {"sessions": 0}
    return {"sessions": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "positive_share": float((x > 0).mean()), "newey_west_t": newey_west_t(x, lag)}


def residual(y, Z):
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def rank_correlation(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def clipped(x):
    med = np.median(x)
    dev = np.median(np.abs(x - med)) * MAD_SCALE
    return np.clip(x, med - 5 * dev, med + 5 * dev) if dev > 0 else x


def session_ics(scores, labels, exposures):
    """One session. scores: names x models; labels: Series over the same names; exposures: names x
    risk-model factors (styles, then ind_* dummies). Returns {model: {variant: ic, "score_r2": r2}}."""
    industry = [c for c in exposures if c.startswith("ind_")]
    out = {}
    for model in scores:
        ok = (scores[model].notna() & labels.notna()).to_numpy()
        s, y = scores[model][ok], labels[ok]
        if ok.sum() < 5 or s.nunique() < 2 or y.nunique() < 2:
            out[model] = {v: np.nan for v in VARIANTS} | {"score_r2": np.nan}
            continue
        X = exposures[ok]
        ones = np.ones((ok.sum(), 1))
        groups = X[industry].loc[:, X[industry].any()].to_numpy(float)
        styles = X[[c for c in STYLES if c in X and X[c].std() > 0]].to_numpy(float)
        x, label = clipped(s.to_numpy(float)), y.to_numpy(float)
        full = residual(x, np.hstack([ones, groups, styles]))
        out[model] = {"raw": rank_correlation(s.to_numpy(float), label),
                      "industry": rank_correlation(residual(x, np.hstack([ones, groups])), label),
                      "industry_style": rank_correlation(full, label),
                      "score_r2": float(1 - full.var() / x.var()) if x.var() > 0 else np.nan}
    return out


def period_of(manifest, cfg):
    if manifest["phase"] == "supplement":
        return manifest["period"]["start"], manifest["period"]["end"]
    if manifest["phase"] == "holdout":
        return cfg.holdout_start, cfg.holdout_end
    return cfg.dev_start, cfg.dev_end


def neutral_ic(run, snapshot):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    values = dict(manifest["config"])
    for key in ("models", "policies", "slippage_grid_bps"):
        values[key] = tuple(values[key])
    cfg = Config(**values)
    start, end = period_of(manifest, cfg)
    bundle = read_bundle(snapshot, allow_after=None if manifest["phase"] in ("holdout", "supplement") else cfg.holdout_start)
    panel = panel_of(bundle, cfg)
    days = bundle.calendar[(bundle.calendar >= start) & (bundle.calendar <= end)]
    predictions = pd.read_csv(run / "predictions.csv.gz", parse_dates=["date"])
    scores = {d: f.pivot(index="symbol", columns="model", values="score") for d, f in predictions.groupby("date")}
    inside = panel[panel.date.isin(days)]
    labels = {d: f.set_index("symbol").label for d, f in inside.groupby("date")}
    risk = RiskModel(panel, market(panel, bundle), bundle, cfg, keep=days)
    rows = []
    for day in days:
        snap = risk.snapshot(day)
        if snap is None or day not in scores:
            continue
        names = snap.names
        ics = session_ics(scores[day].reindex(names), labels[day].reindex(names), snap.exposures)
        rows.extend({"date": day, "model": m, **v} for m, v in ics.items())
    daily = pd.DataFrame(rows)
    reported = pd.read_csv(run / "rank_ic.csv", parse_dates=["date"]).set_index(["date", "model"]).rank_ic
    gap = (daily.set_index(["date", "model"]).raw - reported.reindex(daily.set_index(["date", "model"]).index)).abs()
    lag = cfg.horizon - 1
    out = {"run": str(run.resolve()), "snapshot": str(Path(snapshot).resolve()), "phase": manifest["phase"],
           "period": [str(start), str(end)], "sessions_with_risk_model": int(daily.date.nunique()),
           "sessions_in_period": int(len(days)), "styles": [c for c in STYLES], "newey_west_lag": lag,
           "raw_vs_rank_ic_csv_max_abs_gap": float(gap.max()) if len(gap.dropna()) else None, "models": {}}
    for model, g in daily.groupby("model"):
        key = g.date.dt.year if g.date.dt.year.nunique() > 1 else g.date.dt.quarter.map(lambda q: f"Q{q}")
        out["models"][model] = {v: {**summary(g[v], lag), "by_period": {str(k): float(x) for k, x in g[v].groupby(key).mean().items()}}
                                for v in VARIANTS} | {"mean_score_r2": float(g.score_r2.mean())}
    return clean(out), daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("snapshot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result, daily = neutral_ic(args.run, args.snapshot)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    daily.to_csv(Path(args.output).with_suffix(".daily.csv"), index=False)
    for model, r in result["models"].items():
        print(model, {v: {k: round(r[v][k], 4) for k in ("mean", "newey_west_t") if k in r[v]} for v in VARIANTS},
              "R2", round(r["mean_score_r2"], 3))
    print("raw IC vs rank_ic.csv, max abs gap:", result["raw_vs_rank_ic_csv_max_abs_gap"])


if __name__ == "__main__":
    main()
