"""Risk-model attribution of a completed run's portfolios (RESEARCH section 9, report only).

usage: python scripts/style_report.py <run_dir> <schema3_snapshot> --output report.json
           [--config configs/v04-s2p2.json] [--scenario slip_10bps]

The risk model is built on the schema-3 snapshot (eligible pool and parameters from --config, or
the run's own config), so runs on the main-board v0.2 snapshot can be attributed on the full-A
snapshot's exposures. For each close t: active exposures of the end-of-day holdings against the
equal-weight eligible pool scaled to the run's invested share, and the next session's active
return split into factor contributions (exposure x factor return) and a specific remainder.
Holdings outside that day's eligible pool count with their own exposures (RESEARCH section 9
supplement 7); only holdings without a bar that day drop out, reported as `uncovered_weight`.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import read_bundle
from costrank.factors import market
from costrank.features import prepare
from costrank.risk import STYLES, RiskModel


def attribute(run, bundle, config, scenario, start=None, end=None):
    run = Path(run)
    panel = prepare(bundle, config)
    start, end = start or config.dev_start, end or config.dev_end
    days = bundle.calendar[(bundle.calendar >= start) & (bundle.calendar <= end)]
    risk = RiskModel(panel, market(panel, bundle), bundle, config, keep=days)
    returns = risk.factor_returns
    out = {}
    for path in sorted(run.glob(f"*__*__{scenario}")):
        holdings = pd.read_csv(path / "holdings.csv.gz", parse_dates=["date"])
        rows = []
        for day, h in holdings.groupby("date"):
            snap = risk.snapshot(day)
            following = days[days > day]
            if snap is None or not len(following) or following[0] not in returns:
                continue
            w = h.set_index("symbol").weight
            # Holdings outside the pool count with their own exposures; only uncovered ones drop out.
            active = snap.active(w, config.invested)
            exposure = snap.exposure(active)
            f = returns[following[0]]
            rows.append(dict(date=day, outside_weight=float(w[~w.index.isin(snap.names)].sum()),
                             uncovered_weight=float(w[~w.index.isin(active.index)].sum()),
                             **{f"x_{k}": float(v) for k, v in zip(risk.factors, exposure)},
                             **{f"c_{k}": float(v) for k, v in zip(risk.factors, exposure * f)},
                             te_ex_ante=float(np.sqrt(max(snap.daily_variance(active), 0) * 252))))
        table = pd.DataFrame(rows)
        if table.empty:
            continue
        styles = {k: {"mean_active_exposure": float(table[f"x_{k}"].mean()),
                      "contribution_annual": float(table[f"c_{k}"].mean() * 252)} for k in STYLES}
        industry = table[[c for c in table if c.startswith("c_ind_")]].sum(axis=1)
        out[path.name] = {"days": len(table), "styles": styles,
                          "industry_contribution_annual": float(industry.mean() * 252),
                          "mean_ex_ante_te": float(table.te_ex_ante.mean()),
                          "mean_outside_weight": float(table.outside_weight.mean())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("snapshot")
    parser.add_argument("--config")
    parser.add_argument("--scenario", default="slip_10bps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(args.run) / "manifest.json").read_text(encoding="utf-8"))
    config = Config.load(args.config) if args.config else Config(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in manifest["config"].items()})
    # Holdout and supplementary runs (RESEARCH section 11) are attributed over their own period.
    phase = manifest.get("phase")
    period = (manifest["period"] if phase == "supplement" else
              {"start": config.holdout_start, "end": config.holdout_end} if phase == "holdout" else {})
    bundle = read_bundle(args.snapshot, allow_after=None if period else config.holdout_start)
    report = {"run": str(Path(args.run).resolve()), "snapshot": str(Path(args.snapshot).resolve()),
              "scenario": args.scenario, "period": period or {"start": config.dev_start, "end": config.dev_end},
              "portfolios": attribute(args.run, bundle, config, args.scenario, period.get("start"), period.get("end"))}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, r in report["portfolios"].items():
        print(name, {k: round(v["mean_active_exposure"], 3) for k, v in r["styles"].items()})


if __name__ == "__main__":
    main()
