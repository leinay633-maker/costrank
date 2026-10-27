"""Capacity grid beyond the table of RESEARCH section 9 (report only, written after the grid's
results): per run and model, daily two-way turnover, annual fee and impact drag, impact per traded
yuan, mean invested share and the first session above half invested, fill ratio, partial or
rejected orders, and the information ratio recomputed from the daily ledger beside the decision's.

usage: python scripts/capacity_describe.py <decision.json> <run_grid log>... --output report.json
           [--scenario slip_0bps] [--policy mvo]

Runs come from the `ok <name> <run_dir>` lines of scripts/run_grid.sh logs; the cell and capital
are read from the configuration name (v04-cap-s4p4-1e6 -> S4, 1e6). Capital and the invested
share come from each run's manifest.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEES = ["commission", "stamp", "transfer"]


def grid_runs(logs):
    runs = {}
    for log in logs:
        for line in Path(log).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[1] == "ok":
                runs[parts[2]] = Path(parts[3])
    return runs


def describe(run, model, scenario, policy):
    config = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["config"]
    bench = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    d = pd.read_csv(run / f"{model}__{policy}__{scenario}" / "daily.csv.gz", parse_dates=["date"]).set_index("date")
    nav_prev = d.nav.shift(1).fillna(config["initial_cash"])
    active = d.net_return - config["invested"] * bench.reindex(d.index)
    traded, requested = d.traded_notional.sum(), d.requested_notional.sum()
    invested = d.index[d.exposure > 0.5]
    return {
        "turnover_two_way_daily": float(d.turnover_two_way.mean()),
        "fee_drag_annual": float((d[FEES].sum(axis=1) / nav_prev).mean() * 252),
        "impact_drag_annual": float((d.impact / nav_prev).mean() * 252),
        "impact_bps_per_traded": float(d.impact.sum() / traded * 1e4) if traded else None,
        "exposure_mean": float(d.exposure.mean()),
        "first_session_above_half_invested": str(invested[0].date()) if len(invested) else None,
        "fill_ratio": float(traded / requested) if requested else None,
        "partial_or_rejected_orders": int(d.rejected_or_partial.sum()),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(252)),
        "sessions": int(len(d)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("decision")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--scenario", default="slip_0bps")
    parser.add_argument("--policy", default="mvo")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    table = {(r["cell"], r["model"], r["capital"]): r for r in decision["capacity_table"]}
    rows = []
    for name, run in sorted(grid_runs(args.logs).items()):
        cell, capital = ("S4" if "s4p4" in name else "S2"), float(name.rsplit("-", 1)[1])
        for model in ("lightgbm", "ridge"):
            row = {"name": name, "run": str(run), "cell": cell, "model": model, "capital": capital,
                   **describe(run, model, args.scenario, args.policy)}
            row["information_ratio_decision"] = table[(cell, model, capital)]["information_ratio"]
            rows.append(row)
    gap = max(abs(r["information_ratio"] - r["information_ratio_decision"]) for r in rows)
    report = {"decision": str(Path(args.decision).resolve()), "scenario": args.scenario, "policy": args.policy,
              "max_information_ratio_gap": gap, "rows": rows}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows).drop(columns=["name", "run"]).sort_values(["cell", "model", "capital"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(frame.round(4).to_string(index=False))
    print("max |information ratio - decision|:", gap)


if __name__ == "__main__":
    main()
