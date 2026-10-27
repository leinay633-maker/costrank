"""Optimizer feasibility of a finished run (RESEARCH section 9: the 5% `hold_fallback` rule in
"可行性" and condition 4). Offline.

usage: python scripts/feasibility_report.py <run_dir> --output report.json [--scenario slip_10bps]
           [--policies mvo mvo_rank] [--limit 0.05]

Per model and policy: rebalance days, the share and reasons of `hold_fallback`, solver statuses,
solve seconds, the realized ex-ante tracking error and largest style and industry active
exposures on solved days; for runs after supplement 7 also the fallback attribution, the days
on which participation moved a bound, and the book's weight outside the pool; after supplement 8
the days solved only with widened risk limits, which count with `hold_fallback` against the limit. `over_limit` flags an mvo share above the limit: on 2019-and-earlier
debug runs that means stop and amend section 9; on the development run it voids the primary test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(run, scenario="slip_10bps", policies=("mvo", "mvo_rank"), limit=0.05):
    out = {}
    for path in sorted(Path(run).glob(f"*__*__{scenario}")):
        model, policy, _ = path.name.split("__")
        if policy not in policies:
            continue
        alloc = pd.read_csv(path / "allocations.csv.gz")
        if alloc.empty:
            out[path.name] = {"rebalances": 0}
            continue
        fallback = alloc.status.eq("hold_fallback")
        # Supplement 8: a solution only after widening the risk limits counts against the limit too.
        widened = (alloc.status.eq("optimal") & alloc.risk_limit_scale.fillna(1.0).gt(1.0)
                   if "risk_limit_scale" in alloc else pd.Series(False, index=alloc.index))
        solved = alloc[~fallback]
        reasons = alloc.loc[fallback, "reason"].fillna("unknown") if "reason" in alloc else pd.Series(dtype=str)
        stat = lambda col, fn: float(fn(solved[col])) if col in solved and len(solved) else np.nan
        out[path.name] = {
            "model": model, "policy": policy, "rebalances": int(len(alloc)),
            "hold_fallback_share": float(fallback.mean()),
            "widened_share": float(widened.mean()),
            "max_risk_limit_scale": float(alloc.risk_limit_scale.max()) if "risk_limit_scale" in alloc else 1.0,
            "infeasible_share": float((fallback | widened).mean()),
            "fallback_reasons": {str(k): int(v) for k, v in reasons.value_counts().items()},
            "solver_status": {str(k): int(v) for k, v in alloc.get("solver_status", pd.Series(dtype=str))
                              .fillna("none").value_counts().items()},
            "solve_seconds_median": stat("solve_seconds", np.median),
            "solve_seconds_p95": stat("solve_seconds", lambda s: np.quantile(s, 0.95)),
            "solve_seconds_max": stat("solve_seconds", np.max),
            "ex_ante_te_median": stat("ex_ante_te", np.median), "ex_ante_te_max": stat("ex_ante_te", np.max),
            "max_style_active": stat("max_style_active", np.max),
            "max_industry_active": stat("max_industry_active", np.max),
            "candidates_median": stat("candidates", np.median), "holdings_median": stat("holdings", np.median),
            "over_limit": bool(policy == "mvo" and (fallback | widened).mean() > limit)}
        if "outside_weight" in alloc:
            # Supplement 7: whether the budget and trading limits alone were solvable on the fallback days,
            # how often participation made a bound unreachable, and the book's weight outside the pool.
            out[path.name].update({
                "fallback_without_risk_limits": {str(k): int(v) for k, v in alloc.get(
                    "without_risk_limits", pd.Series(np.nan, index=alloc.index))[fallback].fillna("none").value_counts().items()},
                "days_with_unreachable_names": int(alloc.unreachable_names.fillna(0).gt(0).sum()),
                "days_with_relaxed_industry_bounds": int(alloc.relaxed_industry_bounds.fillna(0).gt(0).sum()),
                "outside_weight_median": float(alloc.outside_weight.median()),
                "outside_weight_max": float(alloc.outside_weight.max()),
                "max_excess": {k: stat(k, np.max) for k in ("te_excess", "style_excess", "industry_excess")}})
    if not out:
        # e.g. the runs root instead of one timestamped run: no evidence is not a pass.
        raise ValueError(f"No {'/'.join(policies)} portfolio for {scenario} under {run}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--scenario", default="slip_10bps")
    parser.add_argument("--policies", nargs="*", default=["mvo", "mvo_rank"])
    parser.add_argument("--limit", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"run": str(Path(args.run).resolve()), "scenario": args.scenario, "limit": args.limit,
              "portfolios": summarize(args.run, args.scenario, tuple(args.policies), args.limit)}
    report["any_over_limit"] = any(v.get("over_limit") for v in report["portfolios"].values())
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, v in report["portfolios"].items():
        print(name, v.get("rebalances"), round(v.get("hold_fallback_share", float("nan")), 4),
              round(v.get("widened_share", float("nan")), 4), v.get("fallback_reasons"))
    print("any_over_limit", report["any_over_limit"])


if __name__ == "__main__":
    main()
