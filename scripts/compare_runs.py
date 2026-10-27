"""A/B two completed runs that share sessions and scenario names, such as two protocol
versions on one snapshot: per scenario the paired daily net-return difference (variant
minus base) with the variant's block bootstrap, Holm within each cost scenario, and the
turnover, cost, beta and active-return change. Offline.

usage: python scripts/compare_runs.py <base_run> <variant_run> <out.json> [--samples N]
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from costrank.experiment import family_holm
from costrank.metrics import paired_block_interval

COSTS = ["total_commission", "total_stamp", "total_transfer", "total_slippage"]


def compare_runs(base, variant, samples=None):
    """samples overrides the variant's bootstrap count: with 1000 draws the smallest p-value
    is 0.002, too coarse for Holm across 15-25 tests."""
    base, variant = Path(base), Path(variant)
    manifests = [json.loads((r / "manifest.json").read_text(encoding="utf-8")) for r in (base, variant)]
    if any(m.get("status") != "complete" for m in manifests):
        raise ValueError("Both runs must be complete")
    config = dict(manifests[1]["config"])
    if samples:
        config["bootstrap_samples"] = samples
    summaries = [pd.read_csv(r / "summary.csv").set_index(["model", "policy", "scenario"]) for r in (base, variant)]
    rows = []
    for key in summaries[1].index:
        name = "__".join(key)
        if key not in summaries[0].index:
            continue
        a, b = (pd.read_csv(r / name / "daily.csv.gz", parse_dates=["date"]).set_index("date").net_return
                for r in (base, variant))
        common = a.index.intersection(b.index)
        stats = paired_block_interval(b.loc[common], a.loc[common], samples=config["bootstrap_samples"],
                                      block=config["bootstrap_block"], seed=config["seed"])
        sa, sb = summaries[0].loc[key], summaries[1].loc[key]
        row = dict(left=name, right=f"base:{name}", **stats,
                   base_annualized_return=float(sa.annualized_return), variant_annualized_return=float(sb.annualized_return),
                   base_mean_two_way_turnover=float(sa.mean_two_way_turnover),
                   variant_mean_two_way_turnover=float(sb.mean_two_way_turnover),
                   base_total_costs=float(sa[COSTS].sum()), variant_total_costs=float(sb[COSTS].sum()))
        for column in ("beta", "active_annualized_return", "information_ratio"):
            if column in sa and column in sb:
                row[f"base_{column}"], row[f"variant_{column}"] = float(sa[column]), float(sb[column])
        rows.append(row)
    family_holm(rows)
    return {"base": str(base.resolve()), "variant": str(variant.resolve()),
            "same_predictions": manifests[0].get("prediction_hash") == manifests[1].get("prediction_hash"),
            "same_data": manifests[0].get("data_prefix_hash") == manifests[1].get("data_prefix_hash"),
            "comparisons": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("variant")
    parser.add_argument("output")
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()
    report = compare_runs(args.base, args.variant, args.samples or None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame(report["comparisons"]).set_index("left")
    columns = ["base_annualized_return", "variant_annualized_return", "annualized_mean_difference", "ci95_low",
               "ci95_high", "p_bootstrap", "holm_reject_5pct", "base_mean_two_way_turnover", "variant_mean_two_way_turnover"]
    print(f"same predictions: {report['same_predictions']}, same data: {report['same_data']}")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(table[columns].round(4).to_string())


if __name__ == "__main__":
    main()
