"""v0.4 decision exactly as declared in RESEARCH section 9. Offline.

usage: python scripts/compare_v04.py --s2p2 RUN --s4p2 RUN --s2p4 RUN --s4p4 RUN
           [--kappa RUN ...] [--capacity RUN ...] [--registry registry.csv]
           [--v02-run RUN] [--snapshot SNAPSHOT] --output decision.json

Information ratios are on daily active returns against `invested` x the eligible-pool equal-weight
benchmark (summary column b95_information_ratio). The single primary test compares LightGBM S4P4
(mvo) with LightGBM S2P2 (cost) at slip_10bps by the Ledoit-Wolf studentized block bootstrap.
The report-only extras (full matrix, turnover and costs, per-year ratios, factor selection
frequency, the S2P2 vs v0.2 paired difference and CSI 500/1000 beta-hedged returns) never enter
the decision.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.metrics import deflated_sharpe, minimum_detectable, pbo_cscv, sharpe_difference_test

PRIMARY = "slip_10bps"
# The four cells as section 9 declares them (feature set, policy); everything else at the defaults.
CELLS = {"S2P2": ("v02", "cost"), "S4P2": ("v04", "cost"), "S2P4": ("v02", "mvo"), "S4P4": ("v04", "mvo")}
DECLARED = ("dev_start", "dev_end", "invested", "initial_cash", "fill_price", "participation_basis",
            "impact_coefficient", "max_weight", "participation")


def load(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{run} is not a complete run")
    bench = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    return run, manifest, bench


def active(run, model, policy, scenario=PRIMARY):
    run, manifest, bench = load(run)
    daily = pd.read_csv(run / f"{model}__{policy}__{scenario}" / "daily.csv.gz", parse_dates=["date"]).set_index("date")
    invested = manifest["config"]["invested"]
    common = daily.index.intersection(bench.index)
    return (daily.net_return.loc[common] - invested * bench.loc[common]), daily


def ratio(a, minimum_te=0.02):
    """Section 9: in the decision, a side with annual tracking error under 2% has no information
    ratio. Report-only figures (years, periods, kappa, capacity) pass minimum_te=0 (supplement 10)."""
    te = a.std(ddof=1) * np.sqrt(252)
    return {"information_ratio": float(a.mean() * 252 / te) if te >= minimum_te else np.nan,
            "tracking_error": float(te), "active_return_arith": float(a.mean() * 252)}


def at_least(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a >= b)


def fallback_share(run, model, policy, scenario=PRIMARY):
    """Rebalance days on which the declared risk limits could not be met: `hold_fallback`, or a
    solution only after widening them (RESEARCH section 9 supplement 8)."""
    path = Path(run) / f"{model}__{policy}__{scenario}" / "allocations.csv.gz"
    alloc = pd.read_csv(path)
    if not len(alloc):
        return np.nan
    widened = (alloc.status.eq("optimal") & alloc.risk_limit_scale.fillna(1.0).gt(1.0)
               if "risk_limit_scale" in alloc else False)
    return float((alloc.status.eq("hold_fallback") | widened).mean())


COSTS = ["commission", "stamp", "transfer", "slippage", "dividend_tax"]
HEDGES = {"csi500": "sh.000905", "csi1000": "sh.000852"}


def matrix(runs):
    """Report only: every model x policy x scenario of the four cells, and the net Sharpe and
    information-ratio differences S4 - S2 (same model, policy, scenario) and P4 - P2-cost. The ratio
    is recomputed from each ledger: S2P2 runs (v0.2 features, P2 policies) carry no b95 columns in
    their summary."""
    rows = []
    for cell, run in runs.items():
        s = pd.read_csv(Path(run) / "summary.csv")
        for r in s.itertuples():
            a, _ = active(run, r.model, r.policy, r.scenario)
            rows.append(dict(cell=cell, model=r.model, policy=r.policy, scenario=r.scenario, sharpe=r.sharpe_rf0,
                             information_ratio=ratio(a, 0)["information_ratio"],
                             annualized_return=r.annualized_return, turnover=r.mean_two_way_turnover))
    table = pd.DataFrame(rows)
    key = ["model", "policy", "scenario"]
    signal = []
    for low, high in (("S2P2", "S4P2"), ("S2P4", "S4P4")):
        a = table[table.cell == low].set_index(key)
        b = table[table.cell == high].set_index(key)
        common = a.index.intersection(b.index)
        for k in common:
            signal.append(dict(zip(key, k), pair=f"{high}-{low}", sharpe=b.loc[k, "sharpe"] - a.loc[k, "sharpe"],
                               information_ratio=b.loc[k, "information_ratio"] - a.loc[k, "information_ratio"]))
    portfolio = []
    for low, high in (("S2P2", "S2P4"), ("S4P2", "S4P4")):
        a = table[(table.cell == low) & (table.policy == "cost")].set_index(["model", "scenario"])
        b = table[(table.cell == high) & (table.policy == "mvo")].set_index(["model", "scenario"])
        for k in a.index.intersection(b.index):
            portfolio.append(dict(model=k[0], scenario=k[1], pair=f"{high}mvo-{low}cost",
                                  sharpe=b.loc[k, "sharpe"] - a.loc[k, "sharpe"],
                                  information_ratio=b.loc[k, "information_ratio"] - a.loc[k, "information_ratio"]))
    return {"table": table.to_dict("records"), "signal_layer": signal, "portfolio_layer": portfolio}


def costs_and_years(runs, cell_policy):
    """Report only: turnover, annual cost drag and per-year information ratios of the 8 main curves."""
    out = {}
    for c, run in runs.items():
        for m in ("lightgbm", "ridge"):
            a, daily = active(run, m, cell_policy[c])
            drag = daily[[k for k in COSTS if k in daily]].sum(axis=1) / daily.nav.shift().fillna(daily.nav.iloc[0])
            years = {int(y): ratio(part, 0)["information_ratio"] for y, part in a.groupby(a.index.year)}
            out[f"{c}/{m}"] = {"mean_two_way_turnover": float(daily.turnover_two_way.mean()),
                               "annual_cost_drag": float(drag.mean() * 252),
                               "information_ratio_by_year": years}
    return out


def hedge(r, x, window=120, minimum=60):
    """r minus a trailing beta times x; the beta for session t uses sessions before t only."""
    frame = pd.DataFrame({"r": r, "x": x}).dropna()
    cov = frame.r.rolling(window, min_periods=minimum).cov(frame.x)
    beta = (cov / frame.x.rolling(window, min_periods=minimum).var()).shift()
    return (frame.r - beta * frame.x).dropna(), beta


def hedged(runs, cell_policy, snapshot):
    """Report only: net returns hedged with the index at a trailing beta, no futures basis, no
    hedge cost."""
    from costrank.data import read_bundle
    bundle = read_bundle(snapshot, allow_after="2025-01-01")
    idx = bundle.tables.get("indices")
    out = {}
    for name, code in HEDGES.items():
        series = idx[idx["index"] == code].set_index("date").close.sort_index().pct_change() if idx is not None else None
        if series is None or series.dropna().empty:
            out[name] = "index not in snapshot"
            continue
        for c, run in runs.items():
            for m in ("lightgbm", "ridge"):
                _, daily = active(run, m, cell_policy[c])
                h, beta = hedge(daily.net_return, series)
                sd = h.std(ddof=1)
                out.setdefault(name, {})[f"{c}/{m}"] = {
                    "sessions": int(len(h)), "mean_beta": float(beta.mean()),
                    "hedged_annual_return": float(h.mean() * 252),
                    "hedged_sharpe": float(h.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan}
    return out


def versus_v02(run_s2p2, run_v02, samples):
    """Report only: S2P2 on the full-A U4 pool against the v0.2 main-board run, same model and the
    cost policy at 10 bps, as v0.2's paired annualized return difference (block bootstrap)."""
    from costrank.metrics import paired_block_interval
    out = {}
    for m in ("lightgbm", "ridge"):
        series = []
        for run in (run_s2p2, run_v02):
            d = pd.read_csv(Path(run) / f"{m}__cost__{PRIMARY}" / "daily.csv.gz", parse_dates=["date"])
            series.append(d.set_index("date").net_return)
        common = series[0].index.intersection(series[1].index)
        out[m] = paired_block_interval(series[0].loc[common], series[1].loc[common], samples=samples, block=20, seed=42)
        out[m]["method"] = "paired circular block bootstrap of daily net-return differences; reported alone, no Holm"
    return out


def check_declared(runs):
    """The four cells must be the declared section 9 runs: same data, the declared feature set and
    policy, LightGBM and Ridge, 10 bps, and every other setting at the declared default."""
    default = Config()
    for cell, run in runs.items():
        cfg = load(run)[1]["config"]
        features, policy = CELLS[cell]
        wrong = [k for k in DECLARED if cfg.get(k) != getattr(default, k)]
        wrong += [] if cfg.get("feature_set") == features else ["feature_set"]
        wrong += [] if policy in cfg.get("policies", []) else ["policies"]
        wrong += [] if {"lightgbm", "ridge"} <= set(cfg.get("models", [])) else ["models"]
        wrong += [] if 10.0 in cfg.get("slippage_grid_bps", []) else ["slippage_grid_bps"]
        if wrong:
            raise ValueError(f"{cell} run {run} is not the declared configuration: {sorted(wrong)}")


def primary_passes(test):
    """Condition 1 (supplement 10): two-sided p < 0.05 and the S4P4 - S2P2 difference positive; an
    undefined test (a series without variance) does not pass."""
    p, diff = test["p_value"], test["sharpe_difference_annual"]
    return bool(np.isfinite(p) and np.isfinite(diff) and p < 0.05 and diff > 0)


def first_quarter_2024(left, right):
    """Section 9: does a positive full-sample difference come mainly from 2024 Q1? Declared in
    supplement 10 as: positive overall, and not positive once 2024 Q1 is left out (no gate on TE)."""
    q1 = (left.index >= "2024-01-01") & (left.index < "2024-04-01")
    diff = lambda mask: ratio(left[mask], 0)["information_ratio"] - ratio(right[mask], 0)["information_ratio"]
    full, rest, inside = diff(np.ones(len(left), bool)), diff(~q1), diff(q1) if q1.sum() > 1 else np.nan
    return {"full": full, "excluding_2024q1": rest, "2024q1": inside,
            "mainly_2024q1": bool(np.isfinite(full) and full > 0 and np.isfinite(rest) and rest <= 0)}


def kappa_rows(runs):
    """Report only: the kappa frontier, S4P4 ridge and lightgbm at 10 bps, no TE gate."""
    rows = []
    for run in runs:
        _, manifest, _ = load(run)
        for model in ("lightgbm", "ridge"):
            a, daily = active(run, model, "mvo")
            rows.append(dict(kappa_scale=manifest["config"]["kappa_scale"], model=model,
                             turnover=float(daily.turnover_two_way.mean()), **ratio(a, 0)))
    return rows


def capacity_table(runs):
    """Report only: every capacity run at zero slippage, no TE gate, with its infeasible share."""
    rows = []
    for run in runs:
        cfg = load(run)[1]["config"]
        for model in ("lightgbm", "ridge"):
            a, _ = active(run, model, "mvo", "slip_0bps")
            rows.append(dict(cell="S4" if cfg["feature_set"] == "v04" else "S2", model=model,
                             capital=cfg["initial_cash"], fallback=fallback_share(run, model, "mvo", "slip_0bps"),
                             **ratio(a, 0)))
    return pd.DataFrame(rows).sort_values(["cell", "model", "capital"])


def capacity(table):
    """First capital, low to high, that is infeasible (fallback or widened > 20%) or whose
    information ratio is at most half the 1e6 one; undefined if the 1e6 ratio is not positive."""
    out = {}
    for (c, m), g in table.groupby(["cell", "model"]):
        g = g.set_index("capital").sort_index()
        base = g.loc[1e6, "information_ratio"] if 1e6 in g.index else np.nan
        if not np.isfinite(base) or base <= 0:
            out[f"{c}/{m}"] = "undefined"
            continue
        hit = [cap for cap, r in g.iterrows() if r.fallback > 0.20 or not (r.information_ratio > base / 2)]
        out[f"{c}/{m}"] = min(hit) if hit else f"above {g.index.max():g}"
    return out


def selection_frequency(run):
    folds = json.loads((Path(run) / "folds.json").read_text(encoding="utf-8"))
    counts, fallback = {}, 0
    for f in folds:
        s = f.get("selection", {})
        fallback += bool(s.get("fallback_core"))
        for name in s.get("selected", []):
            counts[name] = counts.get(name, 0) + 1
    return {"folds": len(folds), "fallback_core_folds": fallback,
            "passed_fdr_by_fold": [f.get("selection", {}).get("passed_fdr") for f in folds],
            "selected_folds": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))}


def decide(args):
    runs = {"S2P2": args.s2p2, "S4P2": args.s4p2, "S2P4": args.s2p4, "S4P4": args.s4p4}
    hashes = {k: load(v)[1].get("data_prefix_hash") for k, v in runs.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError(f"Cells use different data: {hashes}")
    if getattr(args, "check_declared", True):
        check_declared(runs)
    cell = {c: policy for c, (_, policy) in CELLS.items()}
    # Every decision figure on the sessions all eight curves share.
    series = {(c, m): active(runs[c], m, cell[c])[0] for c in runs for m in ("lightgbm", "ridge")}
    common = sorted(set.intersection(*(set(s.index) for s in series.values())))
    series = {k: s.loc[common] for k, s in series.items()}
    stats = {k: ratio(s) for k, s in series.items()}
    left, right = series[("S4P4", "lightgbm")], series[("S2P2", "lightgbm")]
    common = left.index
    test = sharpe_difference_test(left.to_numpy(), right.to_numpy(), block=20, samples=args.samples, seed=42, bandwidth=20)
    ridge_diff = stats[("S4P4", "ridge")]["information_ratio"] - stats[("S2P2", "ridge")]["information_ratio"]  # NaN if either is missing
    fallback = fallback_share(runs["S4P4"], "lightgbm", "mvo")
    lg = lambda c: stats[(c, "lightgbm")]
    te_ok = all(lg(c)["tracking_error"] >= 0.02 for c in ("S4P4", "S2P2"))
    conditions = {
        "primary_p_below_0.05": primary_passes(test),
        "ridge_point_estimate_positive": bool(np.isfinite(ridge_diff) and ridge_diff > 0),
        "s4p4_not_below_s2p4_and_s4p2_cost": at_least(lg("S4P4")["information_ratio"], lg("S2P4")["information_ratio"])
                                             and at_least(lg("S4P4")["information_ratio"], lg("S4P2")["information_ratio"]),
        "s4p4_fallback_at_most_5pct": bool(fallback <= 0.05),
        "tracking_errors_at_least_2pct": bool(te_ok)}
    frozen = "lightgbm+S4+mvo" if all(conditions.values()) else "lightgbm+S2+cost"
    split = {}
    for name, part in (("before_2024", common[common < "2024-01-01"]), ("from_2024", common[common >= "2024-01-01"])):
        split[name] = ratio(left.loc[part], 0)["information_ratio"] - ratio(right.loc[part], 0)["information_ratio"]
    attribution = {}
    for model in ("lightgbm", "ridge"):
        ir = lambda c: stats[(c, model)]["information_ratio"]
        attribution[model] = {"S4P4-S2P4": ir("S4P4") - ir("S2P4"), "S4P2-S2P2": ir("S4P2") - ir("S2P2"),
                              "S2P4-S2P2": ir("S2P4") - ir("S2P2"), "S4P4-S4P2_package": ir("S4P4") - ir("S4P2")}
    # PBO on the eight daily active-return curves (supplement 10); the DSR below uses net returns.
    curves_matrix = pd.concat({f"{c}-{m}": s for (c, m), s in series.items()}, axis=1).dropna()
    report = {"primary_test": test, "minimum_detectable_ir_difference": minimum_detectable(test["se_annual"]),
              "ridge_ir_difference": ridge_diff, "s4p4_fallback_share": fallback, "conditions": conditions,
              "frozen_protocol": frozen, "ir_difference_by_period": split,
              "first_quarter_2024": first_quarter_2024(left, right),
              "cells": {f"{c}/{m}": v for (c, m), v in stats.items()}, "attribution": attribution,
              "pbo": pbo_cscv(curves_matrix.to_numpy(), segments=16)}
    # Report only (section 9), never part of the decision above.
    report["matrix"] = matrix(runs)
    report["costs_and_years"] = costs_and_years(runs, cell)
    report["selection_frequency"] = selection_frequency(runs["S4P4"])
    if getattr(args, "v02_run", None):
        report["versus_v02"] = versus_v02(runs["S2P2"], args.v02_run, args.samples)
    if getattr(args, "snapshot", None):
        report["hedged"] = hedged(runs, cell, args.snapshot)
    if args.kappa:
        # The declared kappa (scale 1) is the S4P4 run itself.
        report["kappa_frontier"] = kappa_rows([runs["S4P4"]] + list(args.kappa))
    if args.capacity:
        table = capacity_table(args.capacity)
        report["capacity_table"], report["capacity"] = table.to_dict("records"), capacity(table)
    if args.registry:
        registry = pd.read_csv(args.registry)
        sharpe = registry.sharpe_daily.dropna()
        frozen_run, model, policy = (runs["S4P4"], "lightgbm", "mvo") if frozen.endswith("mvo") else (runs["S2P2"], "lightgbm", "cost")
        _, daily = active(frozen_run, model, policy)
        report["deflated_sharpe"] = deflated_sharpe(daily.net_return.to_numpy(), len(registry), float(sharpe.var(ddof=1)))
        report["deflated_sharpe"]["returns"] = "daily net returns of the frozen configuration"
    report["provenance"] = {"cells": {c: provenance(r) for c, r in runs.items()},
                            "kappa": [provenance(r) for r in args.kappa],
                            "capacity": [provenance(r) for r in args.capacity],
                            "v02_run": getattr(args, "v02_run", None), "registry": args.registry,
                            "snapshot": getattr(args, "snapshot", None)}
    return report


def provenance(run):
    """Where a number came from: the run directory and the hashes its manifest binds."""
    _, manifest, _ = load(run)
    return {"run": str(Path(run).resolve()), **{k: manifest.get(k) for k in
            ("code_hash", "config_hash", "data_prefix_hash", "prediction_hash", "wall_seconds")}}


def main():
    parser = argparse.ArgumentParser()
    for cell in ("s2p2", "s4p2", "s2p4", "s4p4"):
        parser.add_argument(f"--{cell}", required=True)
    parser.add_argument("--kappa", nargs="*", default=[])
    parser.add_argument("--capacity", nargs="*", default=[])
    parser.add_argument("--registry")
    parser.add_argument("--v02-run", help="v0.2 main-board development run (report-only paired difference)")
    parser.add_argument("--snapshot", help="schema-3 snapshot with CSI 500/1000 bars (report-only beta hedge)")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = decide(args)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("frozen_protocol", "conditions", "ridge_ir_difference",
                                             "minimum_detectable_ir_difference")}, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(report["primary_test"], indent=2))


if __name__ == "__main__":
    main()
