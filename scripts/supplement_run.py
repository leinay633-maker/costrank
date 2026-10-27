"""Post-freeze supplementary out-of-sample run (RESEARCH section 11). Offline.

usage: python scripts/supplement_run.py --freeze <freeze record> --data <schema-3 snapshot>
           --start 2026-01-01 --end 2026-10-09 --output <runs dir> [--allow-proxy] [--workers 8]

The frozen configuration over a period from the declared holdout on, along the holdout path of
experiment.run: one fixed train/validation split before the period, no refit inside it, every
model x policy x fee scenario on one ledger from cash. Config refuses a holdout ending after 2025
(rules review), so the period goes to the folds (forward.forward_folds with a single block) and to
the ledger (simulate's period) instead of the config; the protocol code must hash as frozen. Over the
holdout's own period this reproduces the opened run. The run directory has experiment.run's layout
with phase "supplement", so the audit and report scripts read it.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid
import numpy as np
import pandas as pd
from costrank.backtest import execution_views, simulate
from costrank.data import canonical_hash, read_bundle, sha256
from costrank.experiment import (MVO_POLICIES, benchmark_returns, clean, code_hash, config_hash, exposures,
                                 family_holm, git_info, protocol_code_hash, style_ranks, write_json)
from costrank.features import prepare
from costrank.forward import forward_folds, frozen
from costrank.metrics import paired_block_interval, performance, rank_ic, relative
from costrank.models import predict_oos

_STATE = {}


def _simulate_task(key):
    """Worker entry: state is inherited from the parent through fork, never pickled."""
    st = _STATE
    model, policy, scenario, bps, zero = key
    return key, simulate(st["panel"], st["signals"][model], st["bundle"], st["config"], policy,
                         slippage_bps=bps, zero_cost=zero, views=st["views"], period=st["period"])


def supplement(freeze_file, data, output, start, end, *, allow_proxy=False, workers=1):
    config, record = frozen(freeze_file)
    if (config.feature_set == "v04" or MVO_POLICIES & set(config.policies) or config.fill_price != "open"
            or config.participation_basis != "adv20"):
        raise ValueError("Only the frozen v0.2-feature, open-fill path is covered")
    if not config.holdout_start <= start <= end:
        raise ValueError("The period must start on or after the declared holdout")
    packages = {p: metadata.version(p) for p in record["packages"]}
    if packages != record["packages"]:
        raise ValueError("Dependency versions differ from the frozen development environment")
    started = time.perf_counter()
    root = Path(output) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "phase": "supplement", "period": {"start": start, "end": end},
        "declaration": "RESEARCH section 11", "created_at": datetime.now(timezone.utc).isoformat(),
        "code_hash": code_hash(), "protocol_code_hash": protocol_code_hash(), "config_hash": config_hash(config),
        "config": config.dict(), "git": git_info(), "data_path": str(Path(data).resolve()), "python": sys.version,
        "platform": platform.platform(), "packages": packages,
        "freeze": {"path": str(Path(freeze_file).resolve()), "sha256": sha256(freeze_file),
                   **{k: record.get(k) for k in ("frozen_at", "code_hash", "protocol_code_hash", "config_hash")}}}
    write_json(root / "manifest.json", manifest, exclusive=True)
    def event(kind, **details):
        with (root / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean(dict(time=datetime.now(timezone.utc).isoformat(), kind=kind, **details)))+"\n")
    try:
        bundle = read_bundle(data)
        if bundle.metadata["corporate_actions"] != "explicit_events" and not allow_proxy:
            raise ValueError("Free-data corporate-action proxy requires explicit --allow-proxy acknowledgement")
        if bundle.metadata["end"] < end or bundle.calendar[-1] < pd.Timestamp(end)-pd.Timedelta(days=14):
            raise ValueError("Snapshot does not cover the requested end date")
        if bundle.metadata.get("partial_assembly"):
            raise ValueError("A partial preview snapshot is for 2019-and-earlier engineering debugging only")
        manifest["data_metadata"] = bundle.metadata
        manifest["data_prefix_hash"] = bundle.prefix_hash(config.holdout_start)
        manifest["development_prefix_matches"] = manifest["data_prefix_hash"] == record.get("development_prefix_hash")
        manifest["evidence"] = "synthetic_only" if bundle.metadata["source"] == "SYNTHETIC" else "historical_exploratory_unverified_source"
        write_json(root / "manifest.json", manifest)
        panel = prepare(bundle, config)
        sessions = bundle.calendar[(bundle.calendar >= start) & (bundle.calendar <= end)]
        blocks = forward_folds(bundle.calendar, config, start, end, refit_every=max(len(sessions), 1))
        predictions, audits = predict_oos(panel, bundle.calendar, config, fold_list=blocks,
                                          on_fold=lambda audit: event("fold_complete", **audit))
        predictions.to_csv(root / "predictions.csv.gz", index=False)
        write_json(root / "folds.json", audits)
        manifest["prediction_hash"] = canonical_hash(predictions)
        ic = rank_ic(predictions, panel)
        ic.to_csv(root / "rank_ic.csv", index=False)
        coverage = panel.groupby("date").agg(observed_names=("symbol", "size"), eligible_names=("eligible", "sum"))
        coverage.to_csv(root / "universe_coverage.csv")
        bench = benchmark_returns(panel, start, end, bundle)
        bench.to_csv(root / "benchmark.csv")
        ranks = style_ranks(panel)
        results, return_series, details, years = [], {}, [], []
        views = execution_views(panel, start, end)
        scenarios = [("zero_all_costs", 0.0, True)] + [(f"slip_{x:g}bps", float(x), False) for x in config.slippage_grid_bps]
        keys = [(model, policy, scenario, bps, zero) for model in config.models for policy in config.policies
                for scenario, bps, zero in scenarios]
        _STATE.update(panel=panel, bundle=bundle, config=config, views=views, period=(start, end),
                      signals={m: predictions[predictions.model == m] for m in config.models})
        pool = None
        if workers > 1 and sys.platform != "win32":
            import multiprocessing
            pool = multiprocessing.get_context("fork").Pool(workers)
        try:
            stream = pool.imap(_simulate_task, keys) if pool else map(_simulate_task, keys)
            for (model, policy, scenario, _, _), out in stream:
                name = f"{model}__{policy}__{scenario}"
                target = root / name
                target.mkdir()
                for key, frame in out.items():
                    frame.to_csv(target / f"{key}.csv.gz", index=False)
                d = out["daily"]
                r = d.set_index("date").net_return
                results.append(dict(model=model, policy=policy, scenario=scenario, **performance(d),
                                    **relative(r, bench.benchmark_return), **exposures(out["holdings"], ranks)))
                return_series[(model, policy, scenario)] = r
                for audit in audits:
                    part = d[(d.date >= audit["test_start"]) & (d.date <= audit["test_end"])]
                    if not part.empty:
                        details.append(dict(model=model, policy=policy, scenario=scenario, fold=audit["number"],
                                            **performance(part), **relative(part.set_index("date").net_return, bench.benchmark_return)))
                for year, part in d.groupby(d.date.dt.year):
                    years.append(dict(model=model, policy=policy, scenario=scenario, year=int(year),
                                      **performance(part), **relative(part.set_index("date").net_return, bench.benchmark_return)))
                event("backtest_complete", model=model, policy=policy, scenario=scenario)
        finally:
            if pool:
                pool.close()
                pool.join()
            _STATE.clear()
        pd.DataFrame(results).to_csv(root / "summary.csv", index=False)
        pd.DataFrame(details).to_csv(root / "by_fold.csv", index=False)
        pd.DataFrame(years).to_csv(root / "by_year.csv", index=False)
        comparisons = []
        for key, series in return_series.items():
            model, policy, scenario = key
            references = []
            if policy in {"penalty", "turnover", "cost"} and "risk" in config.policies:
                references.append((model, "risk", scenario))
            if policy == "cost":
                references.extend((model, arm, scenario) for arm in ("penalty", "turnover") if arm in config.policies)
            if model == "lightgbm":
                references.extend((baseline, policy, scenario) for baseline in ("factor", "ridge") if baseline in config.models)
            for reference in references:
                stats = paired_block_interval(series, return_series[reference], samples=config.bootstrap_samples,
                    block=config.bootstrap_block, seed=config.seed)
                comparisons.append(dict(left="__".join(key), right="__".join(reference), **stats))
        family_holm(comparisons)
        write_json(root / "paired_comparisons.json", comparisons)
        against = []
        for key, series in return_series.items():
            common = series.index.intersection(bench.index)
            stats = paired_block_interval(series.loc[common], bench.benchmark_return.loc[common],
                samples=config.bootstrap_samples, block=config.bootstrap_block, seed=config.seed)
            against.append(dict(left="__".join(key), right="benchmark_eligible_ew", **stats))
        family_holm(against)
        write_json(root / "benchmark_comparisons.json", against)
        pct = lambda value: f"{100*value:.2f}%" if np.isfinite(value) else "NA"
        report = ["# Supplementary out-of-sample run (RESEARCH section 11)", "",
                  f"Evidence: **{manifest['evidence']}**. Period {start} to {end}, frozen configuration, one fixed "
                  "train/validation split before the period. Decided after the freeze: not the declared holdout and "
                  "not forward evidence; descriptive only.", "",
                  "| Model | Policy | Scenario | Annual return | Sharpe (rf=0) | Max DD | Mean 2-way turnover | Beta | IR vs pool |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for x in results:
            report.append(f"| {x['model']} | {x['policy']} | {x['scenario']} | {pct(x['annualized_return'])} | "
                          f"{x['sharpe_rf0']:.3f} | {pct(x['max_drawdown'])} | {pct(x['mean_two_way_turnover'])} | "
                          f"{x.get('beta', np.nan):.2f} | {x.get('information_ratio', np.nan):.2f} |")
        (root / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        manifest.update(status="complete", wall_seconds=time.perf_counter()-started,
                        result_files={str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"})
        write_json(root / "manifest.json", manifest)
        return root
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc), wall_seconds=time.perf_counter()-started)
        write_json(root / "manifest.json", manifest)
        event("failed", error_type=type(exc).__name__, error=str(exc))
        raise RuntimeError(f"Run failed; retained evidence at {root}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-proxy", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel scenario processes (Linux fork only)")
    args = parser.parse_args()
    print(supplement(args.freeze, args.data, args.output, args.start, args.end,
                     allow_proxy=args.allow_proxy, workers=args.workers))


if __name__ == "__main__":
    main()
