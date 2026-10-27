"""Immutable run directories, provenance, final-holdout gate, full result matrix."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import platform
import sys
import time
import uuid
import numpy as np
import pandas as pd
from .backtest import execution_views, simulate
from .data import canonical_hash, read_bundle, sha256
from .features import prepare
from .metrics import holm, paired_block_interval, performance, rank_ic, relative
from .models import predict_oos

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm"]
# Recorded only when a run uses them, so v0.1-v0.3 manifests keep their package list.
V04_PACKAGES = ["clarabel", "pyarrow"]
MVO_POLICIES = {"mvo", "mvo_rank"}
_TASK_STATE = {}


def _simulate_task(key):
    """Worker entry: state is inherited from the parent through fork, never pickled."""
    st = _TASK_STATE
    model, policy, scenario, bps, zero = key
    planner = st["planners"].get((model, policy))
    return key, simulate(st["panel"], st["signals"][model], st["bundle"], st["config"], policy,
                         slippage_bps=bps, zero_cost=zero, holdout=st["holdout"], views=st["views"],
                         planner=planner)


def first30_adv(intraday, panel, window=20):
    """Per session: trailing `window`-session mean of each name's first-30-minute amount, sessions
    without a bar counting as zero; names lack a value until `window` sessions of minute data exist."""
    wide = intraday.pivot(index="date", columns="symbol", values="first_amount")
    sessions = pd.DatetimeIndex(sorted(panel.date.unique()))
    wide = wide.reindex(index=sessions[sessions >= wide.index.min()]).fillna(0.0)
    mean = wide.rolling(window, min_periods=window).mean()
    return {day: row.dropna() for day, row in mean.iterrows()}


def benchmark_returns(panel, start, end, bundle):
    """Equal weight over the previous close's eligible pool, close to close, daily
    rebalanced, no costs: the pool the strategies choose from, not an investable index.
    A name removed before the next close counts its outcome, as the ledger does: zero,
    its cash, or its exchange shares at the acquirer's close (carried if not yet trading)."""
    p = panel.sort_values(["symbol", "date"])
    x = pd.DataFrame({"date": p.date, "r": p.groupby("symbol").tr_close.pct_change(),
                      "was": p.groupby("symbol").eligible.shift(fill_value=False).astype(bool)})
    x = x[x.was & x.r.notna()][["date", "r"]]
    last = p.groupby("symbol").tail(1).set_index("symbol")
    removal = bundle.instruments.set_index("symbol").delisted.reindex(last.index)
    gone = last[last.eligible & removal.notna()].assign(date=removal)
    res = bundle.delistings.set_index("symbol").reindex(gone.index)
    closes = p.set_index(["date", "symbol"]).close
    acquirer_close = closes.reindex(pd.MultiIndex.from_arrays([gone.date, res.acquirer])).to_numpy()
    value = np.where(res.kind.eq("cash"), res.cash_per_share,
                     np.where(res.kind.eq("share_exchange"),
                              np.where(np.isfinite(acquirer_close), res.ratio * acquirer_close, gone.close), 0.0))
    x = pd.concat([x, pd.DataFrame({"date": gone.date.to_numpy(), "r": value / gone.close.to_numpy() - 1})])
    x = x[(x.date >= start) & (x.date <= end)]
    g = x.groupby("date").r
    return pd.DataFrame({"benchmark_return": g.mean(), "names": g.size()})


def family_holm(items):
    """Holm within each cost scenario: the same signal under five fee paths gives highly
    dependent p-values, so one 125-test family would be needlessly conservative."""
    by_scenario = {}
    for item in items:
        by_scenario.setdefault(item["left"].rsplit("__", 1)[1], []).append(item)
    for family in by_scenario.values():
        for item, flag in zip(family, holm([c["p_bootstrap"] for c in family])):
            item["holm_family"] = len(family)
            item["holm_reject_5pct"] = bool(flag)


def style_ranks(panel):
    """Same-day percentile of liquidity, volatility and price across all observed names."""
    by = panel.groupby("date")
    return pd.DataFrame({"adv_pct": by.adv20.rank(pct=True), "vol_pct": by.vol20.rank(pct=True),
                         "price_pct": by.close.rank(pct=True)}).set_index(
        pd.MultiIndex.from_frame(panel[["date", "symbol"]]))


def exposures(holdings, ranks):
    if holdings.empty:
        return {}
    h = holdings.join(ranks, on=["date", "symbol"])
    w = h.weight / h.groupby("date").weight.transform("sum")
    out = {f"holding_{c}": float((h[c].fillna(0.5) * w).groupby(h.date).sum().mean())
           for c in ("adv_pct", "vol_pct", "price_pct")}
    out["mean_holdings"] = float(h.groupby("date").size().mean())
    return out


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def config_hash(config):
    return hashlib.sha256(json.dumps(config.dict(), sort_keys=True).encode()).hexdigest()


def code_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# Operational modules that may be fixed during forward paper trading without changing the protocol.
OPERATIONAL = {"live.py", "cli.py"}


def protocol_code_hash():
    """code_hash() without the operational modules: data, signals, models, portfolio and ledger."""
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        if path.name not in OPERATIONAL:
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def git_info():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    try:
        sha = git("rev-parse", "HEAD")
        if sha:
            return {"sha": sha, "dirty": bool(git("status", "--porcelain")), "origin": "git"}
        packed = root / "SOURCE.json"
        if packed.exists():
            record = json.loads(packed.read_text(encoding="utf-8"))
            dirty = False
            for name, expected in record["files"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
                    dirty = True
            return {"sha": record["git_sha"], "dirty": dirty, "origin": "source_archive"}
        return {"sha": None, "dirty": None, "origin": "unversioned"}
    except (OSError, ValueError, KeyError):
        return {"sha": None, "dirty": None, "origin": "invalid_source_manifest"}


def check_freeze(path, config, bundle=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("config_hash") != config_hash(config) or record.get("code_hash") != code_hash():
        raise ValueError("Frozen config/source mismatch: fork a NEW study; do not retune this holdout")
    current = {p: metadata.version(p) for p in record.get("packages", {})}
    if current != record.get("packages", {}):
        raise ValueError("Dependency versions differ from frozen development environment")
    if bundle and bundle.prefix_hash(config.holdout_start) != record["development_prefix_hash"]:
        raise ValueError("Historical data changed since freeze; cannot claim the same final holdout")
    return record


def opened_holdouts(directory, config):
    """Receipts in a protocol directory of openings of a holdout overlapping `config`'s. A receipt
    whose freeze record is gone or unreadable still counts."""
    found = []
    for receipt in sorted(Path(directory).glob("*.opened.json")):
        try:
            other = json.loads(receipt.with_name(receipt.name[:-len(".opened.json")]).read_text(encoding="utf-8"))["config"]
            overlaps = other["holdout_start"] <= config.holdout_end and config.holdout_start <= other["holdout_end"]
        except (OSError, ValueError, KeyError, TypeError):
            overlaps = True
        if overlaps:
            found.append({"receipt": str(receipt.resolve()), **json.loads(receipt.read_text(encoding="utf-8"))})
    return found


def freeze(output, config, data, development_run):
    run = Path(development_run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    bundle = read_bundle(data, allow_after=config.holdout_start)
    if bundle.metadata.get("sample_limit", 0) or bundle.metadata.get("partial_assembly"):
        raise ValueError("Do not freeze a sampled pilot or a partial preview; use the full declared snapshot")
    for name, expected in manifest.get("result_files", {}).items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or sha256(path) != expected:
            raise ValueError(f"Development artifact changed: {name}")
    if manifest.get("status") != "complete" or manifest.get("phase") != "development":
        raise ValueError("Freeze requires a completed DEVELOPMENT run")
    if manifest["config_hash"] != config_hash(config) or manifest["code_hash"] != code_hash():
        raise ValueError("Run/config/code differ; rerun development before freezing")
    if manifest["data_prefix_hash"] != bundle.prefix_hash(config.holdout_start):
        raise ValueError("Development run used different data")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"frozen_at": datetime.now(timezone.utc).isoformat(),
        "code_hash": code_hash(), "protocol_code_hash": protocol_code_hash(), "config_hash": config_hash(config),
        "config": config.dict(), "packages": manifest["packages"], "development_prefix_hash": bundle.prefix_hash(config.holdout_start),
        "development_run": str(run.resolve()), "development_manifest_hash": sha256(run / "manifest.json"),
        # A record written after its holdout was opened (for forward trading) can never open it again.
        "holdout_opened_before": opened_holdouts(path.parent, config),
        "final_protocol": "One fixed train/validation split before holdout; no refit inside holdout",
        "note": "Filesystem guard, not a cryptographic prevention of human peeking or file deletion"}
    write_json(path, record, exclusive=True)
    return path


def run(config, data, output, *, holdout=False, freeze_file=None, allow_proxy=False, workers=1):
    started = time.perf_counter()
    root = Path(output) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "phase": "holdout" if holdout else "development",
        "created_at": datetime.now(timezone.utc).isoformat(), "code_hash": code_hash(),
        "config_hash": config_hash(config), "config": config.dict(), "git": git_info(),
        "data_path": str(Path(data).resolve()), "python": sys.version, "platform": platform.platform(),
        "packages": {p: metadata.version(p) for p in PACKAGES + (
            V04_PACKAGES if config.feature_set == "v04" or MVO_POLICIES & set(config.policies)
            or config.fill_price != "open" else [])}}
    write_json(root / "manifest.json", manifest, exclusive=True)
    def event(kind, **details):
        with (root / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean(dict(time=datetime.now(timezone.utc).isoformat(), kind=kind, **details)))+"\n")
    try:
        if holdout and not freeze_file:
            raise ValueError("Final holdout requires --freeze and --ack-final")
        if holdout:
            check_freeze(freeze_file, config)
        bundle = read_bundle(data, allow_after=None if holdout else config.holdout_start)
        # Hybrid snapshots also price unreconciled ex-dates with the proxy.
        if bundle.metadata["corporate_actions"] != "explicit_events" and not allow_proxy:
            raise ValueError("Free-data corporate-action proxy requires explicit --allow-proxy acknowledgement")
        if holdout:
            record = check_freeze(freeze_file, config, bundle)
            if record.get("holdout_opened_before") or opened_holdouts(Path(freeze_file).parent, config):
                raise ValueError("This holdout has already been opened (see its receipt); it is spent")
            # A one-time local receipt records failed final attempts too. Never erase it.
            receipt = Path(str(freeze_file) + ".opened.json")
            write_json(receipt, {"opened_at": datetime.now(timezone.utc).isoformat(),
                "run": str(root.resolve()), "freeze_hash": sha256(freeze_file)}, exclusive=True)
            manifest["freeze"] = record
        end = config.holdout_end if holdout else config.dev_end
        if bundle.metadata["end"] < end or bundle.calendar[-1] < pd.Timestamp(end)-pd.Timedelta(days=14):
            raise ValueError("Snapshot does not cover the requested end date")
        if bundle.metadata.get("partial_assembly") and (holdout or config.dev_end >= "2020-01-01"):
            raise ValueError("A partial preview snapshot is for 2019-and-earlier engineering debugging only")
        if ((config.fill_price == "vwap30" or config.participation_basis == "first30")
                and (bundle.metadata.get("intraday") or {}).get("omitted")):
            raise ValueError("VWAP fills and first-30-minute participation need a snapshot sealed with minute bars")
        # RESEARCH section 9: debugging before its rules run stays in 2019 and earlier, labels included.
        if not holdout and config.dev_end < "2020-01-01" and bundle.metadata["end"] > "2019-12-31":
            raise ValueError("A pre-2020 debug configuration needs a snapshot sealed by 2019-12-31")
        manifest["data_metadata"] = bundle.metadata
        manifest["data_prefix_hash"] = bundle.prefix_hash(config.holdout_start)
        manifest["evidence"] = "synthetic_only" if bundle.metadata["source"] == "SYNTHETIC" else "historical_exploratory_unverified_source"
        write_json(root / "manifest.json", manifest)
        v04 = config.feature_set == "v04"
        needs_neutral = v04 or bool(MVO_POLICIES & set(config.policies))
        signals = None
        if needs_neutral:
            from .signals import prepare_v04
            panel, signals = prepare_v04(bundle, config, library=v04)
            event("signals_ready", factors=len(signals.names) if v04 else 0)
            if v04:
                signals.ic.to_csv(root / "factor_daily_ic.csv.gz")
                write_json(root / "factor_coverage.json", signals.info.get("coverage", {}))
        else:
            panel = prepare(bundle, config)
        predictions, audits = predict_oos(panel, bundle.calendar, config, holdout=holdout,
            on_fold=lambda audit: event("fold_complete", **audit), signals=signals if v04 else None)
        predictions.to_csv(root / "predictions.csv.gz", index=False)
        write_json(root / "folds.json", audits)
        manifest["prediction_hash"] = canonical_hash(predictions)
        ic = rank_ic(predictions, panel)
        ic.to_csv(root / "rank_ic.csv", index=False)
        coverage = panel.groupby("date").agg(observed_names=("symbol", "size"), eligible_names=("eligible", "sum"))
        coverage.to_csv(root / "universe_coverage.csv")
        start = config.holdout_start if holdout else config.dev_start
        bench = benchmark_returns(panel, start, end, bundle)
        bench.to_csv(root / "benchmark.csv")
        ranks = style_ranks(panel)
        results, return_series, details, years = [], {}, [], []
        intraday = bundle.tables.get("intraday") if config.fill_price == "vwap30" else None
        views = execution_views(panel, start, end, intraday)
        if config.participation_basis == "first30":
            first30 = first30_adv(bundle.tables["intraday"], panel)
            for day, frame in views.items():
                frame["adv_cap"] = first30.get(day, pd.Series(dtype=float)).reindex(frame.index).fillna(0.0)
        planners = {}
        if MVO_POLICIES & set(config.policies):
            from .factors import market
            from .optimize import Planner, matured_ic
            from .risk import RiskModel
            days = bundle.calendar[(bundle.calendar >= start) & (bundle.calendar <= end)]
            risk = RiskModel(panel, market(panel, bundle), bundle, config, keep=days[::config.rebalance])
            event("risk_model_ready", snapshots=len(risk.snapshots))
            for model in config.models:
                folds_info = [dict(test_start=a["test_start"], test_end=a["test_end"],
                                   valid_ic=a["models"][model].get("valid_ic"), valid_rho5=a["models"][model].get("valid_rho5"))
                              for a in audits]
                table = matured_ic(predictions[predictions.model == model], panel)
                for policy in MVO_POLICIES & set(config.policies):
                    planners[(model, policy)] = Planner(config, risk, table, folds_info, objective=policy)
        scenarios = [("zero_all_costs", 0.0, True)] + [(f"slip_{x:g}bps", float(x), False) for x in config.slippage_grid_bps]
        keys = [(model, policy, scenario, bps, zero) for model in config.models for policy in config.policies
                for scenario, bps, zero in scenarios]
        _TASK_STATE.update(panel=panel, bundle=bundle, config=config, holdout=holdout, views=views,
                           planners=planners, signals={m: predictions[predictions.model == m] for m in config.models})
        bench95 = bench.benchmark_return * config.invested
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
                scaled = {f"b95_{k}": v for k, v in relative(r, bench95).items()} if needs_neutral else {}
                results.append(dict(model=model, policy=policy, scenario=scenario, **performance(d),
                                    **relative(r, bench.benchmark_return), **scaled, **exposures(out["holdings"], ranks)))
                return_series[(model, policy, scenario)] = r
                # Rebase each slice at its own starting NAV; do not reset positions at fold boundaries.
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
            _TASK_STATE.clear()
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
        report = ["# v0.1 run report", "", f"Evidence: **{manifest['evidence']}**", "",
            "No performance number is a live-trading result or a promise of future return.", "",
            "## Interpretation", "",
            "summary.csv: all models × policies × fee scenarios, including failures in the run manifest.",
            "by_fold.csv: non-resetting OOS portfolio slices; inspect stability, exposure and drawdowns, not only Sharpe.",
            "paired_comparisons.json: paired DAILY excess, circular block-bootstrap intervals and p-values; Holm within each cost scenario (ablation and model comparisons together).",
            "benchmark.csv: equal-weight previous-close eligible pool, close to close, no costs, removals at their outcome; benchmark_comparisons.json tests each scenario against it (Holm within each cost scenario).",
            "gross_same_trades_return adds that day's cash costs back to the SAME trades; it is not a self-financing frictionless portfolio.",
            "zero_all_costs reruns the ledger with all cash costs removed; slip_0bps still pays commission and statutory charges.",
            "Cash earns zero. No benchmark-index alpha, sector neutrality, capacity estimate or live fill quality is established.",
            "Daily open fills, minimum fees, lagged-ADV caps, queue priority, zero delisting recovery and corporate-action assumptions require review.",
            "Final NAV is marked to close without forced liquidation; no uncharged terminal sale is implied.", "",
            "## Data limitations", ""]
        report.extend("- " + x for x in bundle.metadata.get("limitations", []))
        if bundle.metadata["corporate_actions"] == "reference_reinvestment_proxy":
            report.append("- Corporate actions use synthetic reinvested reference-price units, including fractional units; not actual shareholder entitlements or dividend taxes.")
        report += ["", "## Artifacts", "", "Read manifest.json, folds.json, rank_ic.csv and each scenario's daily/orders/holdings/events/allocations before drawing conclusions.",
                   "A positive result on a sampled or coverage-unverified free dataset is exploratory, not established market alpha."]
        def pct(value):
            return f"{100*value:.2f}%" if np.isfinite(value) else "NA"
        report += ["", "## Complete comparison matrix", "",
            "No best-result selection. Annualization uses 252 sessions; costs are CNY. Initial build is included.", "",
            "| Model | Policy | Scenario | Annual return | Sharpe (rf=0) | Max DD | Mean 2-way turnover | Mean exposure |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
        for result in results:
            report.append(f"| {result['model']} | {result['policy']} | {result['scenario']} | "
                f"{pct(result['annualized_return'])} | {result['sharpe_rf0']:.3f} | "
                f"{pct(result['max_drawdown'])} | {pct(result['mean_two_way_turnover'])} | {pct(result['mean_exposure'])} |")
        report += ["", "Cash/exposure differences can explain lower volatility; they do not establish a better signal.",
                   "The same predictions feed every policy and fee scenario. LP arms form a penalty × turnover-cap 2-by-2 ablation."]
        if len(bench):
            growth = np.cumprod(1 + bench.benchmark_return.to_numpy())
            drawdown = float(np.min(growth / np.maximum.accumulate(np.r_[1.0, growth])[1:] - 1))
            report += ["", "## Against the eligible-pool benchmark", "",
                f"Benchmark annualized {pct(growth[-1] ** (252 / len(growth)) - 1)}, max drawdown {pct(drawdown)}, "
                f"{bench.names.mean():.0f} names a day on average. Beta below one earns 'active return' in a falling pool by itself; read alpha and IR with beta.", "",
                "| Model | Policy | Scenario | Active return | Beta | Alpha (arith) | Tracking error | IR |",
                "|---|---|---|---:|---:|---:|---:|---:|"]
            for result in results:
                report.append(f"| {result['model']} | {result['policy']} | {result['scenario']} | "
                    f"{pct(result['active_annualized_return'])} | {result['beta']:.2f} | {pct(result['alpha_annual_arith'])} | "
                    f"{pct(result['tracking_error'])} | {result['information_ratio']:.2f} |")
        rejected = [c for c in comparisons if c["holm_reject_5pct"]]
        report += ["", "## Paired comparisons after Holm (5%, within each cost scenario)", "",
                   f"{len(rejected)} of {len(comparisons)} paired differences stay significant after Holm."]
        report += [f"- {c['left']} vs {c['right']}: {pct(c['annualized_mean_difference'])} "
                   f"[{pct(c['ci95_low'])}, {pct(c['ci95_high'])}], p={c['p_bootstrap']:.3f}" for c in rejected]
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
