"""v0.4 end to end on a synthetic schema-3 snapshot: library, selection with fallback, risk
model, both optimizer objectives, P2 policies and the parallel path's serial twin."""
import json
import numpy as np
import pandas as pd
import pytest
from costrank.config import Config
from costrank.data import ACTIONS3, CONSTITUENTS, DELISTINGS, INDICES, INTRADAY, RECONCILIATION, read_bundle, seal
from costrank.experiment import run
from test_v04_signals import library_bundle


def empty(columns, dates=("date",)):
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in dates else float) for c in columns})


def schema3(tmp_path, *, bundle=None, before=None, name="snap"):
    """Sealed synthetic schema-3 snapshot; with `before`, only the records dated before that day, as
    a development snapshot sealed ahead of the holdout (its prefix equals the full snapshot's)."""
    bundle = bundle or library_bundle(tmp_path / "src")
    root = tmp_path / name
    root.mkdir()
    cut = lambda frame, column="date": (frame if before is None
                                        else frame[frame[column] < pd.Timestamp(before)].reset_index(drop=True))
    calendar = bundle.calendar if before is None else bundle.calendar[bundle.calendar < pd.Timestamp(before)]
    b = cut(bundle.bars)
    b.to_parquet(root / "bars.parquet", index=False)
    bundle.instruments.assign(listed=bundle.instruments.listed.dt.strftime("%Y-%m-%d"),
                              delisted="").to_csv(root / "instruments.csv", index=False)
    pd.DataFrame({"date": calendar.strftime("%Y-%m-%d")}).to_csv(root / "calendar.csv", index=False)
    pd.DataFrame(columns=DELISTINGS).to_csv(root / "delistings.csv", index=False)
    actions = empty(ACTIONS3, dates=("date", "pay_date")).astype({"symbol": object})
    actions.to_parquet(root / "actions.parquet", index=False)
    t = bundle.tables
    cut(t["attributes"]).to_parquet(root / "attributes.parquet", index=False)
    cut(t["industries"]).to_parquet(root / "industries.parquet", index=False)
    empty(CONSTITUENTS, dates=("date", "update_date")).astype({"index": object, "symbol": object}).to_parquet(
        root / "constituents.parquet", index=False)
    empty(INDICES).astype({"index": object}).to_parquet(root / "indices.parquet", index=False)
    intraday = pd.DataFrame({"date": b.date, "symbol": b.symbol, "first_open": b.open, "first_close": b.close,
                             "first_volume": b.volume / 4.0, "first_amount": b.amount / 4.0 * 1.001,
                             "last_open": b.close, "last_close": b.close, "bars": 8.0, "realized_var": 1e-4,
                             "day_volume": b.volume.astype(float), "day_amount": b.amount.astype(float)})[INTRADAY]
    intraday.to_parquet(root / "intraday.parquet", index=False)
    cut(t["forecasts"], "pub_date").to_parquet(root / "forecasts.parquet", index=False)
    cut(t["express"], "available_date").to_parquet(root / "express.parquet", index=False)
    empty(RECONCILIATION).astype({"symbol": object, "status": object}).to_parquet(root / "reconciliation.parquet", index=False)
    return seal(root, {"source": "SYNTHETIC", "evidence": "synthetic_only", "corporate_actions": "hybrid_explicit_proxy",
                       "coverage": "synthetic", "end": calendar[-1].strftime("%Y-%m-%d"),
                       "limitations": ["Business-day fixture"]}), root


def small(**kw):
    base = dict(dev_start="2018-09-03", dev_end="2019-03-01", holdout_start="2020-01-01", holdout_end="2020-12-31",
                train_days=60, valid_days=20, test_days=20, horizon=2, rebalance=5, universe_size=30, topk=5,
                min_adv=0, max_weight=0.10, trees=20, threads=2, bootstrap_samples=20, risk_min_history=40,
                risk_half_life=20, alpha_ic_min=5, alpha_ic_window=20, candidate_top=15, industry_top=3,
                te_annual=0.60, industry_band=0.30, style_band=3.0, slippage_grid_bps=(10.0,), dividend_tax=True)
    base.update(kw)
    return Config(**base)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return schema3(tmp_path_factory.mktemp("v04"))


def test_schema3_fixture_reads_back(snapshot):
    bundle, root = snapshot
    again = read_bundle(root)
    assert again.metadata["schema_version"] == 3 and set(again.tables) >= {"attributes", "intraday"}


def test_hybrid_snapshot_needs_the_proxy_acknowledgement(snapshot, tmp_path):
    _, root = snapshot
    with pytest.raises(RuntimeError, match="allow-proxy"):
        run(small(), root, tmp_path / "runs")


def test_pre2020_configuration_refuses_a_snapshot_reaching_2020(tmp_path):
    from costrank.data import synthetic
    synthetic(tmp_path / "long", days=530, symbols=3)  # 2018-01-02 .. 2020-01
    cfg = Config(dev_start="2018-10-01", dev_end="2018-10-31", holdout_start="2021-01-01", holdout_end="2021-12-31",
                 universe_size=3, topk=2, min_adv=0)
    with pytest.raises(RuntimeError, match="sealed by 2019-12-31"):
        run(cfg, tmp_path / "long", tmp_path / "runs")


def test_s4p4_run_with_fallback_selection_and_both_objectives(snapshot, tmp_path):
    _, root = snapshot
    cfg = small(feature_set="v04", label_mode="neutral_rank", models=("factor", "ridge", "lightgbm", "ic_linear"),
                policies=("topk", "mvo", "mvo_rank"))
    out = run(cfg, root, tmp_path / "runs", allow_proxy=True)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and "clarabel" in manifest["packages"]
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 4 * 3 * 2 and summary.b95_information_ratio.notna().all()
    folds = json.loads((out / "folds.json").read_text(encoding="utf-8"))
    assert all("selection" in f for f in folds)
    assert all(np.isfinite(f["models"]["lightgbm"]["valid_rho5"]) for f in folds)
    alloc = pd.read_csv(out / "lightgbm__mvo__slip_10bps" / "allocations.csv.gz")
    assert (alloc.status == "optimal").mean() > 0.8
    assert alloc.loc[alloc.status == "optimal", "ex_ante_te"].max() <= cfg.te_annual + 1e-3
    daily = pd.read_csv(out / "lightgbm__mvo__slip_10bps" / "daily.csv.gz")
    assert (daily.cash > -1e-6).all() and "dividend_tax_accrued" in daily
    import sys
    sys.path.insert(0, "scripts")
    from feasibility_report import summarize
    feasibility = summarize(out)
    assert set(feasibility) == {f"{m}__{p}__slip_10bps" for m in cfg.models for p in ("mvo", "mvo_rank")}
    lg = feasibility["lightgbm__mvo__slip_10bps"]
    assert lg["rebalances"] == len(alloc) and lg["hold_fallback_share"] == pytest.approx((alloc.status == "hold_fallback").mean())
    assert lg["ex_ante_te_max"] <= cfg.te_annual * lg["max_risk_limit_scale"] + 1e-3
    assert lg["infeasible_share"] >= lg["hold_fallback_share"] and lg["over_limit"] == (lg["infeasible_share"] > 0.05)
    assert max(lg["max_excess"].values()) <= 1e-4 and 0 <= lg["outside_weight_median"] <= lg["outside_weight_max"] <= 1


def test_s2p4_run_uses_v02_features_with_neutral_alpha_ic(snapshot, tmp_path):
    _, root = snapshot
    cfg = small(models=("ridge",), policies=("mvo",))
    out = run(cfg, root, tmp_path / "runs", allow_proxy=True)
    folds = json.loads((out / "folds.json").read_text(encoding="utf-8"))
    assert "selection" not in folds[0] and np.isfinite(folds[0]["models"]["ridge"]["valid_ic"])
    assert not (out / "factor_daily_ic.csv.gz").exists()


def test_capacity_configuration_fills_at_vwap_on_first30_participation(snapshot, tmp_path):
    _, root = snapshot
    cfg = small(models=("ridge",), policies=("mvo",), fill_price="vwap30", participation_basis="first30",
                impact_coefficient=1.0, slippage_grid_bps=(0.0,), initial_cash=1e6)
    out = run(cfg, root, tmp_path / "runs", allow_proxy=True)
    orders = pd.read_csv(out / "ridge__mvo__slip_0bps" / "orders.csv.gz")
    filled = orders[orders.status != "rejected"]
    assert len(filled) and (filled.fill_price > 0).all() and (filled.impact > 0).all()


def test_compare_v04_reports_the_declared_decision(snapshot, tmp_path):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, "scripts")
    from compare_v04 import decide
    _, root = snapshot
    runs = {}
    for name, kw in (("s2p2", dict(policies=("cost",))), ("s4p2", dict(policies=("cost",), feature_set="v04",
                     label_mode="neutral_rank")), ("s2p4", dict(policies=("mvo",))),
                     ("s4p4", dict(policies=("mvo",), feature_set="v04", label_mode="neutral_rank"))):
        runs[name] = run(small(models=("ridge", "lightgbm"), **kw), root, tmp_path / name, allow_proxy=True)
    args = SimpleNamespace(**{k: str(v) for k, v in runs.items()}, kappa=[], capacity=[], registry=None, samples=99,
                           check_declared=False,
                           v02_run=str(runs["s2p2"]), snapshot=str(root))
    report = decide(args)
    assert report["frozen_protocol"] in {"lightgbm+S4+mvo", "lightgbm+S2+cost"}
    assert set(report["conditions"]) == {"primary_p_below_0.05", "ridge_point_estimate_positive",
                                         "s4p4_not_below_s2p4_and_s4p2_cost", "s4p4_fallback_at_most_5pct",
                                         "tracking_errors_at_least_2pct"}
    assert 0 < report["primary_test"]["p_value"] <= 1 and report["pbo"]["strategies"] == 8
    assert set(report["attribution"]["lightgbm"]) == {"S4P4-S2P4", "S4P2-S2P2", "S2P4-S2P2", "S4P4-S4P2_package"}
    assert len(report["matrix"]["signal_layer"]) == len(report["matrix"]["portfolio_layer"]) == 8
    from compare_v04 import active, ratio
    table = pd.DataFrame(report["matrix"]["table"])
    assert np.isfinite(table[(table.cell == "S2P2") & (table.scenario == "slip_10bps")].information_ratio).all()
    primary = table[(table.cell == "S4P4") & (table.model == "lightgbm") & (table.policy == "mvo")
                    & (table.scenario == "slip_10bps")].information_ratio.iloc[0]
    assert primary == pytest.approx(ratio(active(runs["s4p4"], "lightgbm", "mvo")[0], 0)["information_ratio"], abs=1e-12)
    assert set(report["costs_and_years"]) == {f"{c}/{m}" for c in ("S2P2", "S4P2", "S2P4", "S4P4") for m in ("lightgbm", "ridge")}
    assert report["versus_v02"]["ridge"]["annualized_mean_difference"] == 0  # the S2P2 run stands in for v0.2
    assert report["hedged"]["csi500"] == "index not in snapshot"
    assert report["selection_frequency"]["folds"] >= 1
    from feasibility_report import summarize
    from report_v04 import render
    text = render(json.loads(json.dumps(report, default=float)),
                  feasibility={"limit": 0.05, "portfolios": summarize(runs["s4p4"]), "any_over_limit": False})
    assert f"冻结协议：**{report['frozen_protocol']}**" in text and "| S4P4 |" in text
    assert ("全部条件满足" in text) == all(report["conditions"].values())
    assert all(h in text for h in ("全矩阵的净夏普差", "换手、成本拖累", "因子入选频次", "对 v0.2 主板运行", "index not in snapshot"))
    if not report["conditions"]["primary_p_below_0.05"]:
        assert "此次未检出" in text and "不表示旧方法更优" in text
    cells = report["provenance"]["cells"]
    assert set(cells) == {"S2P2", "S4P2", "S2P4", "S4P4"} and len({v["data_prefix_hash"] for v in cells.values()}) == 1
    assert "## 来源" in text and runs["s4p4"].name in text and cells["S4P4"]["prediction_hash"][:12] in text
    gated = render(json.loads(json.dumps(report, default=float)),
                   gates=[("runs/audit/gate2-valuation-timing.json", {"passed": False})])
    assert "未通过；按 §9 补充的既定处理，各配置 `valuation_fields: false`" in gated


def test_run_audit_passes_v04_ledgers(snapshot, tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from audit_run import audit_run
    _, root = snapshot
    for kw in (dict(models=("ridge",), policies=("mvo", "topk"), feature_set="v04", label_mode="neutral_rank"),
               dict(models=("ridge",), policies=("mvo",), fill_price="vwap30", participation_basis="first30",
                    impact_coefficient=1.0, slippage_grid_bps=(0.0,))):
        out = run(small(**kw), root, tmp_path / str(len(kw)), allow_proxy=True)
        report = audit_run(out)
        assert report["passed"], (report["scenarios_failed"], report["violation_counts"])
        assert report["violation_counts"]["optimizer_bound_breaches"] == 0


def test_style_report_attributes_a_run(snapshot, tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from style_report import attribute
    bundle, root = snapshot
    cfg = small(models=("ridge",), policies=("topk",))
    out = run(cfg, root, tmp_path / "runs", allow_proxy=True)
    report = attribute(out, read_bundle(root), cfg, "slip_10bps")
    item = report["ridge__topk__slip_10bps"]
    assert item["days"] > 50 and set(item["styles"]) == {"size", "beta", "momentum", "residual_vol", "liquidity",
                                                         "value", "earnings_yield", "reversal"}
    assert np.isfinite(item["mean_ex_ante_te"])


def test_factor_report_runs_on_a_window(snapshot, tmp_path):
    import json as js
    import subprocess
    import sys
    _, root = snapshot
    cfg = tmp_path / "cfg.json"
    cfg.write_text(js.dumps({"dev_start": "2018-09-03", "dev_end": "2019-03-01", "holdout_start": "2020-01-01",
                             "holdout_end": "2020-12-31", "universe_size": 30, "topk": 5, "min_adv": 0,
                             "feature_set": "v04", "label_mode": "neutral_rank"}))
    out = tmp_path / "report.json"
    subprocess.run([sys.executable, "scripts/factor_report.py", str(root), "--config", str(cfg), "--start", "2018-06-01",
                    "--end", "2019-02-01", "--output", str(out)], check=True, capture_output=True)
    report = js.loads(out.read_text(encoding="utf-8"))
    assert report["sessions"] > 100 and len(report["factors"]) > 150
    by_name = {f["factor"]: f for f in report["factors"]}
    core = by_name["core_ret5"]
    assert len(core["decile_returns_5d"]) == 10 and set(core["ic_decay"]) == {"5", "10", "20"}
    assert core["long_short"]["periods"] > 10 and 0 <= core["top_decile_replaced_5"] <= 1
    assert all(len(f["nearest"]) == 3 for f in report["factors"])
    assert np.isfinite(core["crowding"]["attention_mean"]) and np.isfinite(core["crowding"]["value_spread_mean"])


def test_holdout_report_follows_supplement_six(snapshot, tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from holdout_report import report
    _, root = snapshot
    out = run(small(models=("factor", "ridge", "lightgbm"), policies=("cost",)), root, tmp_path / "runs", allow_proxy=True)
    r = report(out, development=out, samples=99)
    assert set(r["models"]) == {"lightgbm", "ridge", "factor"}
    lg = r["models"]["lightgbm"]
    assert lg["ir_interval_95"][0] <= lg["ir_interval_95"][1] and lg["max_drawdown"] <= 0
    assert r["minimum_detectable_ratio"] == pytest.approx(2.8016 * np.sqrt(252 / lg["sessions"]), rel=1e-3)
    assert 0 < r["lightgbm_minus_ridge"]["p_value"] <= 1 and r["development_lightgbm_ir_by_year"]
    with pytest.raises(ValueError, match="exactly one policy"):
        report(run(small(models=("factor", "ridge", "lightgbm"), policies=("cost", "topk")), root, tmp_path / "two",
                   allow_proxy=True))


def test_trial_registry_counts_planned_scenarios_a_failed_run_never_wrote(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from trial_registry import scan
    run = tmp_path / "runs" / "r1"
    (run / "ridge__mvo__slip_10bps").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "failed", "config": {
        "models": ["ridge", "lightgbm"], "policies": ["mvo"], "slippage_grid_bps": [10.0]}}))
    pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-06"], "net_return": [0.01, -0.005, 0.002]}).to_csv(
        run / "ridge__mvo__slip_10bps" / "daily.csv.gz", index=False)
    table = scan([tmp_path / "runs"])
    # Two models x one policy x (zero cost + 10 bps): one ledger written, three planned trials without one.
    assert len(table) == 4 and table.sharpe_daily.notna().sum() == 1


def test_report_figures_render_from_finished_runs(snapshot, tmp_path):
    pytest.importorskip("matplotlib")
    import sys
    sys.path.insert(0, "scripts")
    import report_figures
    _, root = snapshot
    runs = {}
    for name, kw in (("S2P2", dict(policies=("cost",))), ("S4P2", dict(policies=("cost",), feature_set="v04",
                     label_mode="neutral_rank")), ("S2P4", dict(policies=("mvo",))),
                     ("S4P4", dict(policies=("mvo",), feature_set="v04", label_mode="neutral_rank"))):
        runs[name] = run(small(models=("ridge", "lightgbm"), **kw), root, tmp_path / name, allow_proxy=True)
    out = tmp_path / "figures"
    out.mkdir()
    report_figures.cumulative(runs, out)
    report_figures.by_year(runs, out)
    kappa = run(small(models=("ridge", "lightgbm"), policies=("mvo",), feature_set="v04", label_mode="neutral_rank",
                      kappa_scale=2.0), root, tmp_path / "k2", allow_proxy=True)
    report_figures.frontier(runs["S4P4"], [kappa], out)
    assert sorted(p.name for p in out.glob("*.png")) == ["active_cumulative.png", "ir_by_year.png", "kappa_frontier.png"]
    assert all(p.stat().st_size > 10_000 for p in out.glob("*.png"))
