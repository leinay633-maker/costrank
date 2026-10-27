"""v0.4 risk model and mean-variance policy: causal snapshots, constraint compliance, kappa."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from costrank.config import Config
from costrank.factors import market
from costrank.features import prepare
from costrank.optimize import Planner
from costrank.risk import STYLES, RiskModel, Snapshot, span_factor_returns, spans, standardize_styles
from test_v04_signals import config as signal_config, library_bundle


def test_risk_snapshots_are_causal_and_well_formed(tmp_path):
    cut = pd.Timestamp("2018-11-30")
    cfg = replace(signal_config(), risk_min_history=60, risk_half_life=30)
    snaps = {}
    for name, moved in (("a", None), ("b", cut)):
        bundle = library_bundle(tmp_path / name, perturb_after=moved)
        panel = prepare(bundle, cfg)
        keep = [d for d in sorted(panel.date.unique()) if d <= cut][-3:]
        snaps[name] = RiskModel(panel, market(panel, bundle), bundle, cfg, keep=keep).snapshots
        if name == "a":
            bars = panel.groupby("date").symbol.apply(set)
    assert len(snaps["a"]) == 3
    for day, a in snaps["a"].items():
        b = snaps["b"][day]
        pd.testing.assert_frame_equal(a.covariance, b.covariance)
        pd.testing.assert_frame_equal(a.exposures, b.exposures)
        pd.testing.assert_series_equal(a.specific, b.specific)
        pd.testing.assert_frame_equal(a.outside_exposures, b.outside_exposures)
        pd.testing.assert_series_equal(a.outside_specific, b.outside_specific)
        assert np.linalg.eigvalsh(a.covariance.to_numpy()).min() > -1e-12
        assert (a.specific > 0).all() and a.exposures[STYLES].abs().max().max() < 10
        # Every symbol with a bar is covered once: the pool as names, the rest outside it.
        assert set(a.names) | set(a.outside_exposures.index) == bars[day] and len(a.outside_exposures) >= 10
        assert not a.names.isin(a.outside_exposures.index).any()
        assert (a.outside_specific > 0).all() and a.outside_exposures[STYLES].abs().max().max() < 10


def test_visit_sees_every_snapshot_and_the_kept_ones_are_identical(tmp_path):
    cfg = replace(signal_config(), risk_min_history=60, risk_half_life=30)
    bundle = library_bundle(tmp_path / "a")
    panel = prepare(bundle, cfg)
    keep = sorted(panel.date.unique())[-3:]
    seen = {}
    model = RiskModel(panel, market(panel, bundle), bundle, cfg, keep=keep, visit=seen.__setitem__)
    assert len(model.snapshots) == 3 and len(seen) > 3
    assert all(seen[day] is snap for day, snap in model.snapshots.items())


def test_bias_statistics_report(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from risk_bias import bias, collect
    b = np.random.default_rng(0).normal(0, 1, 2000)
    calm, wild = bias(b), bias(2 * b)
    assert calm["band"][0] < calm["bias"] < calm["band"][1] and wild["bias"] == pytest.approx(2 * calm["bias"])
    cfg = replace(signal_config(), risk_min_history=60, risk_half_life=30)
    bundle = library_bundle(tmp_path / "a")
    names = sorted(bundle.bars.symbol.unique())[:5]
    book = {day: pd.Series(0.19, index=names) for day in sorted(bundle.bars.date.unique())}
    report = collect(bundle, cfg, "2018-06-01", "2019-03-01", holdings={"fixed": (0.95, book)}, n_random=8, n_names=10)
    assert report["random_total"]["portfolios"] == report["random_active"]["portfolios"] == 8
    assert report["random_total"]["median_bias"] > 0 and report["strategies"]["fixed"]["bias"] > 0
    styles = {k: v for k, v in report["factors"].items() if k in STYLES}
    # The 252-day beta has no history in this 320-session fixture: not estimated rather than a ratio
    # of rounding noise; the other styles count only days with an exposure.
    assert styles["beta"]["bias"] is None and len(styles) == len(STYLES)
    assert all(v["days"] > 40 and np.isfinite(v["bias"]) and v["bias"] > 0 for k, v in styles.items() if k != "beta")
    assert all(v["days"] > 150 for k, v in styles.items() if k not in {"beta", "momentum"})
    from report_v04 import bias_section
    text = "\n".join(bias_section(dict(report, window=["2018-06-01", "2019-03-01"])))
    assert "未估计（窗口内没有暴露）：beta" in text and "| momentum |" in text


def fake_snapshot(n=400, seed=0):
    rng = np.random.default_rng(seed)
    names = pd.Index([f"sh.{600000 + i}" for i in range(n)])
    industry = pd.Series(np.array(["A", "B", "C", "D"])[np.arange(n) % 4], index=names)
    styles = rng.normal(0, 1, (n, len(STYLES)))
    dummies = pd.get_dummies(industry).to_numpy(float)
    exposures = pd.DataFrame(np.column_stack([styles, dummies]), index=names,
                             columns=STYLES + ["ind_A", "ind_B", "ind_C", "ind_D"])
    k = exposures.shape[1]
    q = rng.normal(0, 0.004, (k, k))
    cov = pd.DataFrame(q @ q.T + np.eye(k) * 1e-5, index=exposures.columns, columns=exposures.columns)
    return Snapshot(names, exposures, cov, pd.Series(rng.uniform(1e-4, 9e-4, n), index=names), industry)


class StubRisk:
    def __init__(self, snap):
        self.snap = snap

    def snapshot(self, day):
        return self.snap


def plan(snap, weights, *, kappa_scale=1.0, nav=1e6, objective="mvo", seed=1):
    cfg = Config(kappa_scale=kappa_scale, policies=("mvo",), candidate_top=120, industry_top=10)
    rng = np.random.default_rng(seed)
    score = pd.Series(rng.normal(0, 1, len(snap.names)), index=snap.names)
    ic = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=300), "label_end": pd.date_range("2019-01-08", periods=300),
                       "ic": 0.05})
    folds = [{"test_start": "2020-01-01", "test_end": "2020-12-31", "valid_ic": 0.04, "valid_rho5": 0.7}]
    planner = Planner(cfg, StubRisk(snap), ic, folds, objective=objective)
    market = pd.DataFrame({"adv20": 5e7}, index=snap.names)
    return cfg, planner(pd.Timestamp("2020-06-01"), score, weights, market, nav, pd.Series(dtype=float), weights.sum() == 0)


def test_mvo_respects_every_constraint_from_cash():
    snap = fake_snapshot()
    cfg, (w, audit) = plan(snap, pd.Series(dtype=float))
    assert audit["status"] == "optimal" and audit["ic_source"] == "matured_oos"
    assert w.sum() == pytest.approx(cfg.invested, abs=1e-6)
    assert w.min() >= -1e-9 and w.max() <= cfg.max_weight + 1e-7
    assert audit["ex_ante_te"] <= cfg.te_annual + 1e-4
    assert audit["max_style_active"] <= cfg.style_band + 1e-6
    assert audit["max_industry_active"] <= cfg.industry_band + 1e-6
    assert (w <= cfg.participation * 5e7 / 1e6 + 1e-7).all()
    assert audit["kappa"] == pytest.approx(0.3)


def test_higher_kappa_trades_less():
    snap = fake_snapshot()
    _, (start, _) = plan(snap, pd.Series(dtype=float), seed=1)
    turnovers = []
    for scale in (0.25, 1.0, 4.0):
        _, (w, audit) = plan(snap, start, kappa_scale=scale, seed=2)
        turnovers.append(audit["target_turnover"])
    assert turnovers[0] >= turnovers[1] >= turnovers[2]
    assert turnovers[0] > turnovers[2] + 1e-3


def test_impact_cones_are_tight_at_the_optimum():
    snap = fake_snapshot()
    _, (start, _) = plan(snap, pd.Series(dtype=float), seed=1)
    _, (w, audit) = plan(snap, start, seed=2)
    assert audit["status"] == "optimal" and audit["target_turnover"] > 1e-3 and audit["impact_cost"] > 0
    # The two second-order cones bound t by |buy + sell|^1.5 exactly where impact is charged.
    assert audit["impact_cost_solver"] == pytest.approx(audit["impact_cost"], rel=1e-4, abs=1e-9)


def test_spans_count_the_sessions_since_the_last_traded_close():
    traded = pd.DataFrame({"a": [True, True, False, False, True, True], "b": [False, True, True, False, True, False]})
    got = spans(traded)
    expected = pd.DataFrame({"a": [np.nan, 1, np.nan, np.nan, 3, 1], "b": [np.nan, np.nan, 1, np.nan, 2, np.nan]})
    pd.testing.assert_frame_equal(got, expected)


def test_a_residual_spanning_several_sessions_nets_exactly_their_factor_returns():
    # f_j is the factor return from close j to close j + 1, realized (and added to `carried`) at j + 1.
    f = [np.array([10.0 ** j]) for j in range(8)]
    carried = np.zeros((9, 1))
    for t in range(8):
        carried[t + 1] = carried[t] + (f[t - 1] if t >= 1 else 0.0)
    # At session 5 the return ending at close 6 with span 3 started at close 3: f_3 + f_4 + f_5.
    assert span_factor_returns(f[5], carried, 5, 3)[0] == pytest.approx(1000 + 10_000 + 100_000)
    assert span_factor_returns(f[5], carried, 5, 1)[0] == pytest.approx(100_000)
    assert span_factor_returns(f[5], carried, 5, 6)[0] == pytest.approx(sum(10.0 ** j for j in range(6)))
    np.testing.assert_allclose(span_factor_returns(f[5], carried, 5, np.array([1, 2]))[:, 0], [100_000, 110_000])


def suspended(bundle, symbol, first, sessions, gap):
    """Halt `symbol` for `sessions` sessions from session index `first` (stale close, no volume), then
    reopen `gap` away from the last traded close; later prices move with it, returns unchanged."""
    b = bundle.bars
    days = sorted(b.date.unique())
    halt, reopen = days[first:first + sessions], days[first + sessions]
    mine = b.symbol.eq(symbol)
    last = b.loc[mine & b.date.eq(days[first - 1]), "close"].iloc[0]
    halted = mine & b.date.isin(halt)
    b.loc[halted, ["open", "high", "low", "close", "preclose"]] = last
    b.loc[halted, ["volume", "amount"]] = 0
    b.loc[halted, "tradable"] = 0
    later = mine & b.date.ge(reopen)
    scale = last * (1 + gap) / b.loc[mine & b.date.eq(reopen), "close"].iloc[0]
    b.loc[later, ["open", "high", "low", "close", "preclose"]] *= scale
    b.loc[mine & b.date.eq(reopen), "preclose"] = last
    return days[first + sessions + 2]


def test_a_resumption_gap_enters_the_specific_variance(tmp_path):
    cfg = replace(signal_config(), risk_min_history=60, risk_half_life=30)
    risk = {}
    for gap in (0.0, 0.3):
        bundle = library_bundle(tmp_path / f"g{gap}")
        symbol = sorted(bundle.bars.symbol.unique())[3]
        day = suspended(bundle, symbol, 200, 4, gap)
        panel = prepare(bundle, cfg)
        snap = RiskModel(panel, market(panel, bundle), bundle, cfg, keep=[day]).snapshots[day]
        risk[gap] = snap.cover([symbol])[1].iloc[0]
    # A 30% reopening move over a span of five enters once as about 0.3^2 / 5 = 0.018; with half-life 30
    # (weight 1 - 0.5^(1/30) = 0.023) and 30% shrinkage that adds about 0.7 x 0.023 x 0.018 = 2.9e-4.
    # Without the division by the span it would add five times as much.
    assert np.isfinite(risk[0.0]) and 1.5e-4 < risk[0.3] - risk[0.0] < 6e-4


def test_outside_rows_take_the_pool_standardization():
    rng = np.random.default_rng(0)
    raw, cap = rng.normal(0, 1, (50, 3)), rng.uniform(1, 10, 50)
    other = np.vstack([raw[:5], [[np.nan, 100.0, -100.0]]])
    pool, out = standardize_styles(raw, cap, other)
    np.testing.assert_allclose(pool, standardize_styles(raw, cap))
    np.testing.assert_allclose(out[:5], pool[:5])  # the same raw values, the same exposures
    assert out[5, 0] == 0.0 and 0 < out[5, 1] < 10 and -10 < out[5, 2] < 0  # missing -> 0; pool clip bounds


def without(snap, gone):
    """The same session after `gone` left the pool: covered outside it, not in the benchmark."""
    keep = snap.names.drop(gone)
    return Snapshot(keep, snap.exposures.loc[keep], snap.covariance, snap.specific.loc[keep], snap.industry.loc[keep],
                    snap.exposures.loc[[gone]], snap.specific.loc[[gone]], snap.industry.loc[[gone]])


def test_holding_that_left_the_pool_sells_down_at_the_participation_limit():
    snap = fake_snapshot()
    _, (start, _) = plan(snap, pd.Series(dtype=float), seed=1)
    gone = start.idxmax()
    later = without(snap, gone)
    cfg = Config(policies=("mvo",), candidate_top=120, industry_top=10)
    ic = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=300), "label_end": pd.date_range("2019-01-08", periods=300), "ic": 0.05})
    planner = Planner(cfg, StubRisk(later), ic, [{"test_start": "2020-01-01", "test_end": "2020-12-31", "valid_ic": 0.04, "valid_rho5": 0.7}])
    market = pd.DataFrame({"adv20": 5e7}, index=snap.names)
    market.loc[gone, "adv20"] = 1e6  # 1% of it is 1% of a 1e6 account: the exit takes several sessions
    score = pd.Series(np.random.default_rng(2).normal(0, 1, len(later.names)), index=later.names)
    w, audit = planner(pd.Timestamp("2020-06-01"), score, start, market, 1e6, pd.Series(dtype=float), False)
    assert start[gone] > 0.02 and audit["status"] == "optimal" and audit["unreachable_names"] == 1
    assert w[gone] == pytest.approx(start[gone] - 0.01, abs=1e-6)
    assert audit["outside_weight"] == pytest.approx(start[gone]) and audit["style_excess"] < 1e-4
    # The holding stays in the risk terms: the audit's TE is the snapshot's, over pool and holding.
    union = later.names.union(w.index)
    active = w.reindex(union).fillna(0.0) - pd.Series(cfg.invested / len(later.names), index=later.names).reindex(union).fillna(0.0)
    assert audit["ex_ante_te"] == pytest.approx(np.sqrt(later.daily_variance(active) * 252), rel=1e-9)
    X, d, industry = later.cover([gone, "sz.000001"])
    assert d[gone] == snap.specific[gone] and industry[gone] == snap.industry[gone] and np.isnan(d["sz.000001"])


def planner_for(snap, **kw):
    cfg = Config(policies=("mvo",), candidate_top=120, industry_top=10, **kw)
    ic = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=300), "label_end": pd.date_range("2019-01-08", periods=300), "ic": 0.05})
    return cfg, Planner(cfg, StubRisk(snap), ic, [{"test_start": "2020-01-01", "test_end": "2020-12-31", "valid_ic": 0.04, "valid_rho5": 0.7}])


def test_an_industry_bound_moves_only_when_participation_makes_it_unreachable():
    snap = fake_snapshot()
    _, (start, audit) = plan(snap, pd.Series(dtype=float), seed=1)
    assert audit["relaxed_industry_bounds"] == 0 and audit["unreachable_names"] == 0  # a normal day
    # Five industry-A names at 6% each, each able to trade only 0.1% of NAV: industry A cannot get
    # below 29.5%, above its 23.75% + 3% bound, so that bound moves to 29.5% and nothing else changes.
    pinned = snap.industry[snap.industry.eq("A")].index[:5]
    held = start.drop(pinned, errors="ignore")
    held = held / held.sum() * (0.95 - 0.30)
    held = pd.concat([held, pd.Series(0.06, index=pinned)])
    cfg, planner = planner_for(snap)
    market = pd.DataFrame({"adv20": 5e7}, index=snap.names)
    market.loc[pinned, "adv20"] = 1e5  # 1% of it is 0.1% of a 1e6 account
    score = pd.Series(np.random.default_rng(2).normal(0, 1, len(snap.names)), index=snap.names)
    w, audit = planner(pd.Timestamp("2020-06-01"), score, held, market, 1e6, pd.Series(dtype=float), False)
    assert audit["status"] == "optimal" and audit["unreachable_names"] == 5 and audit["relaxed_industry_bounds"] == 1
    assert w.reindex(snap.industry.index[snap.industry.eq("A")]).fillna(0).sum() == pytest.approx(0.295, abs=1e-6)
    assert audit["industry_excess"] < 1e-6 and audit["risk_limit_scale"] == 1.0


def test_a_holding_in_an_industry_absent_from_the_pool_is_still_bounded():
    snap = fake_snapshot()
    # Five holdings outside the pool in industry Z, which has no pool name today, 2% each and frozen.
    cold = pd.Index([f"sz.{i:06d}" for i in range(1, 6)])
    styles = pd.DataFrame(0.0, index=cold, columns=snap.exposures.columns)
    later = Snapshot(snap.names, snap.exposures, snap.covariance, snap.specific, snap.industry,
                     styles, pd.Series(4e-4, index=cold), pd.Series("Z", index=cold))
    _, (start, _) = plan(snap, pd.Series(dtype=float), seed=1)
    held = pd.concat([start / start.sum() * 0.85, pd.Series(0.02, index=cold)])
    cfg, planner = planner_for(later)
    market = pd.DataFrame({"adv20": 5e7}, index=snap.names.append(cold))
    market.loc[cold, "adv20"] = 0.0
    score = pd.Series(np.random.default_rng(2).normal(0, 1, len(snap.names)), index=snap.names)
    w, audit = planner(pd.Timestamp("2020-06-01"), score, held, market, 1e6, pd.Series(dtype=float), False)
    # Industry Z has benchmark weight 0 and a +3% bound the frozen 10% cannot meet: it is counted,
    # moved to the reachable 10%, and visible to the audit.
    assert audit["status"] == "optimal" and audit["relaxed_industry_bounds"] == 1
    assert audit["max_industry_active"] == pytest.approx(0.10, abs=1e-6) and audit["industry_excess"] < 1e-6


def test_pinned_style_exposure_widens_every_risk_limit_by_the_smallest_common_factor():
    snap = fake_snapshot()
    # Holdings outside the pool that cannot trade (no ADV) and load +3 on size: 40% of NAV pins the
    # book's size exposure near +1.2, far outside the 0.3 band; no pool name can offset it all.
    cold = pd.Index([f"sz.{i:06d}" for i in range(1, 11)])
    styles = pd.DataFrame(0.0, index=cold, columns=snap.exposures.columns)
    styles["size"], styles["ind_A"] = 3.0, 1.0
    later = Snapshot(snap.names, snap.exposures, snap.covariance, snap.specific, snap.industry,
                     styles, pd.Series(4e-4, index=cold), pd.Series("A", index=cold))
    held = pd.Series(0.04, index=cold)
    cfg, planner = planner_for(later)
    market = pd.DataFrame({"adv20": 5e7}, index=snap.names.append(cold))
    market.loc[cold, "adv20"] = 0.0
    score = pd.Series(np.random.default_rng(2).normal(0, 1, len(snap.names)), index=snap.names)
    w, audit = planner(pd.Timestamp("2020-06-01"), score, held, market, 1e6, pd.Series(dtype=float), True)
    assert audit["status"] == "optimal" and audit["declared_status"] not in {"Solved", "AlmostSolved"}
    scale = audit["risk_limit_scale"]
    assert scale > 1.5 and audit["max_style_active"] <= cfg.style_band * scale + 1e-4
    assert audit["style_excess"] < 1e-4 and audit["te_excess"] < 1e-4 and audit["industry_excess"] < 1e-4
    assert np.allclose(w.reindex(cold), 0.04, atol=1e-7) and w.sum() == pytest.approx(cfg.invested, abs=1e-6)
    # A slightly smaller common widening is infeasible: the factor is the smallest one.
    narrower = Config(policies=("mvo",), candidate_top=120, industry_top=10, style_band=cfg.style_band * (scale - 0.01),
                      industry_band=cfg.industry_band * (scale - 0.01) if cfg.industry_band * (scale - 0.01) <= 1 else 1.0,
                      te_annual=min(cfg.te_annual * (scale - 0.01), 1.0))
    _, tight = planner_for(later, style_band=narrower.style_band, industry_band=narrower.industry_band, te_annual=narrower.te_annual)
    _, again = tight(pd.Timestamp("2020-06-01"), score, held, market, 1e6, pd.Series(dtype=float), True)
    assert again["declared_status"] not in {"Solved", "AlmostSolved"}


def test_participation_binds_at_large_capital_and_rank_objective_runs():
    snap = fake_snapshot()
    cfg, (w, audit) = plan(snap, pd.Series(dtype=float), nav=2e9)
    # 1% of a 5e7 ADV is 0.025% of a 2e9 account: 95% cannot be bought in one session, so the
    # budget and trading limits alone have no solution.
    assert audit["status"] == "hold_fallback" and audit["without_risk_limits"] not in {"Solved", "AlmostSolved"}
    _, (w, audit) = plan(snap, pd.Series(dtype=float), objective="mvo_rank")
    assert audit["status"] == "optimal" and audit["objective_kind"] == "mvo_rank"
    assert w.sum() == pytest.approx(0.95, abs=1e-6)


def test_untradable_name_keeps_its_weight_instead_of_a_phantom_sale():
    snap = fake_snapshot()
    _, (start, _) = plan(snap, pd.Series(dtype=float), seed=1)
    held = start.copy()
    stuck = held.index[0]
    held[stuck] = 0.06  # drifted above the 4% cap while halted
    held = held / held.sum() * 0.95
    cfg = Config(policies=("mvo",), candidate_top=120, industry_top=10)
    ic = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=300), "label_end": pd.date_range("2019-01-08", periods=300), "ic": 0.05})
    planner = Planner(cfg, StubRisk(snap), ic, [{"test_start": "2020-01-01", "test_end": "2020-12-31", "valid_ic": 0.04, "valid_rho5": 0.7}])
    market = pd.DataFrame({"adv20": 5e7}, index=snap.names)
    market.loc[stuck, "adv20"] = 0.0
    score = pd.Series(np.random.default_rng(2).normal(0, 1, len(snap.names)), index=snap.names)
    w, audit = planner(pd.Timestamp("2020-06-01"), score, held, market, 1e6, pd.Series(dtype=float), False)
    assert audit["status"] in {"optimal", "hold_fallback"}
    assert w[stuck] == pytest.approx(held[stuck], abs=1e-7)


def test_tracking_error_below_two_percent_voids_the_information_ratio():
    import sys
    sys.path.insert(0, "scripts")
    from compare_v04 import at_least, ratio
    quiet = pd.Series(np.random.default_rng(0).normal(0.0002, 0.0005, 500))
    loud = pd.Series(np.random.default_rng(1).normal(0.0002, 0.005, 500))
    assert np.isnan(ratio(quiet)["information_ratio"]) and np.isfinite(ratio(loud)["information_ratio"])
    assert not at_least(ratio(loud)["information_ratio"], ratio(quiet)["information_ratio"])


def test_capacity_uses_the_information_ratio_without_the_decision_te_gate():
    import sys
    sys.path.insert(0, "scripts")
    from compare_v04 import capacity, ratio
    rng = np.random.default_rng(4)
    # A low-TE level: 1.6% tracking error, ratio about 0.3, at most half of the 1e6 ratio of 1.
    quiet = pd.Series(np.where(np.arange(800) % 2, 0.00102, -0.00098))
    assert np.isnan(ratio(quiet)["information_ratio"]) and 0 < ratio(quiet, 0)["information_ratio"] < 0.5
    rows = lambda irs, fb: pd.DataFrame({"cell": "S4", "model": "lightgbm", "capital": [1e6, 1e7, 1e8, 5e8, 1e9],
                                         "information_ratio": irs, "fallback": fb})
    assert capacity(rows([1.0, 0.9, ratio(quiet, 0)["information_ratio"], 0.4, 0.3], [0] * 5)) == {"S4/lightgbm": 1e8}
    assert capacity(rows([1.0, 0.9, 0.8, 0.7, 0.6], [0, 0.25, 0, 0, 0])) == {"S4/lightgbm": 1e7}
    assert capacity(rows([1.0, 0.9, 0.8, 0.7, 0.6], [0] * 5)) == {"S4/lightgbm": "above 1e+09"}
    assert capacity(rows([0.0, 0.9, 0.8, 0.7, 0.6], [0] * 5)) == {"S4/lightgbm": "undefined"}
    assert capacity(rows([1.0, np.nan, 0.8, 0.7, 0.6], [0] * 5)) == {"S4/lightgbm": 1e7}  # no ratio: not above half


def test_a_difference_carried_by_2024_q1_is_flagged():
    import sys
    sys.path.insert(0, "scripts")
    from compare_v04 import first_quarter_2024
    days = pd.bdate_range("2023-01-02", "2024-12-31")
    rng = np.random.default_rng(6)
    base = rng.normal(0, 0.005, len(days))
    q1 = (days >= "2024-01-01") & (days < "2024-04-01")
    left = pd.Series(base + np.where(q1, 0.01, -0.0003), index=days)
    right = pd.Series(base + rng.normal(0, 0.002, len(days)), index=days)
    flagged = first_quarter_2024(left, right)
    assert flagged["full"] > 0 and flagged["excluding_2024q1"] <= 0 and flagged["mainly_2024q1"]
    steady = first_quarter_2024(pd.Series(base + 0.001, index=days), right)
    assert steady["full"] > 0 and not steady["mainly_2024q1"]


def test_trailing_beta_hedge_uses_only_earlier_sessions():
    import sys
    sys.path.insert(0, "scripts")
    from compare_v04 import hedge
    days = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    r = 0.0005 + 2 * x + pd.Series(rng.normal(0, 0.002, len(days)), index=days)
    h, beta = hedge(r, x)
    assert abs(beta.dropna().mean() - 2) < 0.05 and abs(np.corrcoef(h, x.loc[h.index])[0, 1]) < 0.1
    assert h.mean() == pytest.approx(0.0005, abs=2e-4)
    # Changing a later index return leaves earlier betas unchanged.
    moved = x.copy()
    moved.iloc[400:] *= 3
    _, beta2 = hedge(r, moved)
    pd.testing.assert_series_equal(beta.iloc[:401], beta2.iloc[:401])
