"""v0.4 signal layer: every registered factor is causal, processing is neutral, selection uses
only the training window and prunes sign-aligned duplicates."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from costrank.config import Config
from costrank.data import synthetic
from costrank.factors import REGISTRY, market, raw_factors
from costrank.features import prepare
from costrank.signals import (SignalSet, benjamini_hochberg, cross_rank, daily_rank_ic, industry_groups,
                              newey_west_t, select, standardize)


def library_bundle(tmp_path, *, perturb_after=None, seed=3):
    bundle = synthetic(tmp_path, days=320, symbols=40, seed=seed)
    b = bundle.bars
    rng = np.random.default_rng(11)
    attrs = pd.DataFrame({"date": b.date, "symbol": b.symbol, "turn": rng.uniform(0.5, 5, len(b)),
                          "pe_ttm": rng.normal(20, 15, len(b)), "pb_mrq": rng.uniform(-1, 6, len(b)),
                          "ps_ttm": rng.uniform(0.5, 9, len(b)), "pcf_ttm": rng.normal(10, 20, len(b))})
    symbols = sorted(b.symbol.unique())
    months = sorted(b.date.groupby(b.date.dt.to_period("M")).min())
    industries = pd.DataFrame([{"date": d, "symbol": s, "industry": ("C39计算机" if i % 3 else "J66金融"),
                                "update_date": d} for d in months for i, s in enumerate(symbols)])
    pubs = pd.to_datetime(np.sort(rng.choice(b.date.unique(), 60)))
    forecasts = pd.DataFrame({"symbol": rng.choice(symbols, 60), "pub_date": pubs, "stat_date": pubs,
                              "kind": rng.choice(["预增", "预减", "不确定"], 60), "chg_up": rng.normal(20, 30, 60),
                              "chg_down": rng.normal(0, 30, 60)})
    express = pd.DataFrame({"symbol": rng.choice(symbols, 60), "pub_date": pubs, "stat_date": pubs,
                            "update_date": pubs, "available_date": pubs, **{c: rng.normal(0, 1, 60) for c in
                            ("total_asset", "net_asset", "eps_chg_pct", "roe_wa", "eps_diluted", "gr_yoy", "op_yoy")}})
    if perturb_after is not None:
        later = b.date > perturb_after
        b[["volume", "amount"]] = b[["volume", "amount"]].astype(float)
        noise = np.random.default_rng(seed + 100).uniform(0.8, 1.25, later.sum())
        for col in ("open", "high", "low", "close", "preclose"):
            b.loc[later, col] = b.loc[later, col] * noise
        b.loc[later, ["volume", "amount"]] = b.loc[later, ["volume", "amount"]].mul(noise[::-1], axis=0)
        b.loc[later, "high"] = b.loc[later, ["open", "high", "low", "close"]].max(axis=1)
        b.loc[later, "low"] = b.loc[later, ["open", "high", "low", "close"]].min(axis=1)
        attrs.loc[attrs.date > perturb_after, ["turn", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]] *= 1.7
        industries.loc[industries.date > perturb_after, "industry"] = "K70房地产"
        forecasts.loc[forecasts.pub_date >= perturb_after, "chg_up"] += 50
        express.loc[express.pub_date >= perturb_after, "eps_chg_pct"] += 5
    bundle.tables = {"attributes": attrs, "industries": industries, "forecasts": forecasts, "express": express}
    return bundle


def config(**kw):
    return Config(dev_start="2018-06-01", dev_end="2019-03-01", holdout_start="2020-01-01",
                  holdout_end="2020-12-31", universe_size=30, topk=5, min_adv=0, **kw)


def test_every_registered_factor_ignores_data_after_its_session(tmp_path):
    cut = pd.Timestamp("2018-11-30")
    cfg = config()
    full = library_bundle(tmp_path / "a")
    moved = library_bundle(tmp_path / "b", perturb_after=cut)
    m1, m2 = market(prepare(full, cfg), full), market(prepare(moved, cfg), moved)
    assert len(REGISTRY) >= 150
    checked = 0
    for (name, _, a), (_, _, b) in zip(raw_factors(m1), raw_factors(m2)):
        x, y = a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy()
        same = (x == y) | (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
        assert same.all(), f"{name} changed before {cut.date()} when later data changed"
        later = a.loc[cut:].iloc[1:].to_numpy()
        checked += np.isfinite(later).any()
    assert checked >= 150  # nearly every factor is actually defined on the synthetic panel


def test_standardize_is_neutral_to_industry_and_size_and_keeps_missing():
    rng = np.random.default_rng(0)
    n, days = 300, 3
    dates = np.repeat(pd.date_range("2020-01-01", periods=days).to_numpy(), n)
    groups = np.tile(np.where(np.arange(n) % 3 == 0, "A", np.where(np.arange(n) % 3 == 1, "B", "C")), days)
    size = rng.normal(10, 1, n * days)
    raw = np.column_stack([size * 2 + (groups == "A") * 3 + rng.normal(0, 1, n * days), rng.standard_t(2, n * days)])
    raw[5, 1] = np.nan
    out = standardize(raw, dates, groups, size).astype(float)
    assert np.isnan(out[5, 1]) and np.isfinite(out[6, 1])
    for d in np.unique(dates):
        rows = dates == d
        X = np.column_stack([pd.get_dummies(groups[rows]).to_numpy(float), size[rows]])
        for j in range(2):
            v = np.isfinite(out[rows, j])
            assert abs(np.nanmean(out[rows, j])) < 1e-5 and abs(np.nanstd(out[rows, j]) - 1) < 1e-4
            beta = np.linalg.lstsq(X[v], out[rows, j][v], rcond=None)[0]
            assert np.abs(beta).max() < 1e-4


def test_industry_groups_use_latest_snapshot_and_merge_small_groups(tmp_path):
    bundle = library_bundle(tmp_path / "snap")
    industries = bundle.tables["industries"]
    first = industries.date.min()
    industries.loc[(industries.date == first) & (industries.symbol == "sh.600000"), "industry"] = "R85新闻"
    rows = pd.DataFrame({"date": [first, first + pd.Timedelta(1, unit="D")] * 40,
                         "symbol": sorted(industries.symbol.unique()) * 2})
    g = industry_groups(bundle, rows)
    assert set(g) <= {"C39", "J", "C", "other"}
    assert g[(rows.symbol == "sh.600000") & (rows.date == first)].iloc[0] == "other"  # one name in R


def test_newey_west_and_benjamini_hochberg():
    x = np.random.default_rng(1).normal(0.5, 1, (500, 1))
    t0 = newey_west_t(x, lag=0)[0]
    assert t0 == pytest.approx(x.mean() / (x.std() / np.sqrt(len(x))), rel=1e-12)
    assert benjamini_hochberg([0.001, 0.02, 0.03, 0.5, np.nan], 0.10).tolist() == [True, True, True, False, False]
    assert benjamini_hochberg([0.2, 0.3], 0.10).tolist() == [False, False]


def test_selection_uses_training_dates_and_prunes_sign_aligned_duplicates():
    rng = np.random.default_rng(2)
    sessions = pd.date_range("2020-01-01", periods=200)
    n = 60
    dates = np.repeat(sessions.to_numpy(), n)
    signal = rng.normal(0, 1, len(dates))
    target = cross_rank(signal + rng.normal(0, 1, len(dates)), dates)
    X = np.column_stack([signal, -signal + rng.normal(0, 0.01, len(dates)), rng.normal(0, 1, len(dates)),
                         np.where(dates < sessions[100].to_datetime64(), rng.normal(0, 1, len(dates)), signal)])
    names = ["good", "mirror", "noise", "late"]
    ic = daily_rank_ic(X, target, dates)
    ic.columns = names
    from costrank.signals import correlation_samples
    s = SignalSet(pd.RangeIndex(len(dates)), dates, names, X.astype(np.float32), ic, correlation_samples(X, dates),
                  pd.Series(["A"] * len(dates)), np.zeros(len(dates)))
    train = dates < sessions[100].to_datetime64()
    chosen = select(s, train, sessions[:100], config(), names)
    assert chosen["selected"] == ["good"] or chosen["selected"] == ["mirror"]
    assert "late" not in chosen["selected"]  # informative only after the training window
    assert chosen["passed_fdr"] >= 2


def test_small_manufacturing_groups_merge_into_c_and_snapshot_mode_first(tmp_path):
    from costrank.signals import static_industries
    bundle = library_bundle(tmp_path / "snap")
    ind = bundle.tables["industries"]
    first = ind.date.min()
    ind.loc[(ind.date == first) & ind.symbol.isin(["sh.600000", "sh.600001"]), "industry"] = "C16烟草"
    mapping = static_industries(bundle, first)
    assert mapping("C16烟草") == "C" and mapping("C39计算机") == "C39" and mapping("R85新闻") == "other"
    rows = pd.DataFrame({"date": [ind.date.max()] * 2, "symbol": ["sh.600000", "sh.600003"]})
    ind.loc[(ind.date == ind.date.max()) & (ind.symbol == "sh.600003"), "industry"] = "K70房地产"
    assert industry_groups(bundle, rows, "first").tolist() == industry_groups(bundle, rows.assign(date=first)).tolist()


def test_daily_rank_ic_reranks_on_the_joint_subset():
    dates = np.repeat(np.datetime64("2020-01-02"), 30)
    target = np.arange(30, dtype=float)
    X = np.column_stack([np.where(np.arange(30) % 3 == 0, np.arange(30.0), np.nan), np.arange(30.0)])
    ic = daily_rank_ic(X, target, dates)
    assert ic.iloc[0, 1] == pytest.approx(1.0)
    assert np.isnan(ic.iloc[0, 0])  # only 10 names share values: below the 20-name minimum
    X[:, 0] = np.where(np.arange(30) % 3 == 2, np.nan, np.arange(30.0))
    assert daily_rank_ic(X, target, dates).iloc[0, 0] == pytest.approx(1.0)


def test_revised_express_values_wait_for_their_update(tmp_path):
    cfg = config()
    bundle = library_bundle(tmp_path / "a")
    dates = pd.DatetimeIndex(sorted(bundle.bars.date.unique()))
    symbol = sorted(bundle.bars.symbol.unique())[0]
    pub, update = dates[100], dates[130]
    ex = bundle.tables["express"].iloc[:1].copy()
    ex[["symbol", "pub_date", "stat_date", "update_date", "available_date", "eps_chg_pct"]] = \
        [symbol, pub, pub, update, update, 0.7]
    bundle.tables["express"] = ex
    values = market(prepare(bundle, cfg), bundle).events["ex_eps_chg_pct_60"][symbol]
    assert values[values.index <= update].isna().all()  # not between publication and revision
    assert values[values.index > update].iloc[0] == pytest.approx(0.7)
