from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from costrank.backtest import opening_block, price_band, simulate
from costrank.config import Config
from costrank.data import ACTIONS, Bundle, read_bundle, synthetic, validate
from costrank.experiment import check_freeze, freeze, run
from costrank.features import FEATURES, prepare, qlib_frame
from costrank.metrics import paired_block_interval, performance
from costrank.models import folds, predict_oos
from costrank.portfolio import allocate, fee_rates, fees


def tiny(days=10):
    dates = pd.bdate_range("2018-01-02", periods=days)
    frames = []
    for symbol, price in (("sh.600000", 10.0), ("sz.000001", 20.0)):
        frames.append(pd.DataFrame(dict(date=dates, symbol=symbol, open=price, high=price,
            low=price, close=price, preclose=price, volume=10_000_000, amount=100_000_000,
            tradable=1, is_st=0)))
    b = pd.concat(frames, ignore_index=True)
    m = pd.DataFrame(dict(symbol=["sh.600000", "sz.000001"], listed=pd.to_datetime(["2010-01-01"]*2), delisted=[pd.NaT]*2))
    a = pd.DataFrame(columns=ACTIONS)
    a["date"] = pd.to_datetime(a.date)
    a["pay_date"] = pd.to_datetime(a.pay_date)
    bundle = Bundle(b, m, dates, {"corporate_actions": "explicit_events", "source": "SYNTHETIC",
                                "end": dates[-1].strftime("%Y-%m-%d")}, a)
    validate(bundle)
    panel = bundle.bars.copy()
    panel["adv20"], panel["vol20"], panel["eligible"] = 100_000_000.0, 0.02, True
    cfg = Config(dev_start=str(dates[0].date()), dev_end=str(dates[-1].date()),
        holdout_start="2019-01-01", holdout_end="2019-12-31", horizon=2,
        universe_size=2, topk=1, max_weight=1.0, invested=0.9, initial_cash=10000,
        min_adv=0, rebalance=20, bootstrap_samples=20)
    pred = panel[["date", "symbol"]].copy()
    pred["score"] = np.where(pred.symbol.eq("sh.600000"), 1.0, 0.0)
    return bundle, panel, cfg, pred


@pytest.mark.parametrize("reference,st,expected", [(10, False, (9, 11)), (10, True, (9.5, 10.5)),
    (10.05, False, (9.05, 11.06)), (1.01, True, (0.96, 1.06))])
def test_price_tick_half_up(reference, st, expected):
    assert price_band(reference, st) == expected


@pytest.mark.parametrize("date,stamp,transfer", [("2022-04-28", .001, .00002),
    ("2022-04-29", .001, .00001), ("2023-08-25", .001, .00001), ("2023-08-28", .0005, .00001)])
def test_effective_dated_fee_schedule(date, stamp, transfer):
    assert fee_rates(date) == (stamp, transfer)


def test_minimum_commission_and_sell_only_tax():
    cfg = Config()
    buy = fees(1000, "buy", "2024-01-02", cfg)
    sell = fees(1000, "sell", "2024-01-02", cfg)
    assert buy == {"commission": 5, "stamp": 0, "transfer": .01}
    assert sell["stamp"] == .5
    assert sum(fees(0, "buy", "2024-01-02", cfg).values()) == 0
    assert sum(fees(1000, "sell", "2024-01-02", cfg, zero=True).values()) == 0


@pytest.mark.parametrize("kwargs", [{"max_weight": 0}, {"participation": 2}, {"rebalance": 0},
    {"horizon": -1}, {"commission": float("nan")}, {"models": ("unknown",)},
    {"policies": ()}, {"slippage_grid_bps": (-1,)}, {"holdout_end": "2026-01-01"}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_missing_row_is_not_a_suspension():
    bundle, *_ = tiny()
    bundle.bars = bundle.bars.drop(index=3)
    with pytest.raises(ValueError, match="Lifetime/calendar mismatch"):
        validate(bundle)


def test_explicit_suspension_can_carry_zero_quotes():
    bundle, *_ = tiny()
    bundle.bars.loc[3, ["open", "high", "low", "close", "preclose", "volume", "amount", "tradable"]] = 0
    validate(bundle)
    assert bundle.bars.loc[3, "close"] == 10


def test_duplicate_bar_fails():
    bundle, *_ = tiny()
    bundle.bars = pd.concat([bundle.bars, bundle.bars.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate(bundle)


def test_missing_flags_fail_closed():
    bundle, *_ = tiny()
    bundle.bars.loc[0, "is_st"] = np.nan
    with pytest.raises(ValueError, match="flags"):
        validate(bundle)


def test_flat_market_self_financing_next_open():
    bundle, panel, cfg, pred = tiny()
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert np.allclose(out["daily"].nav, cfg.initial_cash)
    assert out["daily"].iloc[0].holdings == 0
    assert (out["orders"].date > out["orders"].decision_date).all()
    assert (out["orders"].filled_qty % 100 == 0).all()


def test_cash_costs_reconcile_and_gross_not_double_counted():
    bundle, panel, cfg, pred = tiny()
    out = simulate(panel, pred, bundle, cfg, "topk", slippage_bps=5)
    day = out["daily"].iloc[1]
    charge = sum(day[x] for x in ("commission", "stamp", "transfer", "slippage"))
    assert day.nav == pytest.approx(cfg.initial_cash-charge)
    assert day.gross_same_trades_return == pytest.approx(0)
    assert day.net_return < 0
    assert out["orders"].iloc[0].filled_qty == 900


@pytest.mark.parametrize("halt", [False, True])
def test_limit_up_and_suspension_do_not_buy(halt):
    bundle, panel, cfg, pred = tiny()
    mask = panel.date.eq(bundle.calendar[1]) & panel.symbol.eq("sh.600000")
    if halt:
        panel.loc[mask, "tradable"] = 0
    else:
        panel.loc[mask, "open"] = 11
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert out["daily"].holdings.max() == 0
    assert out["orders"].iloc[0].reason == ("suspended" if halt else "limit_up")


def test_limit_down_preserves_unsold_position():
    bundle, panel, cfg, pred = tiny()
    cfg = replace(cfg, rebalance=1)
    pred.loc[pred.date >= bundle.calendar[1], "score"] = np.where(
        pred.loc[pred.date >= bundle.calendar[1], "symbol"].eq("sz.000001"), 1, 0)
    mask = panel.date.eq(bundle.calendar[2]) & panel.symbol.eq("sh.600000")
    panel.loc[mask, "open"] = 9
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    day = out["holdings"][out["holdings"].date.eq(bundle.calendar[2])]
    assert "sh.600000" in set(day.symbol)
    assert "limit_down" in set(out["orders"].reason)
    assert (out["daily"].cash >= -1e-7).all()


def test_target_quantity_does_not_use_tomorrow_open():
    bundle, panel, cfg, pred = tiny()
    baseline = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)["orders"]
    mask = panel.date.eq(bundle.calendar[1]) & panel.symbol.eq("sh.600000")
    panel.loc[mask, "open"] = 9.5
    changed = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)["orders"]
    assert baseline.iloc[0].filled_qty == changed.iloc[0].filled_qty == 900


def test_capacity_uses_decision_day_adv_not_same_day_volume():
    bundle, panel, cfg, pred = tiny()
    cfg = replace(cfg, participation=.001)
    panel.loc[panel.date.eq(bundle.calendar[0]), "adv20"] = 1_000_000
    panel.loc[panel.date.eq(bundle.calendar[1]), ["adv20", "amount", "volume"]] = 1e15
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert out["orders"].iloc[0].notional == 1000
    assert out["orders"].iloc[0].status == "partial"


@pytest.mark.parametrize("kind", ["split", "dividend", "proxy"])
def test_corporate_action_nav_continuity(kind):
    bundle, panel, cfg, pred = tiny()
    date = bundle.calendar[2]
    price = 5.0 if kind == "split" else 9.0
    mask = panel.symbol.eq("sh.600000") & panel.date.ge(date)
    panel.loc[mask, ["open", "high", "low", "close", "preclose"]] = price
    if kind == "proxy":
        bundle.metadata["corporate_actions"] = "reference_reinvestment_proxy"
    else:
        bundle.actions = pd.DataFrame([dict(date=date, symbol="sh.600000", split_ratio=2.0 if kind == "split" else 1.0,
            cash_per_share=0.0 if kind == "split" else 1.0, pay_date=bundle.calendar[4])])
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert np.allclose(out["daily"].nav, cfg.initial_cash)
    if kind == "dividend":
        assert out["daily"].iloc[2].receivable == 900
        assert out["daily"].iloc[4].receivable == 0
    if kind == "split":
        assert out["holdings"].query("date == @date and symbol == 'sh.600000'").iloc[0].quantity == 1800


def test_explicit_mode_refuses_unexplained_adjustment():
    bundle, *_ = tiny()
    bundle.bars.loc[2, "preclose"] = 9
    with pytest.raises(ValueError, match="Missing explicit"):
        validate(bundle)


def test_delisted_position_not_dropped_or_sold_at_stale_close():
    bundle, panel, cfg, pred = tiny()
    day = bundle.calendar[4]
    bundle.instruments.loc[bundle.instruments.symbol.eq("sh.600000"), "delisted"] = day
    panel = panel[~(panel.symbol.eq("sh.600000") & panel.date.ge(day))].copy()
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert out["daily"].iloc[-1].nav == 1000
    assert out["events"].iloc[0]["value"] == 9000
    assert "delisting_zero_recovery_assumption" in set(out["events"].kind)
    assert not out["orders"].side.eq("sell").any()


def test_optimizer_cost_cap_and_ineligible_exit():
    cfg = replace(Config(), topk=2, max_weight=.5, universe_size=3, turnover_cap=.2)
    scores = pd.Series({"b": 1, "c": .5})
    prev = pd.Series({"a": .8})
    w, audit = allocate(scores, prev, pd.Series({"a": .02, "b": .02, "c": .02}), cfg, "cost", "2024-01-02")
    assert (w >= -1e-8).all()
    assert abs(w-prev.reindex(w.index).fillna(0)).sum() <= .2 + 1e-7
    assert w["a"] < .8
    assert audit["status"] == "optimal"


def test_risk_constraint_and_optimizer_fallback(monkeypatch):
    cfg = replace(Config(), topk=2, max_weight=.6, risk_budget=.01)
    scores = pd.Series({"a": 2, "b": 1})
    vols = pd.Series({"a": .10, "b": .02})
    w, _ = allocate(scores, pd.Series(dtype=float), vols, cfg, "risk", "2024-01-02")
    assert (w*vols).sum() <= .01 + 1e-8
    import costrank.portfolio as module
    monkeypatch.setattr(module, "linprog", lambda *a, **k: SimpleNamespace(success=False, message="injected failure"))
    previous = pd.Series({"a": .2})
    w, audit = allocate(scores, previous, vols, cfg, "risk", "2024-01-02")
    assert w.equals(previous)
    assert audit["status"] == "hold_fallback"


def test_realized_turnover_cap_after_initial_build():
    bundle, panel, cfg, pred = tiny(days=20)
    cfg = replace(cfg, rebalance=1, turnover_cap=.2)
    for i, day in enumerate(bundle.calendar):
        mask = pred.date.eq(day)
        chosen = "sh.600000" if i % 2 == 0 else "sz.000001"
        pred.loc[mask, "score"] = np.where(pred.loc[mask, "symbol"].eq(chosen), 1, 0)
    out = simulate(panel, pred, bundle, cfg, "cost", slippage_bps=20)
    assert (out["daily"].iloc[2:].turnover_two_way <= .2+1e-7).all()
    assert (out["daily"].cash >= -1e-7).all()


@pytest.fixture(scope="module")
def historical_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixtures") / "development"
    bundle = synthetic(path, days=260, symbols=8)
    cfg = Config(dev_start="2018-10-01", dev_end="2018-10-31", holdout_start="2019-01-01",
        holdout_end="2019-01-31", train_days=100, valid_days=30, test_days=12, horizon=2,
        universe_size=8, topk=3, max_weight=.4, trees=30, threads=1,
        models=("factor", "ridge", "lightgbm"), policies=("topk",),
        slippage_grid_bps=(5.0,), bootstrap_samples=20)
    return path, bundle, cfg


def test_causal_features_future_perturbation(historical_fixture):
    _, bundle, cfg = historical_fixture
    baseline = prepare(bundle, cfg)
    changed = Bundle(bundle.bars.copy(), bundle.instruments.copy(), bundle.calendar,
                     bundle.metadata.copy(), bundle.actions.copy())
    cutoff = bundle.calendar[160]
    changed.bars.loc[changed.bars.date > cutoff, "close"] *= 2
    modified = prepare(changed, cfg)
    columns = ["date", "symbol", "eligible"]+["x_"+x for x in FEATURES]
    assert_frame_equal(baseline.loc[baseline.date <= cutoff, columns], modified.loc[modified.date <= cutoff, columns])


def test_label_uses_exchange_sessions_and_explicit_maturity(historical_fixture):
    _, bundle, cfg = historical_fixture
    panel = prepare(bundle, cfg)
    stock = panel[panel.symbol.eq("sh.600000")].reset_index(drop=True)
    assert stock.loc[100, "label"] == pytest.approx(stock.loc[103, "tr_open"] / stock.loc[101, "tr_open"] - 1)
    assert stock.loc[100, "label_end"] == stock.loc[103, "date"]
    assert stock.label.iloc[-3:].isna().all()


def test_st_filter_is_historical_not_current_name(historical_fixture):
    _, bundle, cfg = historical_fixture
    changed = Bundle(bundle.bars.copy(), bundle.instruments, bundle.calendar, bundle.metadata, bundle.actions)
    day = bundle.calendar[100]
    mask = changed.bars.symbol.eq("sh.600000") & changed.bars.date.eq(day)
    changed.bars.loc[mask, "is_st"] = 1
    panel = prepare(changed, cfg)
    assert not panel.loc[mask, "eligible"].any()
    assert panel.loc[panel.symbol.eq("sh.600000") & panel.date.eq(bundle.calendar[99]), "eligible"].all()


def test_folds_purge_actual_label_end_and_predictions_are_unique(historical_fixture):
    _, bundle, cfg = historical_fixture
    pred, audits = predict_oos(prepare(bundle, cfg), bundle.calendar, cfg)
    assert not pred.duplicated(["date", "symbol", "model"]).any()
    assert set(pred.model) == {"factor", "ridge", "lightgbm"}
    for item in audits:
        assert pd.Timestamp(item["train_max_label_end"]) < pd.Timestamp(item["train_label_before"])
        assert pd.Timestamp(item["valid_max_label_end"]) < pd.Timestamp(item["validation_label_before"])
    sets = [set(map(tuple, group[["date", "symbol"]].to_numpy())) for _, group in pred.groupby("model")]
    assert sets[0] == sets[1] == sets[2]


def test_insufficient_history_is_not_silently_shortened():
    with pytest.raises(ValueError, match="Insufficient prehistory"):
        folds(pd.bdate_range("2020-01-01", periods=20), Config())


def test_hash_tampering_and_holdout_guard(tmp_path, historical_fixture, monkeypatch):
    path = tmp_path / "corrupt"
    synthetic(path, days=80, symbols=2)
    with (path / "calendar.csv").open("a") as f:
        f.write("2019-01-01\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_bundle(path)
    original, bundle, cfg = historical_fixture
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Read holdout before gate"))
    with pytest.raises(ValueError, match="containing the holdout"):
        read_bundle(original, allow_after="2018-06-01")


def test_bootstrap_pairing_and_initial_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=40)
    a = pd.Series(.001, index=idx)
    stats = paired_block_interval(a, a, samples=20)
    assert stats["ci95_low"] == stats["ci95_high"] == 0
    with pytest.raises(ValueError, match="identical"):
        paired_block_interval(a, a.iloc[:-1])
    bundle, panel, cfg, pred = tiny()
    d = simulate(panel, pred, bundle, cfg, "topk")["daily"]
    d.loc[0, "net_return"] = -.5
    assert performance(d)["max_drawdown"] <= -.5


def test_full_pipeline_manifest_freeze_and_final_receipt(tmp_path, historical_fixture):
    path, bundle, cfg = historical_fixture
    cfg = replace(cfg, models=("factor",))
    first = run(cfg, path, tmp_path / "runs")
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["evidence"] == "synthetic_only"
    assert (first / "summary.csv").exists()
    lock = freeze(tmp_path / "freeze.json", cfg, path, first)
    with pytest.raises(FileExistsError):
        freeze(lock, cfg, path, first)
    with pytest.raises(ValueError, match="mismatch"):
        check_freeze(lock, replace(cfg, seed=99))
    final_path = tmp_path / "final"
    final = synthetic(final_path, days=283, symbols=8)
    assert bundle.prefix_hash(cfg.holdout_start) == final.prefix_hash(cfg.holdout_start)
    result = run(cfg, final_path, tmp_path / "runs", holdout=True, freeze_file=lock)
    assert json.loads((result / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"
    with pytest.raises(RuntimeError, match="retained evidence"):
        run(cfg, final_path, tmp_path / "runs", holdout=True, freeze_file=lock)
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "runs").glob("*/manifest.json")]
    assert sum(x["status"] == "failed" for x in manifests) == 1


def test_qlib_dataframe_contract(historical_fixture):
    _, bundle, cfg = historical_fixture
    frame = qlib_frame(prepare(bundle, cfg))
    assert frame.index.names == ["datetime", "instrument"]
    assert ("label", "LABEL0") in frame.columns
    assert list(frame["feature"].columns) == FEATURES


def test_optional_native_qlib(historical_fixture):
    pytest.importorskip("qlib")
    from qlib.data.dataset import DataHandlerLP
    _, bundle, cfg = historical_fixture
    frame = qlib_frame(prepare(bundle, cfg))
    handler = DataHandlerLP.from_df(frame)
    assert len(handler.fetch()) == len(frame)


def test_leading_explicit_suspension_is_unknown_not_backfilled():
    bundle, *_ = tiny()
    bundle.bars.loc[0, ["open", "high", "low", "close", "preclose", "volume", "amount", "tradable"]] = 0
    validate(bundle)
    assert pd.isna(bundle.bars.loc[0, "close"])
    assert bundle.bars.loc[1, "close"] == 10


def test_delisting_tail_has_negative_one_label(historical_fixture):
    _, bundle, cfg = historical_fixture
    day = bundle.calendar[200]
    b = bundle.bars[~(bundle.bars.symbol.eq("sh.600000") & bundle.bars.date.ge(day))].copy()
    m = bundle.instruments.copy()
    m.loc[m.symbol.eq("sh.600000"), "delisted"] = day
    altered = Bundle(b, m, bundle.calendar, bundle.metadata, bundle.actions)
    validate(altered)
    panel = prepare(altered, cfg)
    row = panel[panel.symbol.eq("sh.600000") & panel.date.eq(bundle.calendar[198])].iloc[0]
    assert row.label == -1
    assert row.label_end == bundle.calendar[201]


def test_false_explicit_dividend_does_not_create_free_cash():
    bundle, *_ = tiny()
    bundle.actions = pd.DataFrame([dict(date=bundle.calendar[2], symbol="sh.600000",
        split_ratio=1, cash_per_share=1, pay_date=bundle.calendar[2])])
    with pytest.raises(ValueError, match="does not reconcile"):
        validate(bundle)


def test_suspended_label_endpoint_not_filtered_using_future_flag(historical_fixture):
    _, bundle, cfg = historical_fixture
    changed = Bundle(bundle.bars.copy(), bundle.instruments, bundle.calendar, bundle.metadata, bundle.actions)
    mask = changed.bars.symbol.eq("sh.600000") & changed.bars.date.eq(bundle.calendar[101])
    changed.bars.loc[mask, "tradable"] = 0
    panel = prepare(changed, cfg)
    row = panel[panel.symbol.eq("sh.600000") & panel.date.eq(bundle.calendar[100])].iloc[0]
    assert pd.notna(row.label)


@pytest.mark.parametrize("policy", ["risk", "penalty", "turnover", "cost"])
def test_lp_ablation_constraints(policy):
    cfg = replace(Config(), topk=2, max_weight=.5, turnover_cap=.1)
    prev = pd.Series({"a": .8})
    sigma = pd.Series({"a": .02, "b": .02, "c": .02})
    w, info = allocate(pd.Series({"b": 2., "c": 1.}), prev, sigma, cfg, policy, "2024-01-02")
    assert info["status"] == "optimal"
    assert (w >= 0).all() and w.sum() <= cfg.invested+1e-7
    if policy in {"turnover", "cost"}:
        assert info["target_turnover"] <= cfg.turnover_cap+1e-7


def test_free_collector_contract_resume_and_no_success_only_sampling(tmp_path, monkeypatch):
    import sys
    from costrank.providers import collect_baostock
    calls = {"bars": 0, "logout": 0}
    dates = ["2020-01-02", "2020-01-03"]
    class Result:
        def __init__(self, fields, rows, error="0"):
            self.fields = fields.split(",")
            self.rows = iter(rows)
            self.error_code, self.error_msg = error, "injected" if error != "0" else "ok"
        def next(self):
            self.row = next(self.rows, None)
            return self.row is not None
        def get_row_data(self):
            return self.row
    def history(code, fields, **kwargs):
        calls["bars"] += 1
        assert kwargs["adjustflag"] == "3"
        if calls["bars"] == 1:
            return Result(fields, [], error="network_test")
        return Result(fields, [[d, code, "10", "10", "10", "10", "10", "10000", "100000", "1", "0"] for d in dates])
    def logout():
        calls["logout"] += 1
    fake = SimpleNamespace(login=lambda: SimpleNamespace(error_code="0"), logout=logout,
        query_trade_dates=lambda **k: Result("calendar_date,is_trading_day", [[d, "1"] for d in dates]),
        query_stock_basic=lambda **k: Result("code,ipoDate,outDate", [["sh.600000", "2010-01-01", ""]]),
        query_all_stock=lambda **k: Result("code,tradeStatus,code_name", [["sh.600000", "1", "fixture"]]),
        query_history_k_data_plus=history)
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr("costrank.providers.time.sleep", lambda _: None)
    root = tmp_path / "acquisition"
    bundle = collect_baostock(root, "2020-01-02", "2020-01-03", pause=0)
    assert len(bundle.bars) == 2 and calls["bars"] == 2 and calls["logout"] >= 2
    assert (root / "collection-failures.jsonl").exists()
    assert bundle.metadata["corporate_actions"] == "reference_reinvestment_proxy"
    with pytest.raises(FileExistsError, match="Sealed snapshot"):
        collect_baostock(root, "2020-01-02", "2020-01-03", pause=0)


@pytest.mark.parametrize("status", ["0", "1"])
def test_collector_drops_suspended_placeholder_on_out_date(tmp_path, monkeypatch, status):
    import sys
    from costrank.providers import collect_baostock
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    class Result:
        def __init__(self, fields, rows):
            self.fields, self.rows, self.error_code, self.error_msg = fields.split(","), iter(rows), "0", "ok"
        def next(self):
            self.row = next(self.rows, None)
            return self.row is not None
        def get_row_data(self):
            return self.row
    def history(code, fields, **kwargs):
        rows = [[d, code, "10", "10", "10", "10", "10", "10000", "100000", "1", "0"] for d in dates]
        if code == "sh.600001":  # delisted on dates[2]; BaoStock still sends that day, volume blank
            rows[2] = [dates[2], code, "10", "10", "10", "10", "10", "", "", status, "0"]
        return Result(fields, rows)
    fake = SimpleNamespace(login=lambda: SimpleNamespace(error_code="0"), logout=lambda: None,
        query_trade_dates=lambda **k: Result("calendar_date,is_trading_day", [[d, "1"] for d in dates]),
        query_stock_basic=lambda **k: Result("code,ipoDate,outDate",
            [["sh.600000", "2010-01-01", ""], ["sh.600001", "2010-01-01", dates[2]]]),
        query_all_stock=lambda **k: Result("code", [["sh.600000"], ["sh.600001"]]),
        query_history_k_data_plus=history)
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr("costrank.providers.time.sleep", lambda _: None)
    if status == "1":
        with pytest.raises(ValueError, match="outDate"):
            collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0)
        return
    bundle = collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0)
    assert bundle.bars.groupby("symbol").size().to_dict() == {"sh.600000": 3, "sh.600001": 2}


# Middle-day row of sh.600001 as BaoStock sent it: open,high,low,close,preclose,volume,amount,tradestatus.
@pytest.mark.parametrize("row,expected", [
    (["10", "10", "10", "10", "10", "", "", "1"], ("active_flag_without_trades", 0, 0, 0)),
    (["10", "10", "10", "10", "10", "10000", "0.0000", "0"], ("stale_volume_on_suspended_row", 0, 0, 0)),
    (["10", "10.5", "9.8", "10.2", "10", "10000", "101000", "0"], ("suspended_flag_with_trades", 1, 10000, 101000)),
    (["10", "10.5", "9.8", "10.2", "10", "", "", "1"], "Invalid volume/amount"),
    (["10", "10.5", "9.8", "10.2", "10", "10000", "200000", "0"], "Suspended row has nonzero"),
    (["10", "10", "10", "10", "10", "10000", "100000", "0"], "Suspended row has nonzero"),
])
def test_collector_repairs_only_self_evident_provider_rows(tmp_path, monkeypatch, row, expected):
    import sys
    from costrank.providers import collect_baostock
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    class Result:
        def __init__(self, fields, rows):
            self.fields, self.rows, self.error_code, self.error_msg = fields.split(","), iter(rows), "0", "ok"
        def next(self):
            self.row = next(self.rows, None)
            return self.row is not None
        def get_row_data(self):
            return self.row
    def history(code, fields, **kwargs):
        rows = [[d, code, "10", "10", "10", "10", "10", "10000", "100000", "1", "0"] for d in dates]
        if code == "sh.600001":
            rows[1] = [dates[1], code, *row, "0"]
        return Result(fields, rows)
    fake = SimpleNamespace(login=lambda: SimpleNamespace(error_code="0"), logout=lambda: None,
        query_trade_dates=lambda **k: Result("calendar_date,is_trading_day", [[d, "1"] for d in dates]),
        query_stock_basic=lambda **k: Result("code,ipoDate,outDate",
            [["sh.600000", "2010-01-01", ""], ["sh.600001", "2010-01-01", ""]]),
        query_all_stock=lambda **k: Result("code", [["sh.600000"], ["sh.600001"]]),
        query_history_k_data_plus=history)
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr("costrank.providers.time.sleep", lambda _: None)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0)
        return
    bundle = collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0)
    rule, tradable, volume, amount = expected
    fixed = bundle.bars.set_index(["symbol", "date"]).loc[("sh.600001", pd.Timestamp(dates[1]))]
    assert (fixed.tradable, fixed.volume, fixed.amount) == (tradable, volume, amount)
    repairs = bundle.metadata["provider_repairs"]
    assert [(x["rule"], x["symbol"], x["date"], x["tradestatus"]) for x in repairs] == [
        (rule, "sh.600001", dates[1], row[7])]
    assert len(bundle.bars) == 6 and bundle.bars.tradable.sum() == 5 + tradable


def test_baostock_closed_peer_fails_instead_of_spinning(monkeypatch):
    import socket
    import threading
    bs = pytest.importorskip("baostock")
    from baostock.common import contants
    from costrank.providers import guard_closed_socket
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    def gateway():
        for _ in range(4):
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            conn.close()
    threading.Thread(target=gateway, daemon=True).start()
    monkeypatch.setattr(contants, "BAOSTOCK_SERVER_IP", "127.0.0.1")
    monkeypatch.setattr(contants, "BAOSTOCK_SERVER_PORT", server.getsockname()[1])
    guard_closed_socket()
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("login", bs.login()), daemon=True)
    worker.start()
    worker.join(10)
    server.close()
    assert not worker.is_alive(), "baostock login spun on a closed socket"
    assert result["login"].error_code != "0"


@pytest.mark.parametrize("name,values", [("models", ("factor", "factor")),
    ("policies", ("risk", "risk")), ("slippage_grid_bps", (5, 5))])
def test_duplicate_scenarios_rejected(name, values):
    with pytest.raises(ValueError, match="Duplicate"):
        Config(**{name: values})


def test_st_change_cancels_buy_but_does_not_prevent_sale():
    row = SimpleNamespace(tradable=1, is_st=1, preclose=10, open=10)
    assert opening_block(row, "buy") == "st_at_execution"
    assert opening_block(row, "sell") is None


def test_manifest_must_bind_every_canonical_file(tmp_path):
    path = tmp_path / "sealed"
    synthetic(path, days=65, symbols=2)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    del meta["files"]["bars.csv.gz"]
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="every canonical file"):
        read_bundle(path)
