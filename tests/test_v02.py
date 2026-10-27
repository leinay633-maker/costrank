"""v0.2: alias codes, curated delisting outcomes, weight-cap enforcement, benchmark and Holm."""
from dataclasses import replace
import hashlib
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from costrank.backtest import simulate
from costrank.config import Config
from costrank.data import ACTIONS, BARS, MASTER, Bundle, canonical_hash, read_bundle, synthetic, validate
from costrank.features import prepare
from costrank.metrics import holm, paired_block_interval, relative
from costrank.portfolio import allocate
from test_foundation import tiny


class Result:
    def __init__(self, fields, rows):
        self.fields, self.rows, self.error_code, self.error_msg = fields.split(","), iter(rows), "0", "ok"

    def next(self):
        self.row = next(self.rows, None)
        return self.row is not None

    def get_row_data(self):
        return self.row


def fake_baostock(monkeypatch, master, history, dates):
    fake = SimpleNamespace(login=lambda: SimpleNamespace(error_code="0"), logout=lambda: None,
        query_trade_dates=lambda **k: Result("calendar_date,is_trading_day", [[d, "1"] for d in dates]),
        query_stock_basic=lambda **k: Result("code,ipoDate,outDate", master),
        query_all_stock=lambda **k: Result("code", [[m[0]] for m in master]),
        query_history_k_data_plus=lambda code, fields, **k: Result(fields, history(code)))
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr("costrank.providers.time.sleep", lambda _: None)


@pytest.mark.parametrize("mode,alias", [("same", True), ("shifted", False), ("one_diff", True), ("two_diff", False)])
def test_collector_drops_code_whose_history_was_refiled(tmp_path, monkeypatch, mode, alias):
    from costrank.providers import collect_baostock
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-02", periods=105)]
    master = [["sz.000022", "1993-05-05", dates[102]], ["sz.001872", "1993-05-05", ""],
              ["sh.600000", "2010-01-01", ""]]
    # 102 traded days: one mismatch keeps 99.0% agreement, two drop it to 98.0%.
    differ = {"same": set(), "shifted": set(range(105)), "one_diff": {50}, "two_diff": {50, 60}}[mode]
    def history(code):
        rows = []
        for i, d in enumerate(dates):
            p = f"{10 + 0.01 * i + (1 if code == 'sz.001872' and i in differ else 0):.2f}"
            rows.append([d, code, p, p, p, p, p, "1000", "10000", "1", "0"])
        if code == "sz.000022":  # code change day: the old code only has a suspended placeholder
            rows = rows[:102] + [[dates[102], code, "", "", "", "", "", "", "", "0", "0"]]
        return rows
    fake_baostock(monkeypatch, master, history, dates)
    bundle = collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0)
    aliases = bundle.metadata["dropped_aliases"]
    if alias:
        assert [(x["symbol"], x["alias_of"]) for x in aliases] == [("sz.000022", "sz.001872")]
        assert aliases[0]["traded_rows"] == 102 and aliases[0]["matching_rows"] == 102 - len(differ)
        assert set(bundle.instruments.symbol) == {"sz.001872", "sh.600000"}
    else:
        assert aliases == []
        assert bundle.instruments.set_index("symbol").delisted.notna().sum() == 1


def test_alias_needs_the_new_code_on_every_traded_day_of_the_old_one():
    from costrank.providers import find_aliases
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-02", periods=25)]
    columns = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"]
    def rows(code, skip=None):
        return pd.DataFrame([[d, code, "10", "10", "10", "10", "10", "1000", "10000", "1", "0"]
                             for i, d in enumerate(dates) if i != skip], columns=columns)
    basic = pd.DataFrame({"code": ["sz.000022", "sz.001872"], "ipoDate": ["1993-05-05"] * 2, "outDate": [dates[22], ""]})
    frames = {"sz.000022": rows("sz.000022").head(22), "sz.001872": rows("sz.001872", skip=5)}
    assert find_aliases(basic, frames) == []
    frames["sz.001872"] = rows("sz.001872")
    assert [x["symbol"] for x in find_aliases(basic, frames)] == ["sz.000022"]
    basic.loc[1, "ipoDate"] = "1994-01-03"
    assert find_aliases(basic, frames) == []


def test_curated_resolution_is_sealed_hashed_and_must_match_removal_date(tmp_path, monkeypatch):
    from costrank.providers import collect_baostock
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    master = [["sh.600000", "2010-01-01", ""], ["sh.600001", "2010-01-01", dates[2]]]
    def history(code):
        rows = [[d, code, "10", "10", "10", "10", "10", "10000", "100000", "1", "0"] for d in dates]
        if code == "sh.600001":
            rows[2] = [dates[2], code, "10", "10", "10", "10", "10", "", "", "0", "0"]
        return rows
    fake_baostock(monkeypatch, master, history, dates)
    curated = tmp_path / "resolutions.csv"
    header = "symbol,delisted,kind,acquirer,ratio,cash_per_share,settle_date,source,note\n"
    curated.write_text(header + f"sh.600001,{dates[1]},share_exchange,sh.600000,0.5,,{dates[2]},test,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from provider"):
        collect_baostock(tmp_path / "bad", dates[0], dates[-1], pause=0, resolutions=curated)
    curated.write_text(header + f"sh.600001,{dates[2]},share_exchange,sh.600000,0.5,,{dates[2]},test,\n"
                       + f"sz.000999,{dates[2]},cash,,,3.0,{dates[2]},test,\n", encoding="utf-8")
    bundle = collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0, resolutions=curated)
    assert bundle.metadata["schema_version"] == 2 and "delistings.csv" in bundle.metadata["files"]
    assert bundle.metadata["delisting_resolutions"]["applied"] == 1
    assert bundle.metadata["delisting_resolutions"]["skipped"] == [{"symbol": "sz.000999", "reason": "not_removed_in_snapshot"}]
    row = read_bundle(tmp_path / "snap").delistings.iloc[0]
    assert (row.symbol, row.kind, row.acquirer, row.ratio, row.settle_date) == (
        "sh.600001", "share_exchange", "sh.600000", 0.5, pd.Timestamp(dates[2]))
    (tmp_path / "snap" / "delistings.csv").write_text("symbol\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_bundle(tmp_path / "snap")


def test_reference_resolutions_file_is_well_formed():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "reference" / "delisting_resolutions.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(table.columns) == ["symbol", "delisted", "kind", "acquirer", "ratio", "cash_per_share",
                                   "settle_date", "source", "note"]
    assert not table.symbol.duplicated().any() and table.kind.isin(["share_exchange", "cash"]).all()
    assert (pd.to_datetime(table.settle_date) >= pd.to_datetime(table.delisted)).all()
    swap = table[table.kind.eq("share_exchange")]
    assert (swap.ratio.astype(float) > 0).all() and swap.acquirer.str.fullmatch(r"(sh\.60|sz\.00)\d{4}").all()
    assert (table[table.kind.eq("cash")].cash_per_share.astype(float) > 0).all()
    assert table.source.str.startswith("http").all()


def delisted(kind, settle_at=6, config=None, **values):
    bundle, panel, cfg, pred = tiny()
    cfg = replace(cfg, **(config or {}))
    day, settle = bundle.calendar[4], bundle.calendar[settle_at]
    bundle.instruments.loc[bundle.instruments.symbol.eq("sh.600000"), "delisted"] = day
    row = dict(symbol="sh.600000", date=day, kind=kind, acquirer=None, ratio=np.nan,
               cash_per_share=np.nan, settle_date=settle)
    bundle.delistings = pd.DataFrame([{**row, **values}])
    panel = panel[~(panel.symbol.eq("sh.600000") & panel.date.ge(day))].copy()
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    return out, day, settle


def test_share_exchange_shares_are_held_locked_until_settlement():
    out, day, settle = delisted("share_exchange", acquirer="sz.000001", ratio=0.5)
    d = out["daily"].set_index("date")
    assert np.allclose(d.nav, 10000)  # flat prices: 900 shares x 10 become 450 shares x 20
    assert d.locked_value[day] == 9000 and d.locked_value[settle] == 0
    assert d.exposure[day] == pytest.approx(.9)  # the exchange shares are part of the book
    held = out["holdings"].set_index(["date", "symbol"])
    assert tuple(held.loc[(day, "sz.000001"), ["quantity", "locked"]]) == (450, 450)
    assert held.loc[(settle, "sz.000001"), "locked"] == 0
    e = out["events"]
    assert e.kind.tolist().count("delisting_share_exchange") == 1
    assert e[e.kind.eq("lock_released")].date.tolist() == [settle]
    assert "delisting_zero_recovery_assumption" not in e.kind.tolist()
    assert not out["orders"].side.eq("sell").any()


def test_locked_exchange_shares_are_seen_by_daily_rebalancing():
    # The acquirer is the only candidate left: v0.2's first cut bought it a second time while
    # its exchange shares were invisible, doubling the economic exposure.
    out, day, settle = delisted("share_exchange", settle_at=7, config={"rebalance": 1, "invested": .5},
                                acquirer="sz.000001", ratio=0.5)
    d = out["daily"].set_index("date")
    assert np.allclose(d.nav, 10000) and d.exposure.max() <= .5 + 1e-9
    o = out["orders"]
    assert not (o.symbol.eq("sz.000001") & o.filled_qty.gt(0)).any()
    held = out["holdings"]
    assert (held.quantity >= held.locked - 1e-9).all()
    assert held[held.date.lt(settle) & held.symbol.eq("sz.000001")].locked.eq(250).all()


def test_exchange_into_a_held_acquirer_settling_the_same_day():
    out, day, _ = delisted("share_exchange", settle_at=4, config={"topk": 2}, acquirer="sz.000001", ratio=0.5)
    d = out["daily"].set_index("date")
    assert np.allclose(d.nav, 10000)  # 400 x 10 + 200 x 20 + cash 2000 -> 400 x 20 + cash 2000
    held = out["holdings"].set_index(["date", "symbol"])
    assert tuple(held.loc[(day, "sz.000001"), ["quantity", "locked"]]) == (400, 0)
    e = out["events"]
    assert e[e.kind.eq("lock_released")].date.tolist() == [day]
    assert out["orders"].date.max() == out["daily"].date.iloc[1]


def test_cash_outcome_is_a_receivable_until_paid():
    out, day, settle = delisted("cash", cash_per_share=12.0)
    d = out["daily"].set_index("date")
    assert d.receivable[day] == 10800 and d.nav[day] == 11800
    assert d.cash[settle] == 11800 and d.receivable[settle] == 0
    assert out["events"].kind.tolist().count("receivable_paid") == 1


@pytest.mark.parametrize("change,message", [
    ({"date": 3}, "removal date"), ({"acquirer": "sh.600009"}, "known acquirer"),
    ({"settle_date": 3}, "settles before"), ({"kind": "rights"}, "known kind")])
def test_delisting_resolutions_are_validated(change, message):
    bundle, *_ = tiny()
    day = bundle.calendar[4]
    bundle.bars = bundle.bars[~(bundle.bars.symbol.eq("sh.600000") & bundle.bars.date.ge(day))]
    bundle.instruments.loc[bundle.instruments.symbol.eq("sh.600000"), "delisted"] = day
    row = dict(symbol="sh.600000", date=day, kind="share_exchange", acquirer="sz.000001", ratio=0.5,
               cash_per_share=np.nan, settle_date=bundle.calendar[6])
    bundle.delistings = pd.DataFrame([row])
    validate(bundle)
    row.update({k: bundle.calendar[v] if k in ("date", "settle_date") else v for k, v in change.items()})
    bundle.delistings = pd.DataFrame([row])
    with pytest.raises(ValueError, match=message):
        validate(bundle)


def test_resolved_delisting_label_uses_exchange_value(tmp_path):
    bundle = synthetic(tmp_path / "s", days=260, symbols=8)
    cfg = Config(dev_start="2018-10-01", dev_end="2018-10-31", holdout_start="2019-01-01",
        holdout_end="2019-01-31", train_days=100, valid_days=30, test_days=12, horizon=2,
        universe_size=8, topk=3, max_weight=.4)
    day = bundle.calendar[200]
    b = bundle.bars[~(bundle.bars.symbol.eq("sh.600000") & bundle.bars.date.ge(day))].copy()
    m = bundle.instruments.copy()
    m.loc[m.symbol.eq("sh.600000"), "delisted"] = day
    swap = pd.DataFrame([dict(symbol="sh.600000", date=day, kind="share_exchange", acquirer="sh.600001",
                              ratio=2.0, cash_per_share=np.nan, settle_date=day)])
    altered = Bundle(b, m, bundle.calendar, bundle.metadata, bundle.actions, swap)
    validate(altered)
    row = prepare(altered, cfg).set_index(["symbol", "date"]).loc[("sh.600000", bundle.calendar[198])]
    opens = altered.bars.set_index(["date", "symbol"]).open
    expected = 2 * opens[(bundle.calendar[201], "sh.600001")] / opens[(bundle.calendar[199], "sh.600000")] - 1
    assert row.label == pytest.approx(expected) and row.label > -0.5


def test_label_carries_the_old_close_until_the_acquirer_trades(tmp_path):
    bundle = synthetic(tmp_path / "s", days=260, symbols=8)
    cfg = Config(dev_start="2018-10-01", dev_end="2018-10-31", holdout_start="2019-01-01",
        holdout_end="2019-01-31", train_days=100, valid_days=30, test_days=12, horizon=2,
        universe_size=8, topk=3, max_weight=.4)
    cal, b = bundle.calendar, bundle.bars
    b = b[~((b.symbol.eq("sh.600000") & b.date.ge(cal[200])) | (b.symbol.eq("sh.600001") & b.date.lt(cal[203])))].copy()
    m = bundle.instruments.copy()
    m.loc[m.symbol.eq("sh.600000"), "delisted"] = cal[200]
    m.loc[m.symbol.eq("sh.600001"), "listed"] = cal[203]  # acquirer lists after the label ends
    swap = pd.DataFrame([dict(symbol="sh.600000", date=cal[200], kind="share_exchange", acquirer="sh.600001",
                              ratio=2.0, cash_per_share=np.nan, settle_date=cal[203])])
    altered = Bundle(b, m, cal, bundle.metadata, bundle.actions, swap)
    validate(altered)
    row = prepare(altered, cfg).set_index(["symbol", "date"]).loc[("sh.600000", cal[198])]
    bars = altered.bars.set_index(["date", "symbol"])
    assert row.label_end == cal[201]
    assert row.label == pytest.approx(bars.close[(cal[199], "sh.600000")] / bars.open[(cal[199], "sh.600000")] - 1)


@pytest.mark.parametrize("kind,values,expected", [
    (None, {}, -.5), ("cash", {"cash_per_share": 12.0}, .1),
    ("share_exchange", {"acquirer": "sz.000001", "ratio": .5}, 0.0)])
def test_benchmark_counts_a_removal_at_its_outcome(kind, values, expected):
    from costrank.experiment import benchmark_returns
    bundle, panel, *_ = tiny()
    day = bundle.calendar[4]
    bundle.instruments.loc[bundle.instruments.symbol.eq("sh.600000"), "delisted"] = day
    if kind:
        bundle.delistings = pd.DataFrame([{**dict(symbol="sh.600000", date=day, kind=kind, acquirer=None, ratio=np.nan,
                                                  cash_per_share=np.nan, settle_date=day), **values}])
    panel = panel[~(panel.symbol.eq("sh.600000") & panel.date.ge(day))].copy()
    panel["tr_close"] = (panel.close / panel.preclose).groupby(panel.symbol).cumprod() * 100
    bench = benchmark_returns(panel, bundle.calendar[1], bundle.calendar[-1], bundle)
    assert bench.loc[day, "benchmark_return"] == pytest.approx(expected) and bench.loc[day, "names"] == 2
    assert bench.drop(index=day).benchmark_return.abs().max() == 0


def test_holm_families_are_per_cost_scenario():
    from costrank.experiment import family_holm
    items = [dict(left=f"m__p{i}__{s}", p_bootstrap=p)
             for s, ps in (("slip_5bps", [.001, .04]), ("slip_10bps", [.02, .03])) for i, p in enumerate(ps)]
    family_holm(items)
    assert [x["holm_reject_5pct"] for x in items] == [True, True, True, True]
    assert {x["holm_family"] for x in items} == {2}
    assert holm([.001, .04, .02, .03]).tolist() == [True, False, False, False]  # one family of four


@pytest.mark.parametrize("mode", ["grandfather", "enforce"])
def test_weight_cap_mode_in_optimizer(mode):
    cfg = replace(Config(), topk=2, max_weight=.04, turnover_cap=.05, weight_cap_mode=mode)
    prev = pd.Series({"a": .08, "b": .03})
    sigma = pd.Series({"a": .02, "b": .02, "c": .02})
    w, info = allocate(pd.Series({"a": 2., "c": 1.}), prev, sigma, cfg, "turnover", "2024-01-02")
    assert info["status"] == "optimal"
    if mode == "enforce":
        assert w["a"] <= .04 + 1e-9 and info["compliance_trim"] == pytest.approx(.04)
        assert info["target_turnover"] <= .05 + .04 + 1e-7
    else:
        assert w["a"] == pytest.approx(.08) and info["compliance_trim"] == 0


@pytest.mark.parametrize("mode", ["grandfather", "enforce"])
def test_drifted_overweight_is_trimmed_only_when_enforced(mode):
    bundle, panel, cfg, pred = tiny(days=12)
    cfg = replace(cfg, rebalance=1, max_weight=.5, turnover_cap=.05, weight_cap_mode=mode)
    jump = panel.symbol.eq("sh.600000") & panel.date.ge(bundle.calendar[3])
    panel.loc[jump, ["open", "high", "low", "close"]] = 15.0
    panel.loc[jump & panel.date.gt(bundle.calendar[3]), "preclose"] = 15.0
    out = simulate(panel, pred, bundle, cfg, "turnover", zero_cost=True)
    d = out["daily"]
    assert d.max_weight.iloc[3] == pytest.approx(.6)
    if mode == "enforce":
        assert d.max_weight.iloc[-1] <= .5 + 1e-9
        assert d.turnover_limit.iloc[4] == pytest.approx(.05 + .1)
        assert (d.turnover_two_way <= d.turnover_limit.fillna(np.inf) + 1e-9).all()
    else:
        assert d.max_weight.iloc[-1] == pytest.approx(.6)


def test_relative_metrics_recover_beta_and_alpha():
    b = pd.Series(np.random.default_rng(0).normal(0, .01, 500))
    m = relative(.5 * b + .0002, b)
    assert m["beta"] == pytest.approx(.5) and m["correlation"] == pytest.approx(1)
    assert m["alpha_annual_arith"] == pytest.approx(.0002 * 252)


def test_holm_step_down_and_bootstrap_p_value():
    assert holm([.01, .04, .03, .2]).tolist() == [True, False, False, False]
    assert holm([.001, .01, .02]).tolist() == [True, True, True]
    idx = pd.bdate_range("2020-01-01", periods=300)
    noise = pd.Series(np.random.default_rng(1).normal(0, .001, 300), index=idx)
    noise -= noise.mean()
    zero = pd.Series(0.0, index=idx)
    assert paired_block_interval(noise + .002, zero, samples=200)["p_bootstrap"] == pytest.approx(2 / 200)
    assert paired_block_interval(noise, zero, samples=200)["p_bootstrap"] > .5


def test_v1_snapshot_has_no_delistings_and_keeps_its_prefix_hash(tmp_path):
    bundle = synthetic(tmp_path / "s", days=60, symbols=3)
    assert bundle.metadata["schema_version"] == 1 and bundle.delistings.empty
    date = pd.Timestamp("2018-03-01")
    m = bundle.instruments[bundle.instruments.listed < date].copy()
    m.loc[m.delisted >= date, "delisted"] = pd.NaT
    frames = [bundle.bars[bundle.bars.date < date][BARS], m[MASTER],
              pd.DataFrame({"date": bundle.calendar[bundle.calendar < date]}),
              bundle.actions[bundle.actions.date < date][ACTIONS]]
    v1 = hashlib.sha256((bundle.metadata["corporate_actions"] + "".join(canonical_hash(f) for f in frames)).encode())
    assert bundle.prefix_hash(date) == v1.hexdigest()
