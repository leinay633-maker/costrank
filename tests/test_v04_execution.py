"""v0.4 execution: board price limits and lots, hybrid corporate actions, FIFO dividend tax,
first-30-minute VWAP fills and square-root impact."""
from dataclasses import replace
from decimal import Decimal
import numpy as np
import pandas as pd
import pytest
from costrank.backtest import (dividend_tax_rate, execution_views, limit_rate, price_band, round_buy,
                               round_sell, simulate)
from costrank.data import validate
from test_foundation import tiny


@pytest.mark.parametrize("symbol,day,st,rate", [
    ("sh.600000", "2021-03-01", 0, "0.10"), ("sz.000001", "2021-03-01", 1, "0.05"),
    ("sz.300750", "2020-08-21", 0, "0.10"), ("sz.300750", "2020-08-21", 1, "0.05"),
    ("sz.300750", "2020-08-24", 0, "0.20"), ("sz.300750", "2020-08-24", 1, "0.20"),
    ("sh.688981", "2019-07-22", 1, "0.20"), ("sh.600000", "2026-07-03", 1, "0.05"),
    ("sz.000001", "2026-07-06", 1, "0.10"), ("sz.300750", "2026-07-06", 1, "0.20")])
def test_limit_rate_by_board_and_date(symbol, day, st, rate):
    assert limit_rate(symbol, pd.Timestamp(day), st) == Decimal(rate)


def test_band_keeps_at_least_one_tick_from_the_reference():
    assert price_band(10, False) == (9.0, 11.0)
    assert price_band(20, False, Decimal("0.20")) == (16.0, 24.0)
    assert price_band(0.10, True) == (0.09, 0.11)  # 0.095 rounds back to the reference
    assert price_band(0.01, True) == (0.01, 0.02)


def test_star_lots_start_at_200_and_step_by_one_share():
    assert round_buy("sh.688001", 199.6) == 0 and round_buy("sh.688001", 250.7) == 250
    assert round_buy("sh.600000", 250.7) == 200 and round_buy("sz.300001", 99.9) == 0
    assert round_sell("sh.688001", 350, 500) == 300  # a balance under 200 could not be sold later
    assert round_sell("sh.688001", 150, 300) == 0
    assert round_sell("sh.688001", 150, 150) == 150 and round_sell("sh.688001", 150.5, 150.5) == 150.5
    assert round_sell("sh.600000", 120, 550) == 100 and round_sell("sh.600000", 550, 550) == 550


@pytest.mark.parametrize("sold,rate", [("2020-02-15", 0.20), ("2020-02-16", 0.10), ("2021-01-15", 0.10),
                                       ("2021-01-16", 0.0)])
def test_dividend_tax_rate_counts_calendar_months_to_the_day_before_sale(sold, rate):
    assert dividend_tax_rate(pd.Timestamp("2020-01-15"), pd.Timestamp(sold)) == rate


def star_tiny():
    bundle, panel, cfg, pred = tiny()
    for frame in (bundle.bars, panel, pred):
        frame.loc[frame.symbol.eq("sz.000001"), "symbol"] = "sh.688001"
    bundle.instruments.loc[bundle.instruments.symbol.eq("sz.000001"), "symbol"] = "sh.688001"
    pred["score"] = np.where(pred.symbol.eq("sh.688001"), 1.0, 0.0)
    return bundle, panel, cfg, pred


def test_star_position_is_not_rounded_to_hundreds():
    bundle, panel, cfg, pred = star_tiny()
    bundle.metadata["schema_version"] = 3
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert out["orders"].iloc[0].filled_qty == 450  # 0.9 * 10000 / 20


def dividend_case(tax, mode="hybrid_explicit_proxy"):
    """Buy sh.600000 at 10 on day 1, 1.0 cash dividend on day 2, sell at 9 on day 4."""
    bundle, panel, cfg, pred = tiny()
    day = bundle.calendar[2]
    for frame in (bundle.bars, panel):
        mask = frame.symbol.eq("sh.600000") & frame.date.ge(day)
        frame.loc[mask, ["open", "high", "low", "close", "preclose"]] = 9.0
    bundle.metadata["corporate_actions"] = mode
    bundle.actions = pd.DataFrame([dict(date=day, symbol="sh.600000", split_ratio=1.0, cash_per_share=1.0,
                                        pay_date=bundle.calendar[3], taxable_stock_per_share=0.0)])
    validate(bundle)
    pred.loc[pred.date >= bundle.calendar[3], "score"] = np.where(
        pred.loc[pred.date >= bundle.calendar[3], "symbol"].eq("sz.000001"), 1.0, 0.0)
    cfg = replace(cfg, rebalance=3, dividend_tax=tax)
    return bundle, panel, cfg, pred


def test_fifo_dividend_tax_is_paid_on_sale_within_a_month():
    bundle, panel, cfg, pred = dividend_case(True)
    out = simulate(panel, pred, bundle, cfg, "topk", slippage_bps=0)
    sale = out["orders"].query("symbol == 'sh.600000' and side == 'sell'").iloc[0]
    assert sale.filled_qty == 900 and sale.dividend_tax == pytest.approx(0.20 * 900)
    d = out["daily"].set_index("date")
    assert d.loc[bundle.calendar[2], "dividend_tax_accrued"] == pytest.approx(180)
    assert d.loc[sale.date, "dividend_tax"] == pytest.approx(180)
    zero = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    assert zero["orders"].dividend_tax.sum() == 0
    untaxed = simulate(panel, pred, bundle, replace(cfg, dividend_tax=False), "topk", slippage_bps=0)
    assert "dividend_tax" not in untaxed["orders"]
    gap = untaxed["daily"].nav.iloc[-1] - out["daily"].nav.iloc[-1]
    assert gap == pytest.approx(180, abs=1e-6)


def test_split_shares_cannot_be_sold_before_they_list():
    """900 shares of sh.600000 at 10; a 10-for-10 bonus issue (taxable at face value) goes ex on day 4,
    the day the book sells them, and the new shares list on day 5."""
    bundle, panel, cfg, pred = tiny()
    ex, listing = bundle.calendar[4], bundle.calendar[5]
    for frame in (bundle.bars, panel):
        mask = frame.symbol.eq("sh.600000") & frame.date.ge(ex)
        frame.loc[mask, ["open", "high", "low", "close", "preclose"]] = 5.0
    bundle.metadata["corporate_actions"] = "hybrid_explicit_proxy"
    bundle.actions = pd.DataFrame([dict(date=ex, symbol="sh.600000", split_ratio=2.0, cash_per_share=0.0, pay_date=ex,
                                        taxable_stock_per_share=1.0, share_listing_date=listing)])
    validate(bundle)
    pred.loc[pred.date >= bundle.calendar[3], "score"] = np.where(
        pred.loc[pred.date >= bundle.calendar[3], "symbol"].eq("sz.000001"), 1.0, 0.0)
    cfg = replace(cfg, rebalance=3, dividend_tax=True)
    out = simulate(panel, pred, bundle, cfg, "topk", slippage_bps=0)
    sells = out["orders"].query("symbol == 'sh.600000' and side == 'sell'")
    # The ex-date sale reaches only the 900 listed shares; the other 900 go at the next decision.
    assert sells[["date", "filled_qty"]].values.tolist() == [[ex, 900.0], [bundle.calendar[7], 900.0]]
    e = out["events"]
    assert e[e.kind.eq("split_shares_listed")][["date", "value"]].values.tolist() == [[listing, 900.0]]
    held = out["holdings"].set_index(["date", "symbol"])
    assert held.loc[(ex, "sh.600000"), ["quantity", "locked"]].tolist() == [900.0, 900.0]
    # Face value 1 x 900 bonus shares, held under a month: 20%, half with each sale.
    assert sells.dividend_tax.tolist() == pytest.approx([90.0, 90.0])
    bundle.actions["share_listing_date"] = bundle.calendar[3]
    with pytest.raises(ValueError, match="cannot list before the ex-date"):
        validate(bundle)


def test_hybrid_books_explicit_events_and_proxies_other_jumps():
    bundle, panel, cfg, pred = dividend_case(False)
    cfg = replace(cfg, topk=2, max_weight=0.45, rebalance=20)
    jump = bundle.calendar[3]
    for frame in (bundle.bars, panel):
        frame.loc[frame.symbol.eq("sz.000001") & frame.date.ge(jump), ["open", "high", "low", "close", "preclose"]] = 18.0
    validate(bundle)  # the unexplained sz.000001 jump is allowed in hybrid mode
    out = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    kinds = out["events"].set_index("symbol").kind.to_dict()
    assert kinds == {"sh.600000": "explicit_action", "sz.000001": "reference_reinvestment_proxy", "": "receivable_paid"}
    assert np.allclose(out["daily"].nav, cfg.initial_cash)
    bundle.metadata["corporate_actions"] = "explicit_events"
    with pytest.raises(ValueError, match="Missing explicit"):
        validate(bundle)


def test_vwap30_fills_and_square_root_impact():
    bundle, panel, cfg, pred = tiny()
    intraday = pd.DataFrame({"date": bundle.calendar[1:3], "symbol": "sh.600000",
                             "first_amount": [10_100.0, 0.0], "first_volume": [1_000.0, 0.0]})
    views = execution_views(panel, cfg.dev_start, cfg.dev_end, intraday)
    cfg = replace(cfg, fill_price="vwap30", impact_coefficient=1.0, initial_cash=1_000_000)
    out = simulate(panel, pred, bundle, cfg, "topk", slippage_bps=0, views=views)
    first = out["orders"].iloc[0]
    assert first.fill_price == pytest.approx(10.1) and first.filled_qty == 90_000
    expected = 0.02 * np.sqrt(first.notional / 1e8) * first.notional
    assert first.impact == pytest.approx(expected)
    assert out["daily"].iloc[1].impact == pytest.approx(expected)
    missing = simulate(panel, pred, bundle, replace(cfg, rebalance=1), "topk", slippage_bps=0,
                       views=execution_views(panel, cfg.dev_start, cfg.dev_end, intraday.iloc[1:]))
    assert missing["orders"].iloc[0].reason == "no_vwap30"


def test_missing_signal_holds_and_the_last_close_decides_pending_orders():
    bundle, panel, cfg, pred = tiny()
    cfg = replace(cfg, rebalance=1)
    days = bundle.calendar
    flip = pred.date.map({d: i % 2 for i, d in enumerate(days)})
    pred["score"] = np.where(pred.symbol.eq("sh.600000"), flip, 1 - flip).astype(float)
    full = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True)
    held = simulate(panel, pred[pred.date != days[4]], bundle, cfg, "topk", zero_cost=True)
    assert held["allocations"].set_index("date").loc[days[4], "status"] == "no_signal"
    assert full["orders"].decision_date.eq(days[4]).any() and not held["orders"].decision_date.eq(days[4]).any()
    k = 5  # forward replay through day k: same history, and day k's close decides tomorrow's orders
    short = simulate(panel, pred, bundle, cfg, "topk", zero_cost=True, period=(days[0], days[k]),
                     next_session=days[k + 1])
    pd.testing.assert_frame_equal(short["daily"], full["daily"].iloc[:k + 1])
    executed = full["orders"][full["orders"].decision_date == days[k]].set_index("symbol")
    pending = short["pending"].set_index("symbol")
    assert len(executed) == 2
    for symbol, order in executed.iterrows():
        sign = 1 if order.side == "buy" else -1
        assert pending.loc[symbol, "target"] - pending.loc[symbol, "held"] == pytest.approx(sign * order.requested_qty)
