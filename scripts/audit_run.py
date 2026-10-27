"""Offline re-derivation of a completed run's ledgers from its own CSVs and the sealed
snapshot it recorded: cash/NAV identities, fees by effective date, lots, T+1 timing,
opening blocks re-checked on raw bars, the realized turnover cap and delisting marks.
Fee constants, board price limits and lot rules are typed from docs/DATA.md, not imported
from the ledger code. v0.4 runs add dividend tax, impact, VWAP fills and optimizer bounds.

usage: python scripts/audit_run.py <run_dir> <out.json>
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from costrank.config import Config
from costrank.data import read_bundle, sha256

FEES = ["commission", "stamp", "transfer", "slippage"]
ZERO_COUNTS = ["odd_lot_buys", "odd_lot_sells_not_full_exit", "orders_not_next_session",
               "duplicate_symbol_orders_same_day", "fills_on_suspended", "fills_open_price_mismatch",
               "buys_at_or_above_limit_up", "sells_at_or_below_limit_down", "buys_while_st",
               "fills_outside_price_band", "turnover_cap_breaches", "holdings_on_or_after_delisting",
               "share_exchanges_not_settled", "locked_shares_sold", "star_lot_violations",
               "vwap_price_mismatch", "negative_tax_or_impact", "dividend_tax_above_bound",
               "optimizer_bound_breaches", "unlisted_split_shares_sold"]
# Excess of the returned weights over the bounds the optimizer imposed, in reported units (annual
# TE, style exposure, industry weight). 1e-4 is economically negligible in each and far above the
# solver accuracy seen so far (RESEARCH section 9 supplement 7 accepts AlmostSolved).
OPTIMIZER_TOLERANCE = 1e-4
TOLERANCES = {"cash_identity_max_abs_cny": 0.01, "nav_identity_max_abs_cny": 0.01,
              "net_return_max_abs_diff": 1e-12, "holding_close_vs_bars_max_abs": 0.0,
              "traded_notional_max_abs_cny": 0.01, "fee_columns_max_abs_cny": 0.01,
              "fee_schedule_max_abs_cny": 1e-6, "summary_total_return_abs_diff": 1e-9}


def frame(path):
    try:
        f = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    for col in ("date", "decision_date"):
        if col in f:
            f[col] = pd.to_datetime(f[col])
    return f


def band(reference, st, symbol="sh.600000", day="2020-01-01"):
    """DATA.md: main board 10% (risk warning 5%, 10% from 2026-07-06); ChiNext 20% from
    2020-08-24; STAR 20%."""
    twenty = symbol.startswith("sh.688") or (symbol.startswith("sz.30") and str(day)[:10] >= "2020-08-24")
    st = st and str(day)[:10] < "2026-07-06"
    rate = Decimal("0.20") if twenty else Decimal("0.05") if st else Decimal("0.10")
    ref, tick = Decimal(str(reference)), Decimal("0.01")
    lo = (ref * (1 - rate)).quantize(tick, rounding=ROUND_HALF_UP)
    hi = (ref * (1 + rate)).quantize(tick, rounding=ROUND_HALF_UP)
    return float(max(min(lo, ref - tick), tick)), float(max(hi, ref + tick))


def finite(value):
    return float(value) if np.isfinite(value) else None


def kind(events, name):
    return events[events.kind.eq(name)] if len(events) else pd.DataFrame(columns=["date", "symbol", "kind", "value"])


def audit(path, config, policy, bps, zero, bars, following, delisted, summary_row, resolutions, intraday=None,
          actions=None):
    daily, orders = frame(path / "daily.csv.gz"), frame(path / "orders.csv.gz")
    extra_costs = [c for c in ("impact", "dividend_tax") if c in orders]
    holdings, events = frame(path / "holdings.csv.gz"), frame(path / "events.csv.gz")
    allocations = frame(path / "allocations.csv.gz")
    days = daily.date
    fills = orders[orders.status.ne("rejected")]
    r = {}
    flow = (np.where(fills.side.eq("sell"), 1.0, -1.0) * fills.notional - fills[FEES + extra_costs].fillna(0).sum(axis=1))
    flow = flow.groupby(fills.date).sum().reindex(days, fill_value=0.0)
    # Cash also moves when a receivable (here: a cash delisting outcome) is paid.
    paid = kind(events, "receivable_paid")
    flow += paid.value.groupby(paid.date).sum().reindex(days, fill_value=0.0).to_numpy() if len(paid) else 0.0
    cash = config.initial_cash + flow.cumsum().to_numpy()
    r["cash_identity_max_abs_cny"] = float(np.abs(daily.cash.to_numpy() - cash).max())
    value = (holdings.quantity * holdings.close).groupby(holdings.date).sum().reindex(days, fill_value=0.0)
    r["nav_identity_max_abs_cny"] = float(np.abs(daily.nav - daily.cash - daily.receivable - value.to_numpy()).max())
    previous = np.r_[config.initial_cash, daily.nav.to_numpy()[:-1]]
    r["net_return_max_abs_diff"] = float(np.abs(daily.net_return.to_numpy() - (daily.nav.to_numpy() / previous - 1)).max())
    marks = holdings.join(bars.close.rename("bar_close"), on=["date", "symbol"])
    r["holding_close_vs_bars_max_abs"] = float((marks.close - marks.bar_close).abs().max()) if len(marks) else 0.0
    by_day = fills.groupby("date")[["notional", *FEES]].sum().reindex(days, fill_value=0.0)
    r["traded_notional_max_abs_cny"] = float(np.abs(daily.traded_notional.to_numpy() - by_day.notional.to_numpy()).max())
    r["fee_columns_max_abs_cny"] = float(max(np.abs(daily[c].to_numpy() - by_day[c].to_numpy()).max() for c in FEES))

    d, n, sell = fills.date, fills.notional, fills.side.eq("sell")
    if zero:
        expected = {c: 0.0 * n for c in FEES}
    else:
        expected = {"commission": np.maximum(config.minimum_commission, config.commission * n),
                    "stamp": np.where(sell, n * np.where(d < "2023-08-28", 0.001, 0.0005), 0.0),
                    "transfer": n * np.where(d < "2022-04-29", 0.00002, 0.00001),
                    "slippage": n * bps / 10_000}
    r["fee_schedule_max_abs_cny"] = float(max((fills[c] - expected[c]).abs().max() for c in FEES)) if len(fills) else 0.0
    r["fills_by_fee_regime"] = {"before_2022-04-29": int((d < "2022-04-29").sum()),
        "from_2022-04-29": int((d >= "2022-04-29").sum()),
        "sells_before_2023-08-28": int((sell & (d < "2023-08-28")).sum()),
        "sells_from_2023-08-28": int((sell & (d >= "2023-08-28")).sum())}

    star = fills.symbol.str.startswith("sh.688")
    lot = fills.filled_qty % 100
    odd = fills[(np.minimum(lot, 100 - lot) > 1e-6) & ~star]
    held = set(zip(holdings.date, holdings.symbol))
    odd_sells = odd[odd.side.eq("sell")]
    r["odd_lot_buys"] = int(odd.side.eq("buy").sum())
    # Odd sells are legal only as full exits, including proxy fractional units.
    r["odd_lot_sells_not_full_exit"] = int(sum(k in held for k in zip(odd_sells.date, odd_sells.symbol)))
    r["odd_lot_full_exits"] = len(odd_sells) - r["odd_lot_sells_not_full_exit"]
    # STAR: buys of at least 200 whole shares; a partial sell keeps at least 200.
    sb = fills[star]
    after = holdings.set_index(["date", "symbol"]).quantity if len(holdings) else pd.Series(dtype=float)
    left = np.array([after.get((d, s), 0.0) for d, s in zip(sb.date, sb.symbol)]) if len(sb) else np.array([])
    whole = (sb.filled_qty - np.floor(sb.filled_qty + 1e-9)).abs() < 1e-6
    bad_buy = sb.side.eq("buy") & ((sb.filled_qty < 200 - 1e-9) | ~whole)
    bad_sell = sb.side.eq("sell") & (left > 1e-8) & ((left < 200 - 1e-6) | (sb.filled_qty < 200 - 1e-9))
    r["star_lot_violations"] = int(bad_buy.sum() + bad_sell.sum())
    r["orders_not_next_session"] = int((orders.date != orders.decision_date.map(following)).sum())
    r["duplicate_symbol_orders_same_day"] = int(orders.duplicated(["date", "symbol"]).sum())

    ex = fills.join(bars[["open", "preclose", "tradable", "is_st"]], on=["date", "symbol"])
    lower, upper = (np.array(x) for x in zip(*[band(p, s, sym, d) for p, s, sym, d in
                                               zip(ex.preclose, ex.is_st, ex.symbol, ex.date)])) if len(ex) else (np.array([]),) * 2
    buy, price = ex.side.eq("buy").to_numpy(), ex.open.to_numpy()
    r["fills_on_suspended"] = int(ex.tradable.ne(1).sum())
    r["fills_open_price_mismatch"] = int((ex.open_price - ex.open).abs().gt(1e-9).sum())
    r["buys_at_or_above_limit_up"] = int((buy & (price >= upper - 1e-8)).sum())
    r["sells_at_or_below_limit_down"] = int((~buy & (price <= lower + 1e-8)).sum())
    r["buys_while_st"] = int((buy & ex.is_st.eq(1).to_numpy()).sum())
    r["fills_outside_price_band"] = int(((price < lower - 0.011) | (price > upper + 0.011)).sum())
    if "fill_price" in fills and intraday is not None and len(fills):
        v = fills.join(intraday, on=["date", "symbol"])
        r["vwap_price_mismatch"] = int(((v.fill_price - v.first_amount / v.first_volume).abs() > 1e-9).sum())
    else:
        r["vwap_price_mismatch"] = 0
    r["negative_tax_or_impact"] = int(sum((fills[c] < -1e-12).sum() for c in extra_costs))
    credited = kind(events, "explicit_action")
    held = holdings.set_index(["date", "symbol"]).quantity if len(holdings) else pd.Series(dtype=float)
    before = dict(zip(days.iloc[1:], days.iloc[:-1]))  # each session's previous session in the run
    # Entitlements belong to the shares held at the previous close, before the ex-date's trades.
    entitled = {(e.date, e.symbol): held.get((before.get(e.date), e.symbol), 0.0) for e in credited.itertuples()}
    r["unlisted_split_shares_sold"] = 0
    if len(credited) and actions is not None and "share_listing_date" in actions:
        for (when, symbol), quantity in entitled.items():
            act = actions.loc[(when, symbol)]
            if act.split_ratio > 1 and act.share_listing_date > when:
                added = quantity * (act.split_ratio - 1)
                window = days[(days >= when) & (days < act.share_listing_date)]
                r["unlisted_split_shares_sold"] += int(any(held.get((d, symbol), 0.0) < added - 1e-6 for d in window))
    if "dividend_tax" in fills:
        cash_div = float(credited.value.sum()) if len(credited) else 0.0
        bonus = 0.0
        if len(credited) and actions is not None:
            for (when, symbol), quantity in entitled.items():
                bonus += quantity * actions.loc[(when, symbol)].taxable_stock_per_share
        # Tax never exceeds 20% of the cash dividends credited plus bonus shares at face value.
        r["dividend_tax_total_cny"] = float(fills.dividend_tax.sum())
        r["dividend_tax_above_bound"] = int(fills.dividend_tax.sum() > 0.2 * (cash_div + bonus) + 1e-6)
    else:
        r["dividend_tax_above_bound"] = 0
    if zero and extra_costs:
        r["negative_tax_or_impact"] += int(sum((fills[c].abs() > 1e-12).sum() for c in extra_costs))

    later = daily[daily.date != orders.date.min()]  # the initial build is exempt from the cap
    r["max_turnover_after_initial"] = float(later.turnover_two_way.max())
    capped = policy in {"turnover", "cost"}
    # Runs that enforce the weight cap allow the forced trim on top of turnover_cap; the
    # ledger records that day's limit, so check against it rather than the bare cap.
    limit = later.turnover_limit.fillna(config.turnover_cap) if "turnover_limit" in later else config.turnover_cap
    r["turnover_cap_breaches"] = int((later.turnover_two_way > limit + 1e-9).sum()) if capped else 0
    r["days_at_turnover_cap"] = int((later.turnover_two_way > limit - 1e-4).sum()) if capped else None
    extra = (later.turnover_limit - config.turnover_cap).clip(lower=0) if "turnover_limit" in later else pd.Series(dtype=float)
    r["days_with_compliance_allowance"] = int(extra.gt(1e-12).sum())
    r["max_compliance_allowance"] = float(extra.max()) if extra.notna().any() else 0.0
    r["order_status"] = orders.status.value_counts().to_dict()
    r["reject_or_partial_reasons"] = orders.loc[orders.status.ne("filled"), "reason"].value_counts().to_dict()
    r["allocation_status"] = allocations.status.value_counts().to_dict() if len(allocations) else {}
    r["optimizer_bound_breaches"] = 0
    if len(allocations) and "ex_ante_te" in allocations:
        ok = allocations[allocations.status.eq("optimal")]
        if "style_excess" in ok:
            excess = ok[["te_excess", "style_excess", "industry_excess"]]
        else:  # runs before the planner recorded excesses: declared bands (a locked relaxation shows as excess)
            excess = pd.DataFrame({"te": ok.ex_ante_te - config.te_annual, "style": ok.max_style_active - config.style_band,
                                   "industry": ok.max_industry_active - config.industry_band})
        r["optimizer_max_excess"] = {c: float(excess[c].max()) if len(ok) else 0.0 for c in excess}
        r["optimizer_bound_breaches"] = int((excess > OPTIMIZER_TOLERANCE).to_numpy().sum())
        r["hold_fallback_share"] = float(allocations.status.eq("hold_fallback").mean())
        widened = (allocations.status.eq("optimal") & allocations.risk_limit_scale.fillna(1.0).gt(1.0)
                   if "risk_limit_scale" in allocations else pd.Series(False, index=allocations.index))
        # RESEARCH section 9 supplement 8: the share compared with the 5% limit counts widened days too.
        r["infeasible_share"] = float((allocations.status.eq("hold_fallback") | widened).mean())
        r["max_ex_ante_te"] = float(ok.ex_ante_te.max()) if len(ok) else None

    for name, label in (("delisting_zero_recovery_assumption", "delisting_zero_recovery"),
                        ("delisting_share_exchange", "delisting_share_exchange"), ("delisting_cash", "delisting_cash")):
        de = kind(events, name)
        r[label] = {"events": len(de), "value_cny": float(de.value.sum()) if len(de) else 0.0,
                    "symbols": sorted(de.symbol.unique().tolist()) if len(de) else []}
    # Exchange shares are released no earlier than settlement, and never sold while locked.
    swaps, released = kind(events, "delisting_share_exchange"), kind(events, "lock_released")
    late = 0
    for e in swaps.itertuples():
        res = resolutions.loc[e.symbol]
        if res.settle_date <= days.max():
            late += int(not ((released.symbol == res.acquirer) & (released.date >= res.settle_date)).any())
    early = 0
    for e in released.itertuples():
        settle = resolutions[resolutions.index.isin(swaps.symbol) & resolutions.acquirer.eq(e.symbol)].settle_date
        early += int(len(settle) > 0 and e.date < settle.min())
    r["share_exchanges_not_settled"] = late + early
    r["locked_shares_sold"] = int((holdings.quantity < holdings.locked - 1e-6).sum()) if "locked" in holdings else 0
    r["proxy_reinvestment_events"] = int(events.kind.eq("reference_reinvestment_proxy").sum()) if len(events) else 0
    gone = holdings.join(delisted.rename("removal"), on="symbol")
    r["holdings_on_or_after_delisting"] = int((gone.removal.notna() & (gone.date >= gone.removal)).sum())
    r["max_exposure"] = float(daily.exposure.max())
    r["days_exposure_above_invested"] = int((daily.exposure > config.invested + 1e-9).sum())
    r["max_single_weight"] = float(daily.max_weight.max())
    r["days_weight_above_cap"] = int((daily.max_weight > config.max_weight + 1e-9).sum())
    r["summary_total_return_abs_diff"] = abs(float(np.prod(1 + daily.net_return) - 1) - float(summary_row.total_return))
    r["failures"] = [k for k, tol in TOLERANCES.items() if not r[k] <= tol] + [k for k in ZERO_COUNTS if r[k]]
    return r


def overview(run, start, end):
    folds = json.loads((run / "folds.json").read_text(encoding="utf-8"))
    ic = pd.read_csv(run / "rank_ic.csv")
    coverage = pd.read_csv(run / "universe_coverage.csv", parse_dates=["date"])
    coverage = coverage[(coverage.date >= start) & (coverage.date <= end)]
    stats = {}
    for model, g in ic.groupby("model"):
        x = g.rank_ic.dropna()
        stats[model] = {"days": len(g), "days_with_ic": len(x), "mean": finite(x.mean()), "std": finite(x.std()),
                        "ir_daily": finite(x.mean() / x.std()), "share_positive": finite((x > 0).mean()),
                        "mean_observed_labels": finite(g.observed_labels.mean())}
    return {"folds": len(folds), "first_test_start": folds[0]["test_start"], "last_test_end": folds[-1]["test_end"],
            "train_rows": [f["train_rows"] for f in folds],
            "lightgbm_best_iteration": [f["models"].get("lightgbm", {}).get("best_iteration") for f in folds],
            "eligible_names_in_evaluation_range": {"mean": float(coverage.eligible_names.mean()),
                "min": int(coverage.eligible_names.min()), "max": int(coverage.eligible_names.max())},
            "rank_ic": stats}


def audit_run(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Run is not complete")
    config = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in manifest["config"].items()})
    # A supplementary run (RESEARCH section 11) reads a snapshot past the holdout, over its own period.
    holdout = manifest["phase"] in ("holdout", "supplement")
    bundle = read_bundle(manifest["data_path"], allow_after=None if holdout else config.holdout_start)
    if bundle.prefix_hash(config.holdout_start) != manifest["data_prefix_hash"]:
        raise ValueError("Snapshot differs from the one the run recorded")
    # Explicit dividends reach cash only through receivable_paid events, which the identity uses.
    intraday = bundle.tables["intraday"].set_index(["date", "symbol"])[["first_amount", "first_volume"]] \
        if "intraday" in bundle.tables else None
    changed = sorted(name for name, expected in manifest["result_files"].items() if sha256(run / name) != expected)
    bars = bundle.bars.set_index(["date", "symbol"])
    sessions = list(bundle.calendar)
    following = dict(zip(sessions[:-1], sessions[1:]))
    delisted = bundle.instruments.set_index("symbol").delisted
    resolutions = bundle.delistings.set_index("symbol")
    summary = pd.read_csv(run / "summary.csv")
    scenarios = [("zero_all_costs", 0.0, True)] + [(f"slip_{x:g}bps", float(x), False) for x in config.slippage_grid_bps]
    results = {}
    for model in config.models:
        for policy in config.policies:
            for scenario, bps, zero in scenarios:
                name = f"{model}__{policy}__{scenario}"
                row = summary[(summary.model == model) & (summary.policy == policy) & (summary.scenario == scenario)].iloc[0]
                results[name] = audit(run / name, config, policy, bps, zero, bars, following, delisted, row, resolutions,
                                      intraday, bundle.actions.set_index(["date", "symbol"]) if len(bundle.actions) else None)
    failed = sorted(k for k, r in results.items() if r["failures"])
    start, end = ((manifest["period"]["start"], manifest["period"]["end"]) if manifest["phase"] == "supplement" else
                  (config.holdout_start, config.holdout_end) if holdout else (config.dev_start, config.dev_end))
    return {"run": str(run.resolve()), "passed": not failed and not changed, "git": manifest.get("git"),
            "code_hash": manifest["code_hash"], "data_prefix_hash": manifest["data_prefix_hash"],
            "prediction_hash": manifest.get("prediction_hash"), "wall_seconds": manifest.get("wall_seconds"),
            "result_files_changed": changed, "scenarios_audited": len(results), "scenarios_failed": failed,
            "worst_case": {k: max(r[k] for r in results.values()) for k in TOLERANCES},
            "violation_counts": {k: sum(r[k] for r in results.values()) for k in ZERO_COUNTS},
            "overview": overview(run, start, end), "scenarios": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("output")
    args = parser.parse_args()
    report = audit_run(args.run)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "scenarios"}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
