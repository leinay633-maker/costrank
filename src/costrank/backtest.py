"""Daily raw-price ledger: close decisions, next-open orders, cash and positions.

The free-data mode is explicitly a reference-price reinvestment approximation;
it is not an exact shareholder cash-dividend/tax ledger. See DATA.md.
"""
from decimal import Decimal, ROUND_HALF_UP
import math
import numpy as np
import pandas as pd
from .data import board
from .portfolio import allocate, fees

CHINEXT_20PCT = pd.Timestamp("2020-08-24")
MAIN_ST_10PCT = pd.Timestamp("2026-07-06")  # 2026 exchange rule revisions (reference/rules_2026_review.md)
TICK = Decimal("0.01")


def limit_rate(symbol, day, is_st):
    """Daily price-limit rate (DATA.md): main board 10%, risk warning 5% (10% from 2026-07-06);
    ChiNext the same until 2020-08-23 and 20% for all its stocks from 2020-08-24; STAR 20%
    including risk warning."""
    kind, day = board(symbol), pd.Timestamp(day)
    if kind == "star" or (kind == "chinext" and day >= CHINEXT_20PCT):
        return Decimal("0.20")
    return Decimal("0.05") if is_st and day < MAIN_ST_10PCT else Decimal("0.10")


def price_band(reference, is_st, rate=None):
    rate = (Decimal("0.05") if is_st else Decimal("0.10")) if rate is None else rate
    ref = Decimal(str(reference))
    lower = (ref * (1-rate)).quantize(TICK, rounding=ROUND_HALF_UP)
    upper = (ref * (1+rate)).quantize(TICK, rounding=ROUND_HALF_UP)
    # A limit less than one tick from the reference moves to one tick away; never below one tick.
    return float(max(min(lower, ref - TICK), TICK)), float(max(upper, ref + TICK))


def lot_rules(symbol):
    """(minimum order, increment) in shares: STAR orders start at 200 and step by 1 share."""
    return (200, 1) if board(symbol) == "star" else (100, 100)


def round_buy(symbol, quantity):
    minimum, step = lot_rules(symbol)
    shares = math.floor((quantity + 1e-8) / step) * step
    return shares if shares >= minimum else 0


def round_sell(symbol, quantity, held):
    """Shares a partial sell may send; a full exit may sell an odd balance, including proxy
    fractional units, so it is returned unchanged."""
    if quantity + 1e-8 >= held:
        return quantity
    minimum, step = lot_rules(symbol)
    shares = math.floor((quantity + 1e-8) / step) * step
    if minimum > step:
        # A STAR balance below the minimum must leave in one order, so keep at least 200.
        shares = min(shares, math.floor(held + 1e-8) - minimum)
    return shares if shares >= minimum or (minimum == step and shares > 0) else 0


def opening_block(row, side, rate=None):
    if not row.tradable:
        return "suspended"
    if side == "buy" and row.is_st:
        return "st_at_execution"
    lower, upper = price_band(row.preclose, row.is_st, rate)
    if row.open < lower - 0.011 or row.open > upper + 0.011:
        return "unsupported_no_limit_session"
    if side == "buy" and row.open >= upper - 1e-8:
        return "limit_up"
    if side == "sell" and row.open <= lower + 1e-8:
        return "limit_down"
    return None


def dividend_tax_rate(acquired, sold):
    """Differentiated individual dividend tax (Caishui [2015] 101): held up to one calendar month
    20%, up to one year 10%, longer exempt; the holding runs to the day before the sale."""
    held_to = sold - pd.Timedelta(1, unit="D")
    if held_to < acquired + pd.DateOffset(months=1):
        return 0.20
    if held_to < acquired + pd.DateOffset(years=1):
        return 0.10
    return 0.0


def execution_views(panel, start, end, intraday=None):
    columns = ["date", "symbol", "open", "close", "preclose", "tradable", "is_st", "adv20", "vol20", "eligible"]
    data = panel[columns].copy()
    data["reinvestment_factor"] = (panel.groupby("symbol").close.shift() / panel.preclose).fillna(1)
    data = data[(data.date >= start) & (data.date <= end)]
    if intraday is not None:
        first = intraday[["date", "symbol"]].assign(
            vwap30=(intraday.first_amount / intraday.first_volume).where(intraday.first_volume > 0))
        data = data.merge(first, on=["date", "symbol"], how="left")
    return {day: f.set_index("symbol") for day, f in data.groupby("date", sort=False)}


def simulate(panel, predictions, bundle, config, policy, *, slippage_bps=5.0,
             zero_cost=False, holdout=False, views=None, planner=None, period=None, next_session=None):
    """`planner(day, score, weights, market, nav, locked, initial)` replaces portfolio.allocate for
    policies that need more than ranks and trailing volatility (the v0.4 optimizer).

    period=(start, end) overrides the configured development or holdout interval (forward paper
    trading). With next_session, the last day's close also decides, and its orders, not executed
    here, are returned as "pending"."""
    start, end = period if period is not None else (
        (config.holdout_start, config.holdout_end) if holdout else (config.dev_start, config.dev_end))
    days = bundle.calendar[(bundle.calendar >= start) & (bundle.calendar <= end)]
    if len(days) == 0:
        raise ValueError("No backtest sessions")
    if predictions.duplicated(["date", "symbol"]).any():
        raise ValueError("Pass exactly one model's predictions")
    by_day = execution_views(panel, start, end) if views is None else views
    scores = {day: f.set_index("symbol").score for day, f in predictions.groupby("date")}
    action_days = {day: f for day, f in bundle.actions.groupby("date")}
    mode = bundle.metadata["corporate_actions"]
    # Tax lots [acquired, quantity, taxable dividend income] exist only where dividends are explicit.
    taxed = config.dividend_tax and mode != "reference_reinvestment_proxy"
    lots = {}
    instruments = bundle.instruments.set_index("symbol")
    resolutions = bundle.delistings.set_index("symbol")
    cash, previous_nav = config.initial_cash, config.initial_cash
    positions, last_marks, pending = {}, {}, None
    receivables = []
    # Acquirer shares from a share exchange sit in positions like any holding (marked, weighted,
    # adjusted for corporate actions) but cannot be sold before their settlement date.
    locks = []
    daily, orders, holdings, events, allocations = [], [], [], [], []
    decisions = 0
    for i, day in enumerate(days):
        market = by_day.get(day, pd.DataFrame())
        # Corporate action entitlements belong to positions held before today's trades.
        explicit_today = action_days.get(day, pd.DataFrame(columns=["symbol"]))
        if mode in {"reference_reinvestment_proxy", "hybrid_explicit_proxy"}:
            symbols = set(positions) | (set(pending["targets"]) if pending else set())
            if mode == "hybrid_explicit_proxy":
                symbols -= set(explicit_today.symbol)  # booked explicitly below
            for symbol in sorted(symbols):
                factor = float(market.loc[symbol, "reinvestment_factor"]) if symbol in market.index else 1.0
                if abs(factor - 1) > 1e-7:
                    if symbol in positions:
                        positions[symbol] *= factor
                    if pending and symbol in pending["targets"]:
                        pending["targets"][symbol] *= factor
                    for lock in locks:
                        if lock["symbol"] == symbol:
                            lock["quantity"] *= factor
                    for lot in lots.get(symbol, []):
                        lot[1] *= factor
                    events.append(dict(date=day, symbol=symbol, kind="reference_reinvestment_proxy", value=factor))
        if mode in {"explicit_events", "hybrid_explicit_proxy"}:
            for event in explicit_today.itertuples(index=False):
                qty = positions.get(event.symbol, 0.0)
                if qty:
                    receivables.append((event.pay_date, qty * event.cash_per_share))
                    positions[event.symbol] = qty * event.split_ratio
                    events.append(dict(date=day, symbol=event.symbol, kind="explicit_action", value=qty*event.cash_per_share))
                # Bonus shares from retained earnings are taxed at their face value of one yuan.
                bonus = float(getattr(event, "taxable_stock_per_share", 0.0))
                for lot in lots.get(event.symbol, []):
                    lot[2] += lot[1] * (event.cash_per_share + bonus)
                    lot[1] *= event.split_ratio
                for lock in locks:
                    if lock["symbol"] == event.symbol:
                        lock["quantity"] *= event.split_ratio
                # The shares a split adds can be sold only from their listing date (Shanghai: the
                # session after the ex-date); until then they are held, marked and taxed, not sold.
                listing = getattr(event, "share_listing_date", pd.NaT)
                if qty and event.split_ratio > 1 and pd.notna(listing) and listing > day:
                    locks.append({"symbol": event.symbol, "quantity": qty * (event.split_ratio - 1),
                                  "settle": listing, "released": "split_shares_listed"})
                if pending and event.symbol in pending["targets"]:
                    pending["targets"][event.symbol] *= event.split_ratio
        for symbol in list(positions):
            removal = instruments.loc[symbol, "delisted"]
            if pd.notna(removal) and day >= removal:
                quantity = positions.pop(symbol)
                lots.pop(symbol, None)
                value = quantity * last_marks.get(symbol, 0)
                if symbol not in resolutions.index:
                    # Not an invented last-close sale: explicit conservative zero recovery.
                    events.append(dict(date=day, symbol=symbol, kind="delisting_zero_recovery_assumption", value=value))
                elif resolutions.loc[symbol, "kind"] == "share_exchange":
                    r = resolutions.loc[symbol]
                    received = quantity * float(r.ratio)
                    # The announced ratio is final, so shares received today miss the acquirer's
                    # same-day ex-date; until it trades they carry the old position's value.
                    positions[r.acquirer] = positions.get(r.acquirer, 0) + received
                    if taxed:
                        lots.setdefault(r.acquirer, []).append([day, received, 0.0])
                    last_marks.setdefault(r.acquirer, value / received)
                    locks.append({"symbol": r.acquirer, "quantity": received, "settle": r.settle_date})
                    events.append(dict(date=day, symbol=symbol, kind="delisting_share_exchange", value=value))
                else:
                    r = resolutions.loc[symbol]
                    receivables.append((r.settle_date, quantity * float(r.cash_per_share)))
                    events.append(dict(date=day, symbol=symbol, kind="delisting_cash", value=quantity * float(r.cash_per_share)))
                if pending:
                    pending["targets"].pop(symbol, None)
        due = sum(amount for when, amount in receivables if when <= day)
        if due:
            events.append(dict(date=day, symbol="", kind="receivable_paid", value=due))
        cash += due
        receivables = [(when, amount) for when, amount in receivables if when > day]
        receivable_value = sum(amount for _, amount in receivables)
        for lock in [x for x in locks if day >= x["settle"]]:
            locks.remove(lock)
            events.append(dict(date=day, symbol=lock["symbol"], kind=lock.get("released", "lock_released"),
                               value=lock["quantity"]))
        locked = {}
        for lock in locks:
            locked[lock["symbol"]] = locked.get(lock["symbol"], 0) + lock["quantity"]
        open_marks = {s: float(market.loc[s, "open"]) if s in market.index else last_marks[s]
                      for s in positions}
        open_nav = cash + receivable_value + sum(positions[s]*open_marks[s] for s in positions)
        if open_nav <= 0:
            raise ValueError("Nonpositive equity")
        paid = {"commission": 0.0, "stamp": 0.0, "transfer": 0.0, "slippage": 0.0}
        if taxed:
            paid["dividend_tax"] = 0.0
        if config.impact_coefficient:
            paid["impact"] = 0.0
        traded, rejected, requested = 0.0, 0, 0.0
        target_turnover, turnover_limit = np.nan, np.nan
        if pending:
            target_turnover = pending["target_turnover"]
            if policy in {"turnover", "cost"} and not pending["initial"]:
                turnover_limit = config.turnover_cap + pending["compliance"]
            cap = turnover_limit * open_nav if np.isfinite(turnover_limit) else float("inf")
            trades = []
            for symbol in sorted(set(positions) | set(pending["targets"])):
                delta = pending["targets"].get(symbol, 0) - positions.get(symbol, 0)
                if abs(delta) > 1e-8:
                    trades.append((delta > 0, symbol not in pending["trim_first"], symbol, delta))
            # Sells first, cap-compliance trims ahead of other sells; deterministic symbol priority.
            for buy, _, symbol, delta in sorted(trades):
                side = "buy" if buy else "sell"
                wanted = abs(delta) if buy else min(abs(delta), positions.get(symbol, 0) - locked.get(symbol, 0))
                if wanted <= 1e-8:
                    continue  # only locked shares (unsettled exchange, unlisted split) stand above the target
                record = dict(date=day, decision_date=pending["date"], symbol=symbol,
                              side=side, requested_qty=wanted, filled_qty=0.0,
                              notional=0.0, commission=0.0, stamp=0.0, transfer=0.0,
                              slippage=0.0, status="rejected", reason="")
                if symbol not in market.index:
                    record["reason"] = "not_listed_or_missing"
                else:
                    row = market.loc[symbol]
                    price = float(row.open) if config.fill_price == "open" else float(row.get("vwap30", np.nan))
                    requested += wanted * row.open
                    record["reason"] = opening_block(row, side, limit_rate(symbol, day, row.is_st)) or ""
                    if not record["reason"] and not np.isfinite(price):
                        record["reason"] = "no_vwap30"
                    if not record["reason"]:
                        adv = pending["adv"].get(symbol, 0)  # participation room
                        budget = min(max(0, cap-traded), config.participation * max(0, adv))
                        quantity = min(wanted, budget / price)
                        if buy:
                            quantity = round_buy(symbol, min(quantity, cash / price))
                        else:
                            quantity = round_sell(symbol, quantity, positions.get(symbol, 0))
                        sigma = pending["vol"].get(symbol, np.nan) if config.impact_coefficient else np.nan
                        while quantity > 0:
                            notional = quantity * price
                            breakdown = fees(notional, side, day, config, zero=zero_cost)
                            slip = 0.0 if zero_cost else notional * slippage_bps / 10_000
                            impact = 0.0
                            depth = pending["adv_impact"].get(symbol, 0) if config.impact_coefficient else 0
                            if config.impact_coefficient and not zero_cost and depth > 0 and np.isfinite(sigma):
                                # Square-root law on the decision day's full-session ADV20 and volatility.
                                impact = config.impact_coefficient * sigma * math.sqrt(notional / depth) * notional
                            total_cost = sum(breakdown.values()) + slip + impact
                            if not buy or notional + total_cost <= cash + 1e-8:
                                break
                            quantity = round_buy(symbol, quantity - lot_rules(symbol)[1])
                        if quantity <= 0:
                            record["reason"] = "cash_capacity_turnover_or_lot"
                        else:
                            tax = 0.0
                            if buy:
                                cash -= notional + total_cost
                                positions[symbol] = positions.get(symbol, 0) + quantity
                                if taxed:
                                    lots.setdefault(symbol, []).append([day, quantity, 0.0])
                            else:
                                if taxed:
                                    left = quantity
                                    for lot in lots.get(symbol, []):
                                        take = min(lot[1], left)
                                        if take <= 0:
                                            break
                                        share = lot[2] * take / lot[1]
                                        tax += share * dividend_tax_rate(lot[0], day)
                                        lot[1] -= take
                                        lot[2] -= share
                                        left -= take
                                    lots[symbol] = [lot for lot in lots.get(symbol, []) if lot[1] > 1e-8]
                                    tax = 0.0 if zero_cost else tax
                                cash += notional - total_cost - tax
                                positions[symbol] -= quantity
                                if positions[symbol] < 1e-8:
                                    del positions[symbol]
                                    lots.pop(symbol, None)
                            if cash < -1e-6:
                                raise AssertionError("Negative cash")
                            for key, value in breakdown.items():
                                paid[key] += value
                                record[key] = value
                            paid["slippage"] += slip
                            if taxed:
                                paid["dividend_tax"] += tax
                                record["dividend_tax"] = tax
                            if config.impact_coefficient:
                                paid["impact"] += impact
                                record["impact"] = impact
                            traded += notional
                            record.update(filled_qty=quantity, notional=notional, slippage=slip,
                                open_price=float(row.open), status="filled" if quantity >= wanted-1e-6 else "partial",
                                reason="" if quantity >= wanted-1e-6 else "cash_capacity_turnover_or_lot")
                            if config.fill_price != "open":
                                record["fill_price"] = price
                rejected += record["status"] != "filled"
                orders.append(record)
            if traded > cap + 1e-5:
                raise AssertionError("Realized turnover cap breached")
            pending = None  # Unfilled orders expire, not silently carried forward.
        for symbol in positions:
            if symbol in market.index:
                last_marks[symbol] = float(market.loc[symbol, "close"])
            elif symbol not in locked:  # exchange shares of a not-yet-listed acquirer keep their carried value
                raise ValueError(f"Cannot mark held position {symbol} on {day}")
        value = sum(positions[s] * last_marks[s] for s in positions)
        nav = cash + receivable_value + value
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("Invalid NAV")
        weights = pd.Series({s: positions[s] * last_marks[s] / nav for s in positions}, dtype=float)
        estimated_risk = sum(weights[s] * (max(.001, float(market.loc[s, "vol20"]))
                             if s in market.index and pd.notna(market.loc[s, "vol20"]) else 2*config.risk_budget)
                             for s in weights.index)
        extra = {}
        if taxed:
            # Tax that would be due if every lot were sold at today's close; not deducted from NAV.
            extra["dividend_tax_accrued"] = sum(lot[2] * dividend_tax_rate(lot[0], day)
                                                for held in lots.values() for lot in held)
        daily.append(dict(date=day, nav=nav, cash=cash, receivable=receivable_value,
            net_return=nav/previous_nav-1, gross_same_trades_return=(nav+sum(paid.values()))/previous_nav-1,
            traded_notional=traded, turnover_two_way=traded/open_nav, exposure=value/nav,
            holdings=len(positions), max_weight=float(weights.max()) if len(weights) else 0,
            estimated_vol_bound=estimated_risk, rejected_or_partial=rejected,
            requested_notional=requested, target_turnover=target_turnover, **paid,
            locked_value=sum(q * last_marks[s] for s, q in locked.items()), turnover_limit=turnover_limit, **extra))
        for symbol, qty in positions.items():
            holdings.append(dict(date=day, symbol=symbol, quantity=qty, close=last_marks[symbol], weight=weights[symbol],
                                 locked=locked.get(symbol, 0.0)))
        previous_nav = nav
        # No order uses tomorrow's price/volume/flags to choose a target quantity.
        last = len(daily) == len(days)
        if (not last or next_session is not None) and (len(daily)-1) % config.rebalance == 0:
            following = pd.Timestamp(next_session) if last else days[i + 1]
            score = scores.get(day, pd.Series(dtype=float))
            eligible = market.index[market.eligible] if len(market) else []
            score = score[score.index.isin(eligible)]
            if score.empty:
                # No signal (e.g. a failed forward-period data gate): keep the book, never liquidate it.
                allocations.append(dict(date=day, status="no_signal", target_turnover=0.0, compliance_trim=0.0))
                continue
            # Shares still locked at the next open (exchange or split shares) are a floor the optimizer must keep.
            floor_qty = {}
            for lock in locks:
                if lock["settle"] > following:
                    floor_qty[lock["symbol"]] = floor_qty.get(lock["symbol"], 0) + lock["quantity"]
            floor_weight = pd.Series({s: q * last_marks[s] / nav for s, q in floor_qty.items()}, dtype=float)
            if planner is None:
                target, audit = allocate(score, weights, market.vol20, config, policy, day, initial=decisions == 0,
                                         locked=floor_weight)
            else:
                target, audit = planner(day, score, weights, market, nav, floor_weight, decisions == 0)
            targets = {s: round_buy(s, w * nav / float(market.loc[s, "close"]))
                       for s, w in target.items() if s in market.index and w > 0}
            for s, q in floor_qty.items():
                targets[s] = max(targets.get(s, 0), q)
            if audit.get("status") == "hold_fallback":
                targets = dict(positions)  # No rounding-induced liquidation after solver failure.
            over = (set(weights.index[weights > config.max_weight])
                    if config.weight_cap_mode == "enforce" else set())
            room = market["adv_cap"] if "adv_cap" in market else market.adv20
            sigma = getattr(planner, "sigma", None) if planner is not None else None
            pending = {"date": day, "targets": targets, "adv": room.fillna(0).to_dict(),
                       "adv_impact": market.adv20.fillna(0).to_dict() if config.impact_coefficient else {},
                       "vol": (sigma.to_dict() if sigma is not None else market.vol20.to_dict())
                              if config.impact_coefficient else {},
                       "initial": decisions == 0, "target_turnover": audit["target_turnover"],
                       "compliance": audit.get("compliance_trim", 0.0), "trim_first": over}
            allocations.append(dict(date=day, **audit))
            decisions += 1
    out = {"daily": pd.DataFrame(daily), "orders": pd.DataFrame(orders),
           "holdings": pd.DataFrame(holdings), "events": pd.DataFrame(events),
           "allocations": pd.DataFrame(allocations)}
    if next_session is not None:
        # As the ledger reads them: a name missing from a decision's targets is sold; no decision holds.
        targets = pending["targets"] if pending is not None and pending["date"] == days[-1] else dict(positions)
        names = sorted(set(targets) | set(positions))
        out["pending"] = pd.DataFrame({"decided": days[-1], "execute": pd.Timestamp(next_session), "symbol": names,
                                       "held": [positions.get(s, 0) for s in names],
                                       "target": [targets.get(s, 0) for s in names]})
    return out
