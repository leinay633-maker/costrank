"""Forward paper trading operations (docs/PAPER.md): run once per trading day after the close.

The protocol pieces (which sessions may trade, on which scores) live in forward.py and are hashed at
the freeze; this module only fetches, records, compares, reports and alerts. The raw cache only
grows. A session is recorded only after its data gate passes (or a person accepts a failed gate):
its scores, every ledger table and the next session's orders are each written once and never
rewritten. Every day the ledger is replayed over the forward period with the stored scores by the
backtest's own code and compared with the whole record; a difference is a data revision, alerted,
never applied to the record.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import time
import urllib.request
import numpy as np
import pandas as pd
from .backtest import execution_views, simulate
from .config import RULES_REVIEWED_THROUGH
from .experiment import MVO_POLICIES, benchmark_returns, clean
from .forward import Protocol, forward_folds, forward_scores, frozen, gates  # noqa: F401 (re-exported)

BEIJING = timezone(timedelta(hours=8))
LEDGER = {"daily": ["date"], "orders": ["date", "symbol", "side"], "holdings": ["date", "symbol"],
          "events": ["date", "symbol", "kind"], "allocations": ["date"]}
LIVE_STAGES = ("daily", "adjust", "industry", "constituents", "index", "master", "master_live", "dividend_live",
               "calendar_ahead")
# BaoStock's free node at times answers a login only after a minute and a half, or drops the connection
# (2026-10-19, the first forward session). While the cache grows no socket wait is shorter than
# FETCH_TIMEOUT seconds (acquire() sets 60), and a failed fetch is tried again every FETCH_RETRY_EVERY
# seconds until FETCH_RETRY_UNTIL o'clock Beijing time, long before the next evening's run.
FETCH_TIMEOUT = 300
FETCH_RETRY_EVERY = 15 * 60
FETCH_RETRY_UNTIL = 6


class GateFailed(RuntimeError):
    """The session's data gate failed and nobody accepted it: nothing was recorded."""


class MissingRecord(RuntimeError):
    """An earlier forward session has no record: resolve it before running later sessions."""


def stamp():
    return datetime.now(BEIJING).isoformat(timespec="seconds")


def differing_dates(old, new, keys):
    """Dates where two versions of a ledger table differ: rows on one side only, or any value."""
    dates = lambda f: {str(pd.Timestamp(d).date()) for d in f["date"]} if len(f) and "date" in f else set()
    if old.empty or new.empty:
        return sorted(dates(old) | dates(new))
    # The record also reads an empty key back as missing: a receivable's payment has no symbol.
    blank = lambda f: f.assign(**{k: f[k].astype(object).where(f[k].notna(), "") for k in keys if k != "date"})
    old, new = blank(old), blank(new)
    # Repeated keys (several events of one kind) pair up in order, not as a cross product.
    old = old.assign(_n=old.groupby(keys).cumcount())
    new = new.assign(_n=new.groupby(keys).cumcount())
    keys = keys + ["_n"]
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    differs = merged["_merge"] != "both"
    for column in [c for c in old.columns if c not in keys and c in new.columns]:
        a, b = merged[column + "_old"], merged[column + "_new"]
        # A CSV round trip changes dtypes (all-missing columns, booleans, dates): compare numbers as
        # numbers, everything else as text, and missing as equal to missing.
        na, nb = (pd.to_numeric(x.astype("datetime64[ns]").astype("int64") if pd.api.types.is_datetime64_any_dtype(x)
                                else x, errors="coerce") for x in (a, b))
        numeric = (na.notna() | a.isna()) & (nb.notna() | b.isna())
        same_number = np.isclose(na, nb, rtol=1e-9, atol=1e-9) | (na.isna() & nb.isna())
        same_text = a.astype(object).where(a.notna(), "").astype(str) == b.astype(object).where(b.notna(), "").astype(str)
        differs |= ~np.where(numeric, same_number, same_text)
    when = merged["date"] if "date" in merged else merged["date_old"].fillna(merged["date_new"])
    return sorted({str(pd.Timestamp(d).date()) for d in when[differs]})


def read_record(folder):
    files = sorted(Path(folder).glob("*.csv"))
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):  # a session without rows still counts as recorded
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for column in ("date", "decision_date", "decided", "execute"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame, {p.stem for p in files}


def on_days(frame, days):
    if frame.empty or "date" not in frame:
        return frame.iloc[0:0]
    return frame[frame.date.dt.strftime("%Y-%m-%d").isin(days)]


def record_session(out, replayed, day):
    """Write the session's rows of each ledger table once; compare every recorded session, today's
    too on a rerun, with the replay."""
    revisions = {}
    for name, keys in LEDGER.items():
        folder = out / "ledger" / name
        folder.mkdir(parents=True, exist_ok=True)
        frame = replayed[name]
        old, recorded = read_record(folder)
        dates = differing_dates(on_days(old, recorded), on_days(frame, recorded),
                                [k for k in keys if k in frame.columns or k in old.columns])
        if dates:
            revisions[name] = dates
        if day not in recorded:
            on_days(frame, {day}).to_csv(folder / f"{day}.csv", index=False)
    return revisions


def send_alert(out, level, message):
    """Local log always; the WeCom robot only once its webhook is provided in the environment."""
    line = f"{stamp()} {level} {message}"
    with (out / "alerts.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    hook = os.environ.get("COSTRANK_WECOM_WEBHOOK")
    if hook:
        body = json.dumps({"msgtype": "text", "text": {"content": line}}).encode("utf-8")
        request = urllib.request.Request(hook, data=body, headers={"Content-Type": "application/json"})
        try:
            answer = json.loads(urllib.request.urlopen(request, timeout=10).read().decode("utf-8") or "{}")
            problem = None if answer.get("errcode") == 0 else f"webhook answered {answer}"
        except (OSError, ValueError) as exc:
            problem = f"webhook failed: {exc}"
        if problem:
            with (out / "alerts.log").open("a", encoding="utf-8") as f:
                f.write(f"{stamp()} WARN {problem}\n")


def shortfall(orders, panel, day):
    """Implementation shortfall of the session's fills, in bps of the benchmark price and weighted by
    notional (positive = cost): against the fill session's previous close (the decision close,
    ex-date adjusted) and against that session's VWAP (amount / volume)."""
    if orders.empty:
        return {"fills": 0}
    filled = orders[(orders.date == pd.Timestamp(day)) & (orders.filled_qty > 0)]
    if filled.empty:
        return {"fills": 0}
    p = panel.set_index(["date", "symbol"])
    keys = pd.MultiIndex.from_arrays([filled.date, filled.symbol])
    sign = np.where(filled.side.eq("buy"), 1.0, -1.0)
    price, weight = filled.open_price.to_numpy(float), filled.notional.to_numpy(float)

    def bps(benchmark):
        ok = np.isfinite(benchmark) & (benchmark > 0) & (weight > 0)
        return float(np.average(sign[ok] * (price[ok] / benchmark[ok] - 1) * 1e4, weights=weight[ok])) if ok.any() else np.nan
    vwap = (p.amount / p.volume.where(p.volume > 0)).reindex(keys).to_numpy(float)
    return {"fills": int(len(filled)), "notional": float(weight.sum()),
            "vs_previous_close_bps": bps(p.preclose.reindex(keys).to_numpy(float)), "vs_session_vwap_bps": bps(vwap)}


def monitor(cfg, replayed, ic, day):
    """Alerts beyond data gates: constraint breaches, solver fallbacks, rejected orders, IC drift."""
    alerts = []
    a = replayed["allocations"]
    today = a[a.date == pd.Timestamp(day)] if len(a) else a
    for r in today.to_dict("records"):
        if r.get("status") == "hold_fallback":
            alerts.append(("WARN", f"{day} optimizer fell back to holding: {r.get('reason')}"))
        scale = r.get("risk_limit_scale")
        if scale is not None and np.isfinite(scale) and scale > 1:
            alerts.append(("WARN", f"{day} risk limits widened x{scale:.3f}: the trading limits allow no closer book"))
        if r.get("te_excess") is not None and np.isfinite(r.get("te_excess")):
            # The planner records the excess over the bounds it imposed (reachability and widening
            # included); beyond the audit tolerance it is a broken constraint, not a moved bound.
            for key, label in (("te_excess", "tracking error"), ("style_excess", "style exposure"),
                               ("industry_excess", "industry weight")):
                if r.get(key, 0.0) > 1e-4:
                    alerts.append(("ERROR", f"{day} {label} {r[key]:.2e} beyond its imposed bound"))
            continue
        te, style, industry = r.get("ex_ante_te"), r.get("max_style_active"), r.get("max_industry_active")
        if te is not None and np.isfinite(te) and te > cfg.te_annual * 1.01:
            alerts.append(("ERROR", f"{day} ex-ante tracking error {te:.4f} above {cfg.te_annual}"))
        if style is not None and np.isfinite(style) and style > cfg.style_band + 1e-6:
            alerts.append(("ERROR", f"{day} style exposure {style:.4f} above {cfg.style_band}"))
        if industry is not None and np.isfinite(industry) and industry > cfg.industry_band + 1e-6:
            alerts.append(("ERROR", f"{day} industry exposure {industry:.4f} above {cfg.industry_band}"))
    o = replayed["orders"]
    sent = o[o.date == pd.Timestamp(day)] if len(o) else o
    short = (sent.filled_qty < sent.requested_qty - 1e-9) if len(sent) else pd.Series(dtype=bool)
    if len(sent) and short.mean() > 0.2:
        alerts.append(("WARN", f"{day} {int(short.sum())} of {len(sent)} orders rejected or partly filled"))
    matured = ic.set_index("date").rank_ic.dropna() if len(ic) else pd.Series(dtype=float)
    weekly = matured.resample("W").mean().dropna() if len(matured) else matured
    if len(weekly) >= 4 and (weekly.iloc[-4:] < 0).all():
        alerts.append(("WARN", f"{day} mean rank IC of matured labels negative for 4 weeks in a row"))
    return alerts


def per_session(x, how):
    """Weekly statistic scaled to a five-session week, so holiday-shortened weeks compare fairly."""
    g = x.dropna().resample("W")
    return (g.mean() * 5 if how == "sum" else g.mean()).dropna()


def first_execution(calendar, protocol):
    """The session after the forward start: the first decision, made at the start's close, trades at its open."""
    c = pd.DatetimeIndex(calendar)
    later = c[c > pd.Timestamp(protocol.forward_start)]
    return later[0] if len(later) else None


def evaluation_benchmark(panel, bundle, protocol, day):
    """The benchmark of the forward evaluation (RESEARCH section 10). None on the forward start: the
    book is cash until the next open, so that session's pool return precedes every decision. On the
    first execution session the eligible pool's equal-weight open-to-close return, since the book is
    bought at that open (names without a bar that session are left out). Close to close from the
    next session on, as in development."""
    bench = benchmark_returns(panel, protocol.forward_start, day, bundle).benchmark_return
    bench = bench[bench.index > pd.Timestamp(protocol.forward_start)].copy()
    build = first_execution(bundle.calendar, protocol)
    if build is not None and build <= pd.Timestamp(day):
        p = panel.sort_values(["symbol", "date"])
        was = p.groupby("symbol").eligible.shift(fill_value=False).astype(bool)
        bars = p[was & p.date.eq(build) & p.open.gt(0)]
        bench.loc[build] = float((bars.close / bars.open - 1).mean()) if len(bars) else np.nan
    return bench.sort_index()


def reference_ranges(path, protocol):
    """5% and 95% quantiles of the development run's weekly statistics for the frozen model and policy."""
    run = Path(path)
    if not path or not (run / "manifest.json").exists():
        return {}
    scenario = f"slip_{protocol.slippage_bps:g}bps"
    daily = pd.read_csv(run / f"{protocol.model}__{protocol.policy}__{scenario}" / "daily.csv.gz", parse_dates=["date"])
    bench = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    invested = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["config"]["invested"]
    # As in the forward evaluation, without the first session (cash) and the build session.
    d = daily.set_index("date").iloc[2:]
    active = (d.net_return - invested * bench.reindex(d.index)).dropna()
    ic = pd.read_csv(run / "rank_ic.csv", parse_dates=["date"])
    ic = ic[ic.model == protocol.model].set_index("date").rank_ic
    q = lambda x: [float(x.quantile(0.05)), float(x.quantile(0.95))]
    return {"weekly_active_return": q(per_session(active, "sum")), "weekly_rank_ic": q(per_session(ic, "mean")),
            "weekly_turnover": q(per_session(d.turnover_two_way, "sum"))}


def weekly_report(out, day, protocol, replayed, ic, bench, invested, build=None):
    """`bench` is the evaluation benchmark; turnover leaves out the build session `build`."""
    d = replayed["daily"].set_index("date")
    week = pd.Timestamp(day).to_period("W")
    active = (d.net_return - invested * bench.reindex(d.index)).dropna()
    in_week = lambda s: s[s.index.to_period("W") == week]
    matured = ic.set_index("date").rank_ic.dropna() if len(ic) else pd.Series(dtype=float)
    # Labels need horizon + 1 sessions: this week's are not known yet, so the latest matured week stands in.
    ic_week = matured.index.to_period("W").max() if len(matured) else None
    # The build session's turnover is the initial purchase, not a rebalance: its week is not compared.
    building = build is not None and build.to_period("W") == week
    turnover = in_week(d.turnover_two_way[d.index > build] if build is not None else d.turnover_two_way)
    weekly = {"weekly_active_return": float(in_week(active).mean() * 5) if len(in_week(active)) else np.nan,
              "weekly_rank_ic": float(matured[matured.index.to_period("W") == ic_week].mean()) if ic_week else np.nan,
              "weekly_turnover": np.nan if building or turnover.empty else float(turnover.mean() * 5)}
    ranges = reference_ranges(protocol.reference_run, protocol)
    lines = [f"# 模拟盘周报 {week.start_time.date()} 至 {pd.Timestamp(day).date()}", "",
             f"周度数字按每周 5 个交易日折算；秩 IC 用标签已成熟的最近一周（{ic_week.start_time.date() if ic_week else '尚无'}"
             f"{' 起' if ic_week else ''}）。主动收益从首个成交日起算。", "",
             "| 指标 | 本周 | 开发期 5%–95% | 是否在区间内 |", "|---|---|---|---|"]
    blank = {"weekly_rank_ic": "未成熟", "weekly_turnover": "建仓周，不比较" if building else "—"}
    for key, label in (("weekly_active_return", "主动收益（算术和）"), ("weekly_rank_ic", "平均秩 IC（已成熟标签）"),
                       ("weekly_turnover", "双边换手")):
        value, band = weekly[key], ranges.get(key)
        inside = "—" if band is None or not np.isfinite(value) else ("是" if band[0] <= value <= band[1] else "否")
        shown = f"{band[0]:.4f} 至 {band[1]:.4f}" if band else "—"
        lines.append(f"| {label} | {blank.get(key, '—') if not np.isfinite(value) else f'{value:.4f}'} | {shown} | {inside} |")
    status = replayed["allocations"].get("status", pd.Series(dtype=str))
    cumulative = active.dropna()
    lines += ["", f"前向期累计：{len(cumulative)} 个交易日，主动收益算术和 {cumulative.sum() * 100:.2f}%，"
              f"已成熟标签的平均秩 IC {format(matured.mean(), '.4f') if len(matured) else '尚无'}，"
              f"无信号 {int((status == 'no_signal').sum())} 次，求解失败保持原仓 {int((status == 'hold_fallback').sum())} 次，"
              f"放宽风险限额 {int(replayed['allocations'].get('risk_limit_scale', pd.Series(dtype=float)).gt(1).sum())} 次。"]
    path = out / "reports" / f"week-{week.start_time.date()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, weekly, ranges


def attribution(risk, holdings, days, invested):
    """Each forward session's active return explained by the risk model: holdings at the previous
    close, active against `invested` x that close's equal-weight eligible pool, times the factor
    returns realized in the session (close to close; trading at the open and costs stay outside).
    From the session after the build: the forward start's close holds nothing."""
    rows = []
    for previous, day in zip(days[1:-1], days[2:]):
        snap, f = risk.snapshot(previous), risk.factor_returns.get(day)
        if snap is None or f is None:
            continue
        held = holdings[holdings.date == previous].set_index("symbol").weight if len(holdings) else pd.Series(dtype=float)
        rows.append({"date": day, **dict(zip(risk.factors, snap.exposure(snap.active(held, invested)) * f))})
    return pd.DataFrame(rows)


def attribution_lines(attrib, day):
    from .risk import STYLES
    if attrib.empty:
        return ["- 风险模型归因：从建仓后的次一交易日起算（按上一收盘的持仓），或前向期还没有可用的风险模型快照"]
    a = attrib.set_index("date")
    industry = [c for c in a.columns if c.startswith("ind_")]
    total = a[STYLES].sum().round(6).to_dict() | {"industry": float(a[industry].sum(axis=1).sum())}
    today = a.loc[pd.Timestamp(day)] if pd.Timestamp(day) in a.index else None
    lines = ["- 风险模型归因（收盘到收盘，单位为净值比例）：前向期累计 "
             + "，".join(f"{k} {v * 100:.2f}%" for k, v in sorted(total.items(), key=lambda x: -abs(x[1])))]
    if today is not None:
        largest = today[STYLES].abs().sort_values(ascending=False).index[:3]
        lines.append("  当日风格贡献最大的三项：" + "，".join(f"{k} {today[k] * 100:.3f}%" for k in largest)
                     + f"；行业合计 {today[industry].sum() * 100:.3f}%")
    return lines


def shortfall_line(slip):
    if not slip or not slip.get("fills"):
        return "- 实施落差：当日无成交"

    def bps(key):
        value = slip.get(key)
        return "—" if value is None or not np.isfinite(value) else f"{value:+.1f} bps"
    return (f"- 实施落差（成交额加权，正为成本）：{slip['fills']} 笔，成交额 {slip['notional']:,.0f}；"
            f"相对前收 {bps('vs_previous_close_bps')}，相对当日 VWAP {bps('vs_session_vwap_bps')}")


def report(out, day, protocol, replayed, bench, gate, revisions, pending, invested, slip=None, attrib=None, final=False):
    d = replayed["daily"].set_index("date")
    nav = d.nav
    active = (d.net_return - invested * bench.reindex(d.index)).dropna()  # `bench`: evaluation_benchmark
    te = (f"{len(active)} 个交易日的年化跟踪误差 {active.std(ddof=1) * np.sqrt(252) * 100:.2f}%" if len(active) > 1
          else "跟踪误差从第二个成交日起计算")
    row = d.loc[pd.Timestamp(day)]
    holdings = replayed["holdings"]
    today = (holdings[holdings.date == pd.Timestamp(day)].sort_values("weight", ascending=False) if len(holdings)
             else pd.DataFrame(columns=["symbol", "quantity", "weight"]))
    orders = replayed["orders"]
    filled = orders[(orders.date == pd.Timestamp(day))] if len(orders) else orders
    trades = pending[pending.target != pending.held] if len(pending) else pending
    fees = float(filled[[c for c in ("commission", "stamp", "transfer", "slippage") if c in filled]].sum().sum()) if len(filled) else 0.0
    lines = [f"# 模拟盘日报 {day}", "",
             f"协议：{protocol.model} + {protocol.policy}，前向起点 {protocol.forward_start}，初始资金 {protocol.initial_cash:,.0f}。", "",
             f"- 数据闸门：{'通过' if gate['passed'] else '未通过，经人工确认按无信号记账'} {json.dumps(gate['checks'], ensure_ascii=False)}",
             f"- 净值 {row.nav:,.2f}，当日收益 {row.net_return * 100:.2f}%，累计 {(nav.iloc[-1] / protocol.initial_cash - 1) * 100:.2f}%",
             (f"- 相对 {invested:.2f} × 合格池等权的累计主动收益（算术和，从首个成交日起算）{active.sum() * 100:.2f}%，{te}"
              if len(active) else "- 主动收益：起点当天只决策、账户全为现金，从次一交易日（首个成交日）起算"),
             f"- 持仓 {len(today)} 只，现金 {row.cash:,.2f}；当日成交 {int((filled.filled_qty > 0).sum()) if len(filled) else 0} 笔，"
             f"费用 {fees:,.2f}",
             (f"- 次日指令：前向期最后一个交易日，交易规则只核实到 {RULES_REVIEWED_THROUGH}，收盘不再决策" if final else
              f"- 次日（{pending.execute.iloc[0].date() if len(pending) else '—'}）指令：{len(trades)} 只股票有变动"
              "（股数按决策日收盘口径；次日开盘若有除权，账本按除权比例调整后执行，权重不变）"),
             shortfall_line(slip),
             *attribution_lines(attrib if attrib is not None else pd.DataFrame(), day),
             f"- 数据修订：{json.dumps(revisions, ensure_ascii=False) if revisions else '无'}", "",
             "| 股票 | 股数 | 权重 |", "|---|---|---|"]
    lines += [f"| {r.symbol} | {r.quantity:,.0f} | {r.weight * 100:.2f}% |" for r in today.head(20).itertuples()]
    path = out / "reports" / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def orders_with_weights(pending, panel, nav, day):
    """Tomorrow's orders as decided at today's close: shares and their weights at that close."""
    close = panel[panel.date == pd.Timestamp(day)].set_index("symbol").close
    price = pending.symbol.map(close)
    return pending.assign(decision_close=price, held_weight=pending.held * price / nav,
                          target_weight=pending.target * price / nav)


def run_day(root, config, protocol, out, day, next_session, *, today_gate=None, bundle=None):
    """Signals, replay, record and report for one session whose raw files are already cached."""
    from .acquire import trading_days
    from .snapshot import assemble
    out = Path(out)
    (out / "gates").mkdir(parents=True, exist_ok=True)
    if day > RULES_REVIEWED_THROUGH:
        raise ValueError(f"Trading rules are reviewed only through {RULES_REVIEWED_THROUGH}")
    final = next_session > RULES_REVIEWED_THROUGH  # its close decides nothing: the orders would trade beyond the rules
    plan = json.loads((Path(root) / "collection-plan.json").read_text(encoding="utf-8"))
    calendar = ([d.strftime("%Y-%m-%d") for d in bundle.calendar] if bundle is not None
                else trading_days(root, plan["start"], day))
    unrecorded = [d for d in calendar if protocol.forward_start <= d < day
                  and not (out / "ledger" / "daily" / f"{d}.csv").exists()]
    if unrecorded:
        send_alert(out, "ERROR", f"{day} not run: earlier sessions have no record {unrecorded[:5]}")
        raise MissingRecord(f"Earlier forward sessions have no record: {unrecorded[:5]}")
    earlier = [d for d in calendar if d < day]
    gate = today_gate if today_gate is not None else gates(root, day, earlier[-1] if earlier else None)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (out / "gates" / f"{day}@{attempt}.json").write_text(json.dumps(clean(gate), ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    accepted = (out / "gates" / f"{day}.accept").exists()
    if not gate["passed"] and not accepted:
        send_alert(out, "ERROR", f"{day} data gate failed, nothing recorded: {json.dumps(gate['checks'], ensure_ascii=False)}"
                                 f" {json.dumps(clean(gate.get('examples', {})), ensure_ascii=False)}")
        raise GateFailed(f"{day} data gate failed: fix the data and rerun, or accept the session as no signal")
    if bundle is None:
        snapshot = out / "snapshots" / day
        if (snapshot / "metadata.json").exists():  # a rerun of the same session reads what it sealed
            from .data import read_bundle
            bundle = read_bundle(snapshot)
        else:
            resolutions = protocol.resolutions if Path(protocol.resolutions).exists() else None
            bundle = assemble(root, plan["start"], day, resolutions=resolutions, target=snapshot, forward=True)
    cfg, panel, scores, audits = forward_scores(bundle, config, protocol, day)
    folder = out / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    stored, recorded = read_record(folder)
    revisions = {}
    dates = differing_dates(on_days(stored, recorded), on_days(scores, recorded), ["date", "symbol"])
    if dates:
        revisions["scores"] = dates
    if gate["passed"] and day not in recorded:
        on_days(scores, {day}).to_csv(folder / f"{day}.csv", index=False)
    stored, _ = read_record(folder)  # only recorded scores decide; an accepted failed gate leaves the book alone
    stored = stored if len(stored) else pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "symbol": [], "score": []})
    from .factors import market
    from .risk import RiskModel
    days = bundle.calendar[(bundle.calendar >= protocol.forward_start) & (bundle.calendar <= day)]
    # Every forward session is kept for attribution; the optimizer reads only its decision days.
    risk = RiskModel(panel, market(panel, bundle), bundle, cfg, keep=days)
    planner = None
    if protocol.policy in MVO_POLICIES:
        from .optimize import Planner, matured_ic
        info = [dict(test_start=a["test_start"], test_end=a["test_end"], valid_ic=a["models"][protocol.model].get("valid_ic"),
                     valid_rho5=a["models"][protocol.model].get("valid_rho5")) for a in audits]
        planner = Planner(cfg, risk, matured_ic(stored, panel), info, objective=protocol.policy)
    ledger_cfg = replace(cfg, initial_cash=protocol.initial_cash)
    replayed = simulate(panel, stored, bundle, ledger_cfg, protocol.policy, slippage_bps=protocol.slippage_bps,
                        views=execution_views(panel, protocol.forward_start, day), planner=planner,
                        period=(protocol.forward_start, day), next_session=None if final else next_session)
    revisions.update(record_session(out, replayed, day))
    nav = float(replayed["daily"].set_index("date").nav.loc[pd.Timestamp(day)])
    if final:
        pending = pd.DataFrame(columns=["decided", "execute", "symbol", "held", "target"])
    else:
        pending = orders_with_weights(replayed["pending"], panel, nav, day)
        orders = out / "orders" / f"{next_session}.csv"
        orders.parent.mkdir(parents=True, exist_ok=True)
        if orders.exists():
            old = pd.read_csv(orders, parse_dates=["decided", "execute"])
            if differing_dates(old.rename(columns={"decided": "date"}), pending.rename(columns={"decided": "date"}), ["symbol"]):
                revisions["orders"] = [day]
        else:
            pending.to_csv(orders, index=False)
    bench = evaluation_benchmark(panel, bundle, protocol, day)
    slip = shortfall(replayed["orders"], panel, day)
    attrib = attribution(risk, replayed["holdings"], days, cfg.invested)
    path = report(out, day, protocol, replayed, bench, gate, revisions, pending, cfg.invested, slip, attrib, final)
    from .metrics import rank_ic
    ic = rank_ic(stored.assign(model=protocol.model), panel) if len(stored) else pd.DataFrame()
    for level, message in monitor(cfg, replayed, ic, day):
        send_alert(out, level, message)
    if pd.Timestamp(next_session).to_period("W") != pd.Timestamp(day).to_period("W"):
        _, weekly, ranges = weekly_report(out, day, protocol, replayed, ic, bench, cfg.invested,
                                          first_execution(bundle.calendar, protocol))
        outside = [k for k, v in weekly.items() if k in ranges and np.isfinite(v) and not ranges[k][0] <= v <= ranges[k][1]]
        if outside:
            send_alert(out, "WARN", f"{day} weekly statistics outside the development 5-95% range: {outside}")
    if not gate["passed"]:
        send_alert(out, "WARN", f"{day} recorded as no signal after an accepted gate failure")
    if revisions:
        send_alert(out, "WARN", f"{day} replay differs from the record: {json.dumps(revisions, ensure_ascii=False)}")
    send_alert(out, "INFO", f"{day} done: NAV {nav:,.2f}; report {path}")
    return {"day": day, "gate": gate, "revisions": revisions, "report": str(path)}


class PatientSocket:
    """The socket module, except that its default timeout is never set below FETCH_TIMEOUT."""

    def __getattr__(self, name):
        return getattr(socket, name)

    @staticmethod
    def setdefaulttimeout(seconds):
        socket.setdefaulttimeout(seconds if seconds is None else max(seconds, FETCH_TIMEOUT))


def fetch_until(root, through, out, *, reports=False, clock=None, sleep=time.sleep):
    """fetch(), and after a failure again every FETCH_RETRY_EVERY seconds until FETCH_RETRY_UNTIL
    o'clock Beijing time, each try resuming from the raw cache. The first failure is alerted; the
    last one is raised."""
    clock = clock or (lambda: datetime.now(BEIJING))
    start = clock()
    deadline = (start + timedelta(days=1 if start.hour >= FETCH_RETRY_UNTIL else 0)).replace(
        hour=FETCH_RETRY_UNTIL, minute=0, second=0, microsecond=0)
    attempt = 0
    while True:
        attempt += 1
        try:
            return fetch(root, through, reports=reports)
        except RuntimeError as exc:
            if clock() + timedelta(seconds=FETCH_RETRY_EVERY) > deadline:
                raise
            if attempt == 1:
                send_alert(out, "WARN", f"{through} fetch failed, trying again every {FETCH_RETRY_EVERY // 60} "
                                        f"minutes until {deadline:%Y-%m-%d %H:%M}: {exc}")
            print(f"{stamp()} fetch attempt {attempt} failed: {exc}", flush=True)
            sleep(FETCH_RETRY_EVERY)


def fetch(root, through, *, reports=False, budget=45_000):
    """Grow the live raw cache to `through` (network): new sessions, uncovered ranges, a master
    refresh, a month of calendar ahead, and a fresh dividend version per new ex-date."""
    from . import acquire as acquisition
    root = Path(root)
    plan = json.loads((root / "collection-plan.json").read_text(encoding="utf-8"))
    base = json.loads((root / "extended-from.json").read_text(encoding="utf-8"))["source_plan"]["end"]
    stages = LIVE_STAGES + (("reports",) if reports else ())
    # acquire.py is protocol code (hashed at the freeze), so its 60-second timeout is lengthened from here.
    acquisition.socket = PatientSocket()
    try:
        summary = acquisition.acquire(root, plan["start"], through, stages=stages, budget=budget, extend=True,
                                      live_since=acquisition.shift_day(base, 1))
    finally:
        acquisition.socket = socket
    if summary.get("stopped"):
        raise RuntimeError(f"Request budget exhausted before the cache reached {through}: {summary['stopped']}")
    return summary


def set_aside(root, day):
    """Move a session's cached market and adjustment files to raw/rejected/ (kept as evidence) so the
    next fetch requests them again. Only for a session that is not recorded yet."""
    raw = Path(root) / "raw"
    moved = []
    for folder in ("daily", "adjust"):
        path = raw / folder / f"{day}.csv"
        if path.exists():
            target = raw / "rejected" / folder / f"{day}@{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            moved.append(str(target))
    return moved


def daily(root, freeze_file, protocol_file, output, *, day=None, do_fetch=True, now=None, refetch=False):
    """`costrank daily`: grow the cache to the latest published session, then run `day`, or without it
    every forward session not yet recorded, in order (the latest again, as a comparison, when all are).
    An unattended run whose data gate fails sets that session's raw files aside, so the next run
    fetches them again; nothing is recorded for it, as for every failed gate."""
    from .acquire import trading_days
    config, _ = frozen(freeze_file)
    protocol = Protocol.load(protocol_file)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    today = (now or datetime.now(BEIJING)).strftime("%Y-%m-%d")
    if refetch:
        if not day:
            raise ValueError("--refetch needs --day")
        if (out / "ledger" / "daily" / f"{day}.csv").exists():
            raise ValueError(f"{day} is already recorded; its raw files are part of the record")
        send_alert(out, "INFO", f"{day} raw files set aside for a new fetch: {set_aside(root, day)}")
    if do_fetch:
        target = day or today
        weekday = pd.Timestamp(target).weekday()
        try:
            fetch_until(root, target, out, reports=config.feature_set == "v04" and weekday == protocol.reports_weekday)
        except Exception as exc:
            send_alert(out, "ERROR", f"{target} fetch failed: {exc}")
            raise
    plan = json.loads((Path(root) / "collection-plan.json").read_text(encoding="utf-8"))
    if day:
        todo = [day]
    else:  # through the last session the cache was grown to
        cached = [d for d in trading_days(root, plan["start"], plan["end"])
                  if protocol.forward_start <= d <= RULES_REVIEWED_THROUGH and (Path(root) / "raw" / "daily" / f"{d}.csv").exists()]
        todo = [d for d in cached if not (out / "ledger" / "daily" / f"{d}.csv").exists()] or cached[-1:]
        if not todo:
            send_alert(out, "INFO", f"nothing to run before the forward start {protocol.forward_start}")
            return None
    result = None
    for session in todo:
        try:
            result = run_day(root, config, protocol, out, session, next_trading_day(root, session))
        except GateFailed:
            if not day:
                send_alert(out, "WARN", f"{session} raw files set aside for the next fetch: {set_aside(root, session)}")
            raise
        except MissingRecord:
            raise
        except Exception as exc:
            send_alert(out, "ERROR", f"{session} run failed: {type(exc).__name__}: {exc}")
            raise
    return dict(result, sessions_run=todo)


def next_trading_day(root, day):
    """The first session after `day` in the cached calendars, which must cover the days between."""
    from .acquire import calendar_parts, gaps, shift_day
    parts = calendar_parts(root)
    cal = pd.concat([f for *_, f in parts], ignore_index=True).drop_duplicates("calendar_date")
    later = sorted(d for d in cal.loc[cal.is_trading_day.eq("1"), "calendar_date"] if d > day)
    if not later or gaps(parts, shift_day(day, 1), later[0]):
        raise ValueError(f"The cached calendar does not reach the session after {day}")
    return later[0]
