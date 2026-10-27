"""Forward paper-trading report as declared in RESEARCH section 10, written before the first forward
session. Offline: reads the paper output's records (ledger, scores, gates), the latest snapshot the
daily run sealed (for the eligible-pool benchmark, labels and prices) and the frozen development run
named by the protocol (for the comparison distributions).

usage: python scripts/forward_report.py <paper_output> --protocol configs/forward.json
           --freeze <freeze record> --output forward-report.json [--markdown report.md]

Descriptive only: net information ratio against 0.95 x the eligible-pool equal weight with a 95%
circular block bootstrap interval (block 20, 10,000 samples, seed 42), its percentile among the
development run's rolling windows of the same length, turnover and cost drag beside development,
matured rank IC, implementation shortfall and the sessions without a signal. Returns count from the
first execution session (live.evaluation_benchmark); turnover and cost drag from the session after
it, the build session's shown apart; the development figures leave out the same two first sessions.
Revision two of section 10 (also before the first forward session, also descriptive) shows beside
the benchmark the wide equal weight (every name listed 180 days, tradable and non-ST, without the
ADV floor or rank cut) and the CSI 1000 and CSI 500 price indices over the same window, with the net
return's active return and information ratio against 0.95 x each. With n forward sessions the standard error of one information ratio is about sqrt(252 / n): 60
sessions cannot confirm or refute ratios below about 5.7.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.data import read_bundle
from costrank.experiment import MVO_POLICIES, clean
from costrank.forward import Protocol, frozen
from costrank.live import evaluation_benchmark, first_execution, read_record, shortfall
from costrank.metrics import minimum_detectable, rank_ic
from holdout_report import block_interval, information_ratio

COSTS = ["commission", "stamp", "transfer", "slippage", "dividend_tax"]
INDICES = {"sh.000852": "CSI 1000 price index", "sh.000905": "CSI 500 price index"}


def panel_of(bundle, cfg):
    """The panel the forward run builds for this configuration (labels, eligibility, prices)."""
    if cfg.feature_set == "v04" or MVO_POLICIES & set(cfg.policies):
        from costrank.signals import prepare_v04
        return prepare_v04(bundle, cfg, library=cfg.feature_set == "v04")[0]
    from costrank.features import prepare
    return prepare(bundle, cfg)


def cost_drag(daily):
    d = daily.set_index("date") if "date" in daily else daily
    if d.empty:
        return np.nan
    return float((d[[c for c in COSTS if c in d]].sum(axis=1) / d.nav.shift().fillna(d.nav.iloc[0])).mean() * 252)


def rolling_ratios(active, n):
    """Information ratio of every window of n consecutive sessions."""
    x = active.to_numpy(float)
    if len(x) < n or n < 2:
        return np.array([])
    windows = np.lib.stride_tricks.sliding_window_view(x, n)
    sd = windows.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, windows.mean(axis=1) / sd * np.sqrt(252), np.nan)


def comparison_returns(panel, bundle, cfg, build, dates):
    """Daily returns of the revision-two comparisons on `dates`. The wide equal weight holds every
    name listed `min_listing_days`, tradable and non-ST at the previous close, on total-return closes
    with delisting outcomes left out, as scripts/benchmark_decompose.py. On the build session each
    comparison is taken open to close, as the evaluation benchmark is."""
    p = panel.sort_values(["symbol", "date"])
    listed = p.symbol.map(bundle.instruments.set_index("symbol").listed)
    wide = ((p.date - listed).dt.days >= cfg.min_listing_days) & p.tradable.eq(1) & p.is_st.eq(0)
    was = wide.groupby(p.symbol).shift(fill_value=False).astype(bool)
    r = p.groupby("symbol").tr_close.pct_change()
    if build is not None:
        r = r.mask(p.date.eq(build), (p.close / p.open - 1).where(p.open.gt(0)))
    series = {"wide equal weight": r[was].groupby(p.date[was]).mean()}
    idx = bundle.tables.get("indices")
    for code, name in INDICES.items():
        x = idx[idx["index"] == code] if idx is not None else None
        if x is None or x.empty:
            series[name] = pd.Series(dtype=float)
            continue
        x = x.assign(date=pd.to_datetime(x.date)).set_index("date").sort_index()
        ret = x.close.pct_change()
        if build is not None and build in x.index:
            ret.loc[build] = x.close.loc[build] / x.open.loc[build] - 1 if x.open.loc[build] > 0 else np.nan
        series[name] = ret
    return pd.DataFrame({k: v.reindex(pd.DatetimeIndex(dates)) for k, v in series.items()})


def against(net, comparison, invested):
    """A comparison's compounded return over the sessions it covers, and the net return's active
    return against `invested` x the comparison on those sessions."""
    c = comparison.dropna()
    active = (net.reindex(c.index) - invested * c).dropna()
    return {"sessions": int(len(c)), "total_return": float((1 + c).prod() - 1) if len(c) else np.nan,
            "active_return_arith_annual": float(active.mean() * 252) if len(active) else np.nan,
            "information_ratio": information_ratio(active) if len(active) > 1 else np.nan}


def development(run, protocol, n):
    """The frozen development run's same model, policy and slippage: whole-period figures and the
    distribution over windows as long as the forward record."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    name = f"{protocol.model}__{protocol.policy}__slip_{protocol.slippage_bps:g}bps"
    # Without the first session (cash) and the build session, as in the forward evaluation.
    daily = pd.read_csv(run / name / "daily.csv.gz", parse_dates=["date"]).set_index("date").iloc[2:]
    bench = pd.read_csv(run / "benchmark.csv", parse_dates=["date"]).set_index("date").benchmark_return
    active = (daily.net_return - manifest["config"]["invested"] * bench.reindex(daily.index)).dropna()
    ic = pd.read_csv(run / "rank_ic.csv", parse_dates=["date"])
    ic = ic[ic.model == protocol.model].set_index("date").rank_ic.dropna()
    windows = rolling_ratios(active, n)
    ic_windows = ic.rolling(n).mean().dropna().to_numpy() if len(ic) >= n else np.array([])
    q = lambda x: [float(np.nanquantile(x, 0.05)), float(np.nanquantile(x, 0.95))] if len(x) else None
    return {"run": str(run), "scenario": name, "sessions": int(len(active)),
            "information_ratio": information_ratio(active), "mean_two_way_turnover": float(daily.turnover_two_way.mean()),
            "annual_cost_drag": cost_drag(daily), "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
            "window_sessions": int(n), "windows": int(len(windows)), "window_ir_5_95": q(windows[np.isfinite(windows)]),
            "window_mean_ic_5_95": q(ic_windows)}, windows


def report(paper, protocol_file, freeze_file, *, samples=10_000):
    paper = Path(paper)
    protocol = Protocol.load(protocol_file)
    cfg, record = frozen(freeze_file)
    daily, recorded = read_record(paper / "ledger" / "daily")
    if daily.empty:
        raise ValueError("No forward session recorded yet")
    daily = daily.sort_values("date").reset_index(drop=True)
    last = daily.date.max().strftime("%Y-%m-%d")
    bundle = read_bundle(paper / "snapshots" / last)
    panel = panel_of(bundle, cfg)
    bench = evaluation_benchmark(panel, bundle, protocol, last)
    d = daily.set_index("date")
    active = (d.net_return - cfg.invested * bench.reindex(d.index)).dropna()
    n = int(len(active))
    build = first_execution(bundle.calendar, protocol)
    steady = d[d.index > build] if build is not None else d.iloc[0:0]
    out = {"protocol": json.loads(Path(protocol_file).read_text(encoding="utf-8")),
           "freeze": {k: record.get(k) for k in ("frozen_at", "code_hash", "protocol_code_hash", "config_hash")},
           "first_session": str(d.index.min().date()), "last_session": last, "sessions": n,
           "evaluation_start": str(active.index.min().date()) if n else None}
    out["forward"] = {"information_ratio": information_ratio(active),
                      "ir_interval_95": block_interval(active, samples=samples) if n >= 20 else None,
                      "active_return_arith_annual": float(active.mean() * 252),
                      "tracking_error_annual": float(active.std(ddof=1) * np.sqrt(252)) if n > 1 else np.nan,
                      "total_return": float(d.nav.iloc[-1] / protocol.initial_cash - 1),
                      "max_drawdown": float((d.nav / d.nav.cummax() - 1).min()),
                      "mean_two_way_turnover": float(steady.turnover_two_way.mean()) if len(steady) else np.nan,
                      "annual_cost_drag": cost_drag(steady)}
    if build is not None and build in d.index:
        out["build_session"] = {"date": str(build.date()), "two_way_turnover": float(d.turnover_two_way.loc[build]),
                                "costs": float(d.loc[build, [c for c in COSTS if c in d]].sum())}
    net = d.net_return.reindex(active.index)
    series = {"eligible-pool benchmark": bench.reindex(active.index),
              **dict(comparison_returns(panel, bundle, cfg, build, active.index).items())}
    out["comparisons"] = {"invested": cfg.invested, "min_listing_days": cfg.min_listing_days,
                          "net_total_return": float((1 + net).prod() - 1) if n else np.nan,
                          "series": {k: against(net, v, cfg.invested) for k, v in series.items()}}
    out["minimum_detectable_ratio"] = float(minimum_detectable(np.sqrt(252 / n))) if n > 1 else None
    scores, _ = read_record(paper / "scores")
    if len(scores):
        ic = rank_ic(scores.assign(model=protocol.model), panel)
        matured = ic.set_index("date").rank_ic.dropna()
        out["forward"]["mean_rank_ic"] = float(matured.mean()) if len(matured) else np.nan
        out["forward"]["matured_ic_sessions"] = int(len(matured))
    orders, _ = read_record(paper / "ledger" / "orders")
    fills = [shortfall(orders, panel, day) for day in sorted(recorded)] if len(orders) else []
    fills = [f for f in fills if f.get("fills")]
    notional = np.array([f["notional"] for f in fills], float)
    for key in ("vs_previous_close_bps", "vs_session_vwap_bps"):
        values = np.array([f[key] for f in fills], float)
        ok = np.isfinite(values) & (notional > 0)
        out.setdefault("shortfall", {})[key] = float(np.average(values[ok], weights=notional[ok])) if ok.any() else np.nan
    out["shortfall"]["fill_sessions"] = len(fills)
    allocations, _ = read_record(paper / "ledger" / "allocations")
    status = allocations.get("status", pd.Series(dtype=str))
    gates = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((paper / "gates").glob("*@*.json"))]
    out["operations"] = {"no_signal": int((status == "no_signal").sum()), "hold_fallback": int((status == "hold_fallback").sum()),
                         "gate_attempts": len(gates), "gate_failures": int(sum(not g.get("passed") for g in gates)),
                         "accepted_sessions": sorted(p.stem for p in (paper / "gates").glob("*.accept"))}
    if protocol.reference_run and (Path(protocol.reference_run) / "manifest.json").exists():
        out["development"], windows = development(protocol.reference_run, protocol, n)
        finite = windows[np.isfinite(windows)]
        ir = out["forward"]["information_ratio"]
        out["forward"]["percentile_among_development_windows"] = (
            float((finite <= ir).mean()) if len(finite) and np.isfinite(ir) else None)
    return clean(out)


def markdown(r):
    f, dev = r["forward"], r.get("development") or {}
    pct = lambda x: "—" if x is None or not np.isfinite(x) else f"{x * 100:.2f}%"
    num = lambda x, k=3: "—" if x is None or not np.isfinite(x) else f"{x:.{k}f}"
    band = lambda b, k=3, p=False: "—" if not b else (f"{pct(b[0])} 至 {pct(b[1])}" if p else f"{num(b[0], k)} 至 {num(b[1], k)}")
    lines = [f"# 模拟盘前向报告 {r['evaluation_start']} 至 {r['last_session']}（{r['sessions']} 个交易日）", "",
             f"协议：{r['protocol']['model']} + {r['protocol']['policy']}，前向起点 {r['protocol']['forward_start']}；"
             f"冻结记录 {r['freeze']['frozen_at']}（协议代码 {(r['freeze']['protocol_code_hash'] or '')[:12]}）。只描述，不做采纳判定。", "",
             "| 指标 | 前向 | 开发期（同配置） |", "|---|---|---|",
             f"| 年化净信息比率（相对 0.95 × 合格池等权） | {num(f['information_ratio'])}"
             f"（95% 区间 {band(f.get('ir_interval_95'))}） | {num(dev.get('information_ratio'))} |",
             f"| 同长度窗口信息比率 5%–95% | — | {band(dev.get('window_ir_5_95'))} |",
             f"| 前向信息比率在开发期窗口中的分位 | {pct(f.get('percentile_among_development_windows'))} | — |",
             f"| 年化主动收益（算术）/ 跟踪误差 | {pct(f['active_return_arith_annual'])} / {pct(f['tracking_error_annual'])} | — |",
             f"| 日均双边换手 | {pct(f['mean_two_way_turnover'])} | {pct(dev.get('mean_two_way_turnover'))} |",
             f"| 年化费用拖累 | {pct(f['annual_cost_drag'])} | {pct(dev.get('annual_cost_drag'))} |",
             f"| 平均秩 IC（已成熟标签） | {num(f.get('mean_rank_ic'), 4)} | {num(dev.get('mean_rank_ic'), 4)}"
             f"（同长度窗口 5%–95% {band(dev.get('window_mean_ic_5_95'), 4)}） |",
             f"| 累计收益 / 最大回撤 | {pct(f['total_return'])} / {pct(f['max_drawdown'])} | — |", "",
             "主动收益从首个成交日起算（起点当天只决策、账户全为现金；首个成交日的基准取合格池开盘到收盘），"
             "换手与费用从其后一日起算，开发期同样去掉前两个交易日。"
             + (f"建仓日 {r['build_session']['date']}：双边换手 {pct(r['build_session']['two_way_turnover'])}，"
                f"费用 {r['build_session']['costs']:,.2f} 元。" if r.get("build_session") else ""),
             f"实施落差（成交额加权，正为成本，{r['shortfall']['fill_sessions']} 个成交日）：相对前收 "
             f"{num(r['shortfall']['vs_previous_close_bps'], 1)} bps，相对当日 VWAP {num(r['shortfall']['vs_session_vwap_bps'], 1)} bps。",
             f"运行：无信号 {r['operations']['no_signal']} 次，求解失败保持原仓 {r['operations']['hold_fallback']} 次，"
             f"闸门尝试 {r['operations']['gate_attempts']} 次、失败 {r['operations']['gate_failures']} 次，"
             f"人工确认按无信号记账的交易日 {r['operations']['accepted_sessions'] or '无'}。"]
    c = r.get("comparisons")
    if c:
        label = {"eligible-pool benchmark": "合格池等权（主基准）",
                 "wide equal weight": f"宽口径等权（上市满 {c['min_listing_days']} 天、可交易、非 ST）",
                 "CSI 1000 price index": "中证 1000（价格指数）", "CSI 500 price index": "中证 500（价格指数）"}
        lines += ["", f"对照（§10 修订二，只描述）：同一区间组合净收益 {pct(c['net_total_return'])}，"
                  f"主动收益为组合净收益减 {c['invested']:g} × 对照。", "",
                  "| 对照 | 区间收益 | 年化主动收益（算术） | 信息比率 | 交易日 |", "|---|---|---|---|---|"]
        lines += [f"| {label.get(k, k)} | {pct(v['total_return'])} | {pct(v['active_return_arith_annual'])} | "
                  f"{num(v['information_ratio'])} | {v['sessions']} |" for k, v in c["series"].items()]
        lines += ["", "指数是价格指数，不含分红；组合和两个等权池含分红，宽口径等权不计退市结果。"]
    lines += ["", f"{r['sessions']} 个交易日上单条信息比率的标准误约 {np.sqrt(252 / max(r['sessions'], 1)):.2f}，"
                  f"最小可检出值约 {num(r.get('minimum_detectable_ratio'), 1)}；只看方向与量级是否落在开发期分布内，不能确认或否定。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    r = report(args.paper, args.protocol, args.freeze, samples=args.samples)
    Path(args.output).write_text(json.dumps(r, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(markdown(r), encoding="utf-8")
    print(json.dumps({"sessions": r["sessions"], "forward": r["forward"]}, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
