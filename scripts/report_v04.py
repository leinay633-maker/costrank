"""Render the v0.4 result as Markdown from the decision and the report-only JSON files. Offline.

usage: python scripts/report_v04.py --decision runs/audit/v04-decision.json
           [--feasibility f.json] [--risk-bias b.json] [--style s.json] [--factors factor-report.json]
           [--gates g1.json g2.json g3.json] [--audits a.json ...] --output report.md

Every number comes from the inputs. The conclusion follows RESEARCH section 9: the freeze rule, and
"not detected at the pre-declared minimum detectable effect" when the primary test is not
significant, never "the old method is better" or "this layer does not work".
"""
import argparse
import json
from pathlib import Path
import numpy as np

CONDITIONS = {
    "primary_p_below_0.05": "主要检验双侧 p < 0.05 且差值为正",
    "ridge_point_estimate_positive": "Ridge 同一比较的点估计为正",
    "s4p4_not_below_s2p4_and_s4p2_cost": "LightGBM S4P4 信息比率不低于 S2P4 与 S4P2-cost",
    "s4p4_fallback_at_most_5pct": "S4P4 的 hold_fallback 与放宽风险限额的再平衡日合计占比 ≤ 5%",
    "tracking_errors_at_least_2pct": "两侧年化跟踪误差都 ≥ 2%",
}
CELLS = ("S2P2", "S4P2", "S2P4", "S4P4")


def num(x, digits=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.{digits}f}%" if pct else f"{x:.{digits}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def decision_section(d):
    test, cond = d["primary_test"], d["conditions"]
    lines = ["## 判定（RESEARCH §9）", "",
             f"冻结协议：**{d['frozen_protocol']}**。", ""]
    lines += table(["条件", "结果"], [[CONDITIONS.get(k, k), "满足" if v else "不满足"] for k, v in cond.items()])
    lines += ["", "主要检验：10 bps、开盘价成交、初始资金 100 万，LightGBM 的 S4P4 对 S2P2-cost，相对 0.95 × 合格池等权的年化信息比率之差（"
              + test.get("method", "") + f"，块长 {test.get('block')}，{test.get('samples')} 次）。", ""]
    lines += table(["项目", "数值"], [
        ["信息比率差（年化）", num(test["sharpe_difference_annual"], 3)],
        ["标准误（年化）", num(test["se_annual"], 3)],
        ["p 值", num(test["p_value"], 4)],
        ["S4P4 / S2P2 信息比率", f"{num(test['sharpe_left_annual'], 3)} / {num(test['sharpe_right_annual'], 3)}"],
        ["交易日", test["sessions"]],
        ["最小可检出差（双侧 5%、功效 80%）", num(d["minimum_detectable_ir_difference"], 2)],
        ["Ridge 同一比较的点估计", num(d["ridge_ir_difference"], 3)],
        ["S4P4 hold_fallback 与放宽风险限额占比", num(d["s4p4_fallback_share"], 2, pct=True)],
        ["2024-01-01 以前 / 以后的信息比率差",
         f"{num(d['ir_difference_by_period'].get('before_2024'), 3)} / {num(d['ir_difference_by_period'].get('from_2024'), 3)}"],
        ["去掉 2024 年一季度后的信息比率差 / 一季度内", f"{num(d.get('first_quarter_2024', {}).get('excluding_2024q1'), 3)} / "
         f"{num(d.get('first_quarter_2024', {}).get('2024q1'), 3)}"]])
    if d.get("first_quarter_2024", {}).get("mainly_2024q1"):
        lines += ["", "全样本的正差值主要来自 2024 年一季度（去掉该季度后不再为正）。按 §9 预先写明：这与两侧的规模暴露有关，"
                  "不能当作整个开发期稳定存在的差别。"]
    failed = [CONDITIONS.get(k, k) for k, v in cond.items() if not v]
    lines.append("")
    if not failed:
        lines.append("全部条件满足，按 §9 冻结 LightGBM + S4 + `mvo`。")
    else:
        lines.append("未满足：" + "；".join(failed) + "。按 §9 冻结 LightGBM + S2 + `cost`。")
        if not cond.get("primary_p_below_0.05", False):
            lines.append(f"按预先写明的最小可检出效应（约 {num(d['minimum_detectable_ir_difference'], 2)}），"
                         "此次未检出 S4P4 相对 S2P2-cost 的信息比率差；这不表示旧方法更优，也不表示信号层或组合层无效。")
    return lines + [""]


def cells_section(d):
    lines = ["## 四格（10 bps，开盘价，100 万）", "", "信息比率按 §9 口径；年化跟踪误差低于 2% 的一侧不计信息比率。", ""]
    rows = []
    for c in CELLS:
        row = [c]
        for m in ("lightgbm", "ridge"):
            v = d["cells"].get(f"{c}/{m}", {})
            row += [num(v.get("information_ratio"), 3), num(v.get("tracking_error"), 2, pct=True),
                    num(v.get("active_return_arith"), 2, pct=True)]
        rows.append(row)
    lines += table(["格", "LightGBM IR", "TE", "主动收益", "Ridge IR", "TE", "主动收益"], rows)
    lines += ["", "S4P2 与 S2P4 只报告，不进 2025，不能写成“信号层有效”或“组合层有效”。", "",
              "归因（信息比率差，只报告；S4P4 − S4P2 是整包差别，不能写成均值方差、κ 或冲击模型的效果）：", ""]
    keys = ("S4P4-S2P4", "S4P2-S2P2", "S2P4-S2P2", "S4P4-S4P2_package")
    lines += table(["模型"] + list(keys), [[m] + [num(d["attribution"][m].get(k), 3) for k in keys]
                                         for m in ("lightgbm", "ridge")])
    pbo = d.get("pbo", {})
    lines += ["", f"PBO（CSCV，{pbo.get('segments')} 段，{pbo.get('strategies')} 条日主动收益曲线，只作描述，"
              f"不能写成“PBO 低所以没有过拟合”）：{num(pbo.get('pbo'), 3)}。"]
    if "deflated_sharpe" in d:
        ds = d["deflated_sharpe"]
        lines.append(f"冻结配置的 Deflated Sharpe：{num(ds['deflated_sharpe'], 3)}（试验数 {ds['trials']}，"
                     f"冻结配置日净收益的夏普 {num(ds['sharpe_annual'], 2)}，N 次无技能试验的期望最大夏普 {num(ds['expected_max_sharpe_annual'], 2)}；"
                     "这里的采纳不是取最大，不能当作已惩罚选择偏差）。")
    return lines + [""]


def frontier_section(d):
    lines = []
    if d.get("kappa_frontier"):
        lines += ["## κ 前沿（S4P4，10 bps，只报告，不得据此更换 κ）", ""]
        rows = sorted(d["kappa_frontier"], key=lambda r: (r["model"], r["kappa_scale"]))
        lines += table(["模型", "κ 倍数", "日均双边换手", "IR", "TE"],
                       [[r["model"], num(r["kappa_scale"], 2), num(r["turnover"], 2, pct=True),
                         num(r["information_ratio"], 3), num(r["tracking_error"], 2, pct=True)] for r in rows]) + [""]
    if d.get("capacity_table"):
        lines += ["## 容量（VWAP 头 30 分钟成交、平方根冲击、零滑点）", ""]
        lines += table(["格", "模型", "初始资金", "IR", "TE", "hold_fallback"],
                       [[r["cell"], r["model"], f"{r['capital']:.0e}", num(r["information_ratio"], 3),
                         num(r["tracking_error"], 2, pct=True), num(r["fallback"], 1, pct=True)]
                        for r in d["capacity_table"]])
        lines += ["", "容量（第一档不可行或信息比率不高于 1e6 档一半的资金，不插值）："]
        lines += [f"- {k}：{v if isinstance(v, str) else f'{v:.0e}'}" for k, v in d["capacity"].items()] + [""]
    return lines


def extras_section(d, top=15):
    """Report-only items of section 9 that compare_v04 computes besides the decision."""
    lines = []
    if "matrix" in d:
        lines += ["## 全矩阵的净夏普差（10 bps，只报告）", ""]
        for label, key in (("信号层 S4 − S2（同模型、同政策）", "signal_layer"), ("组合层 P4 − P2-cost（同信号、同模型）", "portfolio_layer")):
            rows = [r for r in d["matrix"][key] if r["scenario"] == "slip_10bps"]
            lines += [label, ""] + table(["对比", "模型", "政策", "净夏普差", "信息比率差"],
                                         [[r["pair"], r["model"], r.get("policy", "—"), num(r["sharpe"], 3),
                                           num(r["information_ratio"], 3)] for r in rows]) + [""]
    if "costs_and_years" in d:
        years = sorted({y for v in d["costs_and_years"].values() for y in v["information_ratio_by_year"]})
        lines += ["## 换手、成本拖累与逐年信息比率（只报告）", ""]
        lines += table(["曲线", "日均双边换手", "年化成本拖累"] + [str(y) for y in years],
                       [[k, num(v["mean_two_way_turnover"], 2, pct=True), num(v["annual_cost_drag"], 2, pct=True)]
                        + [num(v["information_ratio_by_year"].get(y, v["information_ratio_by_year"].get(str(y))), 2) for y in years]
                        for k, v in d["costs_and_years"].items()]) + [""]
    if "selection_frequency" in d:
        s = d["selection_frequency"]
        lines += ["## 因子入选频次（S4，各折训练窗内筛选）", "",
                  f"{s['folds']} 折，其中 {s['fallback_core_folds']} 折入选不足 5 个、退回 v0.2 的 10 个特征；"
                  f"各折通过 FDR 的因子数：{s['passed_fdr_by_fold']}。", ""]
        lines += table(["因子", "入选折数"], [[k, v] for k, v in list(s["selected_folds"].items())[:top]]) + [""]
    if "versus_v02" in d:
        lines += ["## S2P2（全 A，U4）对 v0.2 主板运行（只报告，股票池扩展的描述，不是第三层的干净归因）", ""]
        lines += table(["模型", "年化收益差", "95% 区间", "p（bootstrap）", "交易日"],
                       [[m, num(v["annualized_mean_difference"], 2, pct=True),
                         f"{num(v['ci95_low'], 2, pct=True)} 至 {num(v['ci95_high'], 2, pct=True)}",
                         num(v["p_bootstrap"], 4), v["sessions"]] for m, v in d["versus_v02"].items()]) + [""]
    if "hedged" in d:
        lines += ["## 指数 beta 对冲后的收益（滚动 120 日 beta，只用此前数据；不含基差和对冲成本，只报告）", ""]
        for name, v in d["hedged"].items():
            if isinstance(v, str):
                lines.append(f"- {name}：{v}")
                continue
            lines += [f"**{name}**", ""] + table(["曲线", "平均 beta", "对冲后年化收益", "对冲后夏普"],
                                                [[k, num(x["mean_beta"], 2), num(x["hedged_annual_return"], 2, pct=True),
                                                  num(x["hedged_sharpe"], 2)] for k, x in v.items()]) + [""]
        lines.append("")
    return lines


def feasibility_section(f):
    lines = ["## 优化器可行性", ""]
    rows = [[k, v.get("rebalances"), num(v.get("hold_fallback_share"), 2, pct=True), num(v.get("widened_share", 0.0), 2, pct=True),
             num(v.get("infeasible_share", v.get("hold_fallback_share")), 2, pct=True),
             json.dumps(v.get("fallback_reasons"), ensure_ascii=False),
             num(v.get("solve_seconds_median"), 3), num(v.get("ex_ante_te_max"), 2, pct=True)]
            for k, v in f["portfolios"].items()]
    lines += table(["组合", "再平衡日", "hold_fallback", "放宽风险限额", "合计（对照门槛）", "原因", "求解秒数中位", "事前 TE 最大"], rows)
    return lines + ["", f"超过 {num(f['limit'], 0, pct=True)} 的 mvo 组合：{'有' if f.get('any_over_limit') else '无'}。", ""]


def bias_section(b):
    lines = ["## 风险模型偏差统计量（只报告，不据此改参数）", "",
             f"窗口 {b['window'][0]} 至 {b['window'][1]}；偏差统计量为标准化结果 r/σ 的标准差，预测准确时为 1。", ""]
    styles = {k: v for k, v in b["factors"].items() if not k.startswith("ind_")}
    rows = [[k, v["days"], num(v["bias"], 3), f"{num(v['band'][0], 3)}–{num(v['band'][1], 3)}" if v["band"] else "—",
             num(v["forecast_vol_annual"], 2, pct=True), num(v["realized_vol_annual"], 2, pct=True)]
            for k, v in styles.items() if v.get("bias") is not None]
    lines += table(["风格因子", "天数", "偏差", "约 95% 区间", "预测波动", "实现波动"], rows)
    # Styles without any exposure in the window (risk_bias.py) carry a note instead of a ratio.
    skipped = [k for k, v in styles.items() if v.get("bias") is None]
    if skipped:
        lines += ["", f"未估计（窗口内没有暴露）：{'、'.join(skipped)}。"]
    for label in ("random_total", "random_active"):
        r = b[label]
        lines.append("")
        lines.append(f"{'随机组合（总风险）' if label == 'random_total' else '随机组合（主动风险）'}：{r['portfolios']} 个，"
                     f"偏差中位数 {num(r['median_bias'], 3)}（10%–90% 分位 {num(r['p10_bias'], 3)}–{num(r['p90_bias'], 3)}），"
                     f"落在区间内的占 {num(r['share_inside_band'], 0, pct=True)}。")
    if b.get("strategies"):
        lines += ["", *table(["策略组合", "天数", "偏差", "预测 TE", "实现 TE"],
                             [[k, v["days"], num(v["bias"], 3), num(v["forecast_vol_annual"], 2, pct=True),
                               num(v["realized_vol_annual"], 2, pct=True)] for k, v in b["strategies"].items()])]
    return lines + [""]


def style_section(s):
    lines = ["## 风格归因（只报告）", ""]
    for name, p in s["portfolios"].items():
        styles = p["styles"]
        lines += [f"**{name}**（{p['days']} 天，平均事前 TE {num(p['mean_ex_ante_te'], 2, pct=True)}，"
                  f"池外持仓 {num(p['mean_outside_weight'], 1, pct=True)}，行业贡献年化 {num(p['industry_contribution_annual'], 2, pct=True)}）", ""]
        lines += table(["风格", "平均主动暴露", "年化贡献"], [[k, num(v["mean_active_exposure"], 3),
                                                     num(v["contribution_annual"], 2, pct=True)] for k, v in styles.items()]) + [""]
    return lines


def factor_section(f, top=15):
    factors = sorted(f["factors"], key=lambda x: -abs(x["nw_t"]) if np.isfinite(x["nw_t"]) else 0)
    lines = [f"## 单因子研究（{f['start']} 至 {f['end']}，{f['sessions']} 个交易日，只报告；入模筛选只在各折训练窗内做）", ""]
    lines += table(["因子", "族", "覆盖", "IC", "IC/sd", "NW t", "20 日后 IC", "多空夏普", "单调性", "5 日秩自相关"],
                   [[x["factor"], x["family"], num(x["coverage"], 0, pct=True), num(x["mean_ic"], 4), num(x["ic_ir"], 3),
                     num(x["nw_t"], 2), num(x["ic_decay"].get("20"), 4), num(x["long_short"]["sharpe_annual"], 2),
                     num(x["monotonicity"], 2), num(x["rank_autocorr_5"], 2)] for x in factors[:top]])
    families = {}
    for x in f["factors"]:
        families.setdefault(x["family"], []).append(x)
    lines += ["", *table(["族", "因子数", "|NW t| ≥ 2 的个数", "平均 |IC|"],
                         [[k, len(v), sum(abs(x["nw_t"]) >= 2 for x in v if np.isfinite(x["nw_t"])),
                           num(float(np.nanmean([abs(x["mean_ic"]) for x in v])), 4)] for k, v in sorted(families.items())])]
    lines += ["", f"平均截面秩相关 |ρ| ≥ 0.7 的因子簇：{len(f['clusters_at_0.7'])} 个，涉及 {f['clustered_factors']} 个因子。", ""]
    return lines


# RESEARCH section 9 (supplement, 2026-10-13): the handling written down before the gates were run.
GATE_HANDLING = {"gate2-valuation-timing": "按 §9 补充的既定处理，各配置 `valuation_fields: false`：去掉全部估值比率因子和价值、盈利收益两个风格",
                 "gate3-industry-history": "按 §9 补充的既定处理，`industry_snapshot: \"first\"`：全区间用第一期行业快照"}


def gates_section(gates):
    lines = ["## 数据闸门", ""]
    for path, g in gates:
        handling = GATE_HANDLING.get(Path(path).stem) if not g.get("passed") else None
        lines.append(f"- `{Path(path).name}`：{'通过' if g.get('passed') else '未通过'}" + (f"；{handling}" if handling else ""))
    return lines + [""]


def provenance_section(decision, freeze=None):
    """Runs, hashes and the freeze record behind every number above."""
    p = decision.get("provenance")
    if not p:
        return []
    short = lambda h: (h or "—")[:12]
    rows = [[c, Path(v["run"]).name, short(v["code_hash"]), short(v["config_hash"]), short(v["data_prefix_hash"]),
             short(v["prediction_hash"])] for c, v in p["cells"].items()]
    rows += [[f"κ 运行 {i + 1}", Path(v["run"]).name, short(v["code_hash"]), short(v["config_hash"]),
              short(v["data_prefix_hash"]), short(v["prediction_hash"])] for i, v in enumerate(p["kappa"])]
    rows += [[f"容量运行 {i + 1}", Path(v["run"]).name, short(v["code_hash"]), short(v["config_hash"]),
              short(v["data_prefix_hash"]), short(v["prediction_hash"])] for i, v in enumerate(p["capacity"])]
    lines = ["## 来源", "", *table(["运行", "目录", "代码哈希", "配置哈希", "数据前缀哈希", "预测哈希"], rows), ""]
    lines += [f"- 快照：`{p.get('snapshot') or '—'}`；v0.2 对照运行：`{p.get('v02_run') or '—'}`；试验登记：`{p.get('registry') or '—'}`。"]
    if freeze:
        lines += [f"- 冻结记录（{freeze['frozen_at']}）：配置 `{', '.join(freeze['config']['models'])}` × "
                  f"`{', '.join(freeze['config']['policies'])}`，代码 {short(freeze['code_hash'])}、协议代码 "
                  f"{short(freeze['protocol_code_hash'])}、配置 {short(freeze['config_hash'])}、开发前缀 "
                  f"{short(freeze['development_prefix_hash'])}，开发运行 `{Path(freeze['development_run']).name}`。"]
    return lines + [""]


def audits_section(audits):
    lines = ["## 账本审计", ""]
    for path, a in audits:
        lines.append(f"- `{Path(path).name}`：{'通过' if a.get('passed') else '未通过'}"
                     + (f"，违例 {a.get('violation_counts')}" if not a.get("passed") else ""))
    return lines + [""]


def picture(figures, name, caption):
    """A figure from scripts/report_figures.py, if it was drawn; every number in it is also in a table."""
    return [f"![{caption}]({figures[name]})", ""] if figures and name in figures else []


def render(decision, *, feasibility=None, risk_bias=None, style=None, factors=None, gates=(), audits=(), figures=None,
           freeze=None):
    lines = ["# v0.4 结果（按 RESEARCH §9 生成）", ""]
    if gates:
        lines += gates_section(gates)
    lines += decision_section(decision)
    lines += picture(figures, "active_cumulative.png", "八条主曲线相对 0.95 × 合格池等权的累计主动收益（10 bps）")
    lines += cells_section(decision)
    lines += picture(figures, "ir_by_year.png", "逐年信息比率（10 bps，不设跟踪误差门槛）")
    lines += frontier_section(decision)
    lines += picture(figures, "kappa_frontier.png", "κ 前沿：信息比率与换手")
    lines += picture(figures, "capacity.png", "容量：信息比率与初始资金")
    lines += extras_section(decision)
    if feasibility:
        lines += feasibility_section(feasibility)
    if audits:
        lines += audits_section(audits)
    if risk_bias:
        lines += bias_section(risk_bias)
    if style:
        lines += style_section(style)
    if factors:
        lines += factor_section(factors)
    lines += provenance_section(decision, freeze)
    return "\n".join(lines) + "\n"


def figure_links(directory, output):
    """PNG name -> path relative to the report, for every figure in `directory`."""
    if not directory:
        return None
    import os
    base = Path(output).resolve().parent
    return {p.name: Path(os.path.relpath(p.resolve(), base)).as_posix() for p in sorted(Path(directory).glob("*.png"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True)
    parser.add_argument("--feasibility")
    parser.add_argument("--risk-bias")
    parser.add_argument("--style")
    parser.add_argument("--factors")
    parser.add_argument("--gates", nargs="*", default=[])
    parser.add_argument("--audits", nargs="*", default=[])
    parser.add_argument("--figures", help="directory written by scripts/report_figures.py")
    parser.add_argument("--freeze", help="freeze record written after the decision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None
    text = render(load(args.decision), feasibility=load(args.feasibility), risk_bias=load(args.risk_bias),
                  style=load(args.style), factors=load(args.factors),
                  gates=[(p, load(p)) for p in args.gates], audits=[(p, load(p)) for p in args.audits],
                  figures=figure_links(args.figures, args.output), freeze=load(args.freeze))
    Path(args.output).write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
