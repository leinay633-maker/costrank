"""Figures for the v0.4 report (RESEARCH section 9, report only). Offline; needs the `report` extra.

usage: python scripts/report_figures.py --s2p2 RUN --s4p2 RUN --s2p4 RUN --s4p4 RUN
           [--kappa RUN ...] [--capacity RUN ...] --output-dir runs/audit/v04-figures

Static PNGs for the Markdown report, each backed by a table there: cumulative active return of
the eight main curves (10 bps, against 0.95 x the equal-weight eligible pool), information ratio
by year, the kappa frontier and the capacity grid. Information ratios here carry no 2% tracking
error gate (supplement 10). Palette: the first four categorical slots of the dataviz reference
palette, validated for adjacent pairs on the light surface; slots 3 and 4 are below 3:1 contrast,
so every series also has a direct label and the report keeps the table view.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from compare_v04 import CELLS, active, capacity, capacity_table, kappa_rows, ratio  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow, in this fixed order
COLOR = dict(zip(CELLS, SERIES))  # a cell keeps its color in every figure
MODELS = {"lightgbm": "LightGBM", "ridge": "Ridge"}
DASH = {"lightgbm": "-", "ridge": "--"}  # where models share a panel, line style tells them apart


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axhline(0, color=AXIS, linewidth=0.8)


def legend(fig, ax):
    """One legend for the whole figure, above the panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))


def figure(columns, width=11, height=4.2):
    fig, axes = plt.subplots(1, columns, figsize=(width, height), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0]


def end_labels(ax, ends):
    """Direct labels just right of each line's end, nudged apart; in ink, not the series color."""
    lo, hi = ax.get_ylim()
    gap, last = (hi - lo) * 0.05, None
    for name, x, y in sorted(ends, key=lambda e: e[2]):
        last = y if last is None else max(y, last + gap)
        ax.text(x, last, "  " + name, va="center", fontsize=8, color=INK2, clip_on=False)


def cumulative(runs, out):
    fig, axes = figure(2)
    for ax, model in zip(axes, MODELS):
        ends = []
        for cell, (_, policy) in CELLS.items():
            a, _ = active(runs[cell], model, policy)
            path = a.cumsum() * 100
            ax.plot(path.index, path.to_numpy(), color=COLOR[cell], linewidth=2, label=f"{cell} ({policy})")
            ends.append((cell, path.index[-1], float(path.iloc[-1])))
        style(ax, f"{MODELS[model]}: cumulative active return, % (10 bps)")
        ax.set_xlim(right=max(e[1] for e in ends) + (max(e[1] for e in ends) - path.index[0]) * 0.12)
        end_labels(ax, ends)
    legend(fig, axes[0])
    fig.savefig(out / "active_cumulative.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def by_year(runs, out):
    fig, axes = figure(2)
    for ax, model in zip(axes, MODELS):
        rows = {}
        for cell, (_, policy) in CELLS.items():
            a, _ = active(runs[cell], model, policy)
            rows[cell] = {int(y): ratio(part, 0)["information_ratio"] for y, part in a.groupby(a.index.year)}
        table = pd.DataFrame(rows)
        x = np.arange(len(table))
        width = 0.8 / len(CELLS)
        for i, cell in enumerate(CELLS):
            # The 0.92 factor leaves a surface gap between neighbouring bars.
            ax.bar(x + (i - (len(CELLS) - 1) / 2) * width, table[cell].to_numpy(), width * 0.92,
                   color=COLOR[cell], label=cell)
        ax.set_xticks(x, [str(y) for y in table.index])
        style(ax, f"{MODELS[model]}: information ratio by year (10 bps)")
    legend(fig, axes[0])
    fig.savefig(out / "ir_by_year.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def frontier(base, kappa, out):
    rows = pd.DataFrame(kappa_rows([base] + list(kappa)))
    fig, axes = figure(1, width=6.5, height=4.2)
    ax = axes[0]
    for model in MODELS:
        r = rows[rows.model == model].sort_values("kappa_scale")
        ax.plot(r.turnover * 100, r.information_ratio, color=COLOR["S4P4"], linestyle=DASH[model], linewidth=2,
                marker="o", markersize=5, label=f"S4P4 · {MODELS[model]}")
        for t, ir, k in zip(r.turnover * 100, r.information_ratio, r.kappa_scale):
            ax.annotate(f"κ×{k:g}", (t, ir), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    style(ax, "S4P4 kappa frontier: information ratio vs turnover (10 bps)")
    ax.set_xlabel("mean two-way turnover per session, %", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "kappa_frontier.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def capacity_figure(runs, out):
    table = capacity_table(runs)
    found = capacity(table)
    fig, axes = figure(1, width=6.5, height=4.2)
    ax = axes[0]
    for (cell, model), g in table.groupby(["cell", "model"]):
        key, color = f"{cell}/{model}", COLOR[f"{cell}P4"]
        ax.plot(g.capital, g.information_ratio, color=color, linestyle=DASH[model], linewidth=2,
                label=f"{cell}P4 · {MODELS[model]} (capacity {found[key] if isinstance(found[key], str) else f'{found[key]:.0e}'})")
        bad = g.fallback > 0.20
        ax.scatter(g.capital[~bad], g.information_ratio[~bad], color=color, s=36, zorder=3)
        # Hollow marker: more than 20% of rebalances could not meet the declared risk limits.
        ax.scatter(g.capital[bad], g.information_ratio[bad], facecolor=SURFACE, edgecolor=color, s=36, zorder=3)
    ax.set_xscale("log")
    style(ax, "Capacity: information ratio vs initial capital (VWAP fills, impact)")
    ax.set_xlabel("initial capital, CNY (hollow: infeasible on > 20% of rebalances)", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "capacity.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for cell in ("s2p2", "s4p2", "s2p4", "s4p4"):
        parser.add_argument(f"--{cell}", required=True)
    parser.add_argument("--kappa", nargs="*", default=[])
    parser.add_argument("--capacity", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = {"S2P2": args.s2p2, "S4P2": args.s4p2, "S2P4": args.s2p4, "S4P4": args.s4p4}
    cumulative(runs, out)
    by_year(runs, out)
    if args.kappa:
        frontier(runs["S4P4"], args.kappa, out)
    if args.capacity:
        capacity_figure(args.capacity, out)
    print("\n".join(str(p) for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
