"""Vectorized research backtest of the top-k policy, reconciled against the event ledger of a
completed run (engineering check; never used for selection or the section 9 decision).

usage: python scripts/fast_backtest.py <run_dir> --model ridge --output report.json

The vectorized version holds the equal-weight top-k names chosen at each decision close from
the next open to the open after the next decision, with proportional costs on two-way weight
turnover (commission + transfer + slippage on both sides, stamp duty on sells), and ignores lots,
cash, minimum commission, price limits, suspensions and participation. Marks are open to open,
the ledger's are close to close, so the two are compared over the whole period and per year, and
the gap is itemized: order rejections and partial fills, invested share, and cost differences.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import read_bundle
from costrank.features import prepare


def vector_topk(predictions, panel, config, slippage_bps):
    opens = panel.pivot(index="date", columns="symbol", values="tr_open")
    nxt = opens.shift(-1) / opens - 1  # open t to open t+1, total-return index
    days = sorted(predictions.date.unique())
    weights, rows = pd.Series(dtype=float), []
    sessions = opens.index[(opens.index >= days[0]) & (opens.index <= config.dev_end)]
    scores = {d: g.set_index("symbol").score for d, g in predictions.groupby("date")}
    for i, day in enumerate(sessions[:-1]):
        turnover, cost = 0.0, 0.0
        if i % config.rebalance == 0 and day in scores:
            top = scores[day].sort_values(ascending=False, kind="stable").head(config.topk).index
            target = pd.Series(config.invested / len(top), index=top)
            diff = target.reindex(target.index.union(weights.index)).fillna(0) - weights.reindex(
                target.index.union(weights.index)).fillna(0)
            stamp = 0.001 if day < pd.Timestamp("2023-08-28") else 0.0005
            transfer = 0.00002 if day < pd.Timestamp("2022-04-29") else 0.00001
            side = config.commission + transfer + slippage_bps / 10_000
            cost = float(diff.abs().sum() * side + diff.clip(upper=0).abs().sum() * stamp)
            turnover, weights = float(diff.abs().sum()), target
        # Positions decided at close t are entered at the open of t+1 and earn open t+1 -> open t+2.
        held = nxt.loc[sessions[i + 1]] if i + 1 < len(sessions) else None
        r = float((weights * held.reindex(weights.index).fillna(0)).sum()) if held is not None else 0.0
        rows.append(dict(date=sessions[i + 1], gross=r, cost=cost, turnover=turnover))
    out = pd.DataFrame(rows).set_index("date")
    out["net"] = out.gross - out.cost
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--scenario", default="slip_10bps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in manifest["config"].items()})
    bundle = read_bundle(manifest["data_path"], allow_after=config.holdout_start)
    panel = prepare(bundle, config)
    predictions = pd.read_csv(run / "predictions.csv.gz", parse_dates=["date"])
    predictions = predictions[predictions.model == args.model]
    bps = 0.0 if args.scenario == "zero_all_costs" else float(args.scenario.split("_")[1].replace("bps", ""))
    fast = vector_topk(predictions, panel, config, bps)
    name = f"{args.model}__topk__{args.scenario}"
    daily = pd.read_csv(run / name / "daily.csv.gz", parse_dates=["date"]).set_index("date")
    orders = pd.read_csv(run / name / "orders.csv.gz")
    common = fast.index.intersection(daily.index)
    annual = lambda r: float(np.prod(1 + r) ** (252 / len(r)) - 1)
    report = {"run": str(run.resolve()), "portfolio": name,
              "vector_annual_net": annual(fast.net.loc[common]), "ledger_annual_net": annual(daily.net_return.loc[common]),
              "vector_annual_gross": annual(fast.gross.loc[common]),
              "ledger_annual_gross_same_trades": annual(daily.gross_same_trades_return.loc[common]),
              "daily_correlation": float(fast.net.loc[common].corr(daily.net_return.loc[common])),
              "by_year": {int(y): {"vector": annual(fast.net.loc[g]), "ledger": annual(daily.net_return.loc[g])}
                          for y, g in pd.Series(common, index=common).groupby(common.year)},
              "gap_items": {"orders_rejected_or_partial_share": float(orders.status.ne("filled").mean()),
                            "rejection_reasons": orders.loc[orders.status.ne("filled"), "reason"].value_counts().to_dict(),
                            "ledger_mean_exposure": float(daily.exposure.mean()), "vector_invested": config.invested,
                            "vector_mean_daily_cost": float(fast.cost.mean()),
                            "ledger_mean_daily_cost_share": float((daily[["commission", "stamp", "transfer", "slippage"]]
                                                                   .sum(axis=1) / daily.nav.shift().fillna(config.initial_cash)).mean()),
                            "vector_mean_turnover": float(fast.turnover.mean()),
                            "ledger_mean_turnover": float(daily.turnover_two_way.mean())}}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
