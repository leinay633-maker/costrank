"""Trial registry for the deflated Sharpe ratio (RESEARCH section 9): every configuration in the
project that produced, or was set up to produce, a backtest Sharpe ratio. Offline.

usage: python scripts/trial_registry.py <runs_root> [<runs_root> ...] --output registry.csv

Each scenario directory with a daily ledger is one trial with its daily net-return Sharpe; a run
that failed before writing ledgers counts its planned scenarios as trials without a Sharpe.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def scan(roots):
    rows = []
    for root in roots:
        for manifest in sorted(Path(root).rglob("manifest.json")):
            run = manifest.parent
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ledgers = sorted(run.glob("*/daily.csv.gz"))
            base = dict(run=str(run), status=record.get("status"), phase=record.get("phase"),
                        git=(record.get("git") or {}).get("sha"), config_hash=record.get("config_hash"))
            for path in ledgers:
                model, policy, scenario = (path.parent.name.split("__") + [None, None])[:3]
                r = pd.read_csv(path, usecols=["date", "net_return"]).net_return.to_numpy(float)
                sd = r.std(ddof=1) if len(r) > 1 else np.nan
                rows.append(dict(base, model=model, policy=policy, scenario=scenario, sessions=len(r),
                                 sharpe_daily=float(r.mean() / sd) if sd > 0 else np.nan))
            # Planned scenarios a failed run never wrote still count as trials, without a Sharpe.
            cfg = record.get("config", {})
            planned = (len(cfg.get("models", [])) * len(cfg.get("policies", []))
                       * (1 + len(cfg.get("slippage_grid_bps", []))))
            rows += [dict(base, model=None, policy=None, scenario=None, sessions=0, sharpe_daily=np.nan)
                     for _ in range(max(planned - len(ledgers), 0))]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    table = scan(args.roots)
    table.to_csv(args.output, index=False)
    sharpe = table.sharpe_daily.dropna()
    print(json.dumps({"trials": len(table), "with_sharpe": int(len(sharpe)),
                      "sharpe_daily_variance": float(sharpe.var(ddof=1)) if len(sharpe) > 1 else None,
                      "runs": int(table.run.nunique()) if len(table) else 0}, indent=2))


if __name__ == "__main__":
    main()
