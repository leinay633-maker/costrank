"""Bias statistics of the v0.4 risk model (RESEARCH section 9: report only; never used to change
its parameters).

usage: python scripts/risk_bias.py <schema3_snapshot> --output report.json [--config configs/v04-s4p4.json]
           [--start 2020-01-01 --end 2024-12-31] [--runs RUN ...] [--scenario slip_10bps]
           [--random 100 --names 50 --seed 0]

A portfolio held from close t to close t+1 with daily risk s_t forecast by the close-t snapshot
has the standardized outcome b_t = r_{t+1} / s_t. The bias statistic is the standard deviation of
b over the window: 1 when the forecast is right, inside 1 +- sqrt(2 / T) about 95% of the time
for T independent days; above the band the model understates risk. Three families:
- pure factor portfolios: the realized factor return over the forecast factor volatility;
- random portfolios: `names` equal-weight eligible names chosen by fixed random keys (so the
  holdings persist), in total and active (minus the equal-weight eligible pool) form;
- strategy portfolios of finished runs: end-of-day holdings, active against `invested` x the
  equal-weight pool, holdings outside the pool with their own exposures (section 9 supplement 7),
  as scripts/style_report.py attributes them.
Returns are close-to-close total returns from the snapshot; a name without one counts as flat.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from costrank.config import Config
from costrank.data import read_bundle
from costrank.factors import market
from costrank.features import prepare
from costrank.risk import RiskModel


def bias(b):
    b = pd.Series(b, dtype=float).dropna()
    t = len(b)
    if t < 2:
        return {"days": t, "bias": None, "band": None}
    half = float(np.sqrt(2 / t))
    return {"days": t, "bias": float(b.std(ddof=1)), "band": [1 - half, 1 + half]}


def summarize(forecast, realized):
    """forecast, realized: date-indexed daily risk forecasts and next-day returns of one portfolio;
    only days with a positive forecast count."""
    ok = forecast.notna() & (forecast > 0) & realized.notna()
    forecast, realized = forecast[ok], realized[ok]
    b = realized / forecast
    out = bias(b)
    out["forecast_vol_annual"] = float(forecast.mean() * np.sqrt(252))
    out["realized_vol_annual"] = float(realized.std(ddof=1) * np.sqrt(252))
    out["by_year"] = {str(y): bias(g)["bias"] for y, g in b.groupby(b.index.year)}
    return out


def spread(forecast, realized):
    """Many portfolios (columns): the distribution of their bias statistics."""
    stats = (realized / forecast).std(ddof=1)
    half = float(np.sqrt(2 / len(forecast)))
    by_year = {str(y): float((realized.loc[g] / forecast.loc[g]).std(ddof=1).median())
               for y, g in forecast.groupby(forecast.index.year).groups.items()}
    return {"portfolios": int(forecast.shape[1]), "days": int(len(forecast)), "band": [1 - half, 1 + half],
            "median_bias": float(stats.median()), "p10_bias": float(stats.quantile(0.1)),
            "p90_bias": float(stats.quantile(0.9)), "share_inside_band": float(stats.between(1 - half, 1 + half).mean()),
            "median_forecast_vol_annual": float((forecast.mean() * np.sqrt(252)).median()),
            "median_realized_vol_annual": float((realized.std(ddof=1) * np.sqrt(252)).median()),
            "median_bias_by_year": by_year}


def load_holdings(runs, scenario):
    """{portfolio name: (invested, {date: weights by symbol})} for every model and policy."""
    out = {}
    for run in runs:
        run = Path(run)
        invested = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["config"]["invested"]
        for path in sorted(run.glob(f"*__*__{scenario}")):
            h = pd.read_csv(path / "holdings.csv.gz", parse_dates=["date"])
            out[f"{run.name}/{path.name}"] = (invested, {d: g.set_index("symbol").weight for d, g in h.groupby("date")})
    return out


def collect(bundle, config, start, end, *, holdings=None, n_random=100, n_names=50, seed=0):
    panel = prepare(bundle, config)
    m = market(panel, bundle)
    after = m.ret.shift(-1)  # close t to close t+1
    last = m.ret.index[-1]
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    keys = pd.DataFrame(np.random.default_rng(seed).random((m.ret.shape[1], n_random)), index=m.ret.columns)
    holdings = holdings or {}
    factor_sd, rand, strat = {}, {"total": {}, "active": {}}, {k: {} for k in holdings}

    def visit(day, snap):
        if not lo <= day <= hi or day == last or len(snap.names) < n_names:
            return
        X, F, d = snap.exposures.to_numpy(), snap.covariance.to_numpy(), snap.specific.to_numpy()
        r = after.loc[day].reindex(snap.names).fillna(0.0).to_numpy()
        factor_sd[day] = np.sqrt(np.diag(F))
        n = len(snap.names)
        pick = np.argsort(keys.reindex(snap.names).to_numpy(), axis=0, kind="stable")[:n_names]
        W = np.zeros((n, n_random))
        W[pick, np.arange(n_random)] = 1.0 / n_names
        for label, P in (("total", W), ("active", W - 1.0 / n)):
            Y = X.T @ P
            var = np.einsum("kp,kl,lp->p", Y, F, Y) + (d[:, None] * P ** 2).sum(axis=0)
            rand[label][day] = (np.sqrt(np.maximum(var, 0.0)), P.T @ r)
        for name, (invested, book) in holdings.items():
            if day not in book:
                continue
            a = snap.active(book[day], invested)  # holdings outside the pool keep their own exposures
            realized = after.loc[day].reindex(a.index).fillna(0.0).to_numpy()
            strat[name][day] = (float(np.sqrt(max(snap.daily_variance(a), 0.0))), float(a.to_numpy() @ realized))

    risk = RiskModel(panel, m, bundle, config, visit=visit)
    days = sorted(factor_sd)
    if not days:
        raise ValueError("No risk-model snapshot inside the window")
    sessions = m.ret.index
    following = {day: sessions[sessions.get_loc(day) + 1] for day in days}
    sd = pd.DataFrame([factor_sd[day] for day in days], index=pd.DatetimeIndex(days), columns=risk.factors)
    realized = pd.DataFrame([risk.factor_returns.get(following[day], np.full(len(risk.factors), np.nan))
                             for day in days], index=sd.index, columns=risk.factors)
    estimated = sd.where(sd > 1e-8)
    report = {"window": [str(lo.date()), str(hi.date())], "random_names": n_names, "seed": seed,
              # A style without exposure that day (value and earnings yield with valuation fields off;
              # a 252-day descriptor before its history) has no factor return to forecast: the ratio
              # would be rounding noise over rounding noise, so such days do not count.
              "factors": {k: summarize(estimated[k], realized[k]) if estimated[k].notna().any() else
                          {"days": 0, "bias": None, "note": "no exposure in the window; not estimated"}
                          for k in risk.factors}}
    for label, rows in rand.items():
        idx = pd.DatetimeIndex(sorted(rows))
        report[f"random_{label}"] = spread(pd.DataFrame([rows[d][0] for d in idx], index=idx),
                                           pd.DataFrame([rows[d][1] for d in idx], index=idx))
    report["strategies"] = {}
    for name, rows in strat.items():
        if len(rows) >= 2:
            frame = pd.DataFrame(rows, index=["forecast", "realized"]).T.sort_index()
            report["strategies"][name] = summarize(frame.forecast, frame.realized)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--config", default="configs/v04-s4p4.json")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--scenario", default="slip_10bps")
    parser.add_argument("--random", type=int, default=100)
    parser.add_argument("--names", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config.load(args.config)
    bundle = read_bundle(args.snapshot, allow_after=config.holdout_start)
    report = collect(bundle, config, args.start or config.dev_start, args.end or config.dev_end,
                     holdings=load_holdings(args.runs, args.scenario), n_random=args.random,
                     n_names=args.names, seed=args.seed)
    report.update(snapshot=str(Path(args.snapshot).resolve()), config=args.config, runs=args.runs,
                  scenario=args.scenario)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    styles = {k: round(v["bias"], 3) for k, v in report["factors"].items() if not k.startswith("ind_") and v["bias"]}
    print(json.dumps({"style_factor_bias": styles, "random_total_median": report["random_total"]["median_bias"],
                      "random_active_median": report["random_active"]["median_bias"]}, indent=2))


if __name__ == "__main__":
    main()
