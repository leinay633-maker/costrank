"""Single-factor research report of the registered library on a schema-3 snapshot (report only;
models select factors inside each training window, never from this report).

usage: python scripts/factor_report.py <snapshot> --config configs/v04-s4p2.json
           --start 2017-01-01 --end 2019-12-31 --output report.json

Per factor over sessions in [start, end] whose labels matured by `end`:
- information: coverage; daily rank IC against the industry- and size-neutral 5-day label (mean,
  standard deviation, mean / sd, Newey-West t with Bartlett lag 5, share of positive days, by year);
- decay: mean rank IC of the factor against that label 5, 10 and 20 sessions later;
- deciles: mean raw 5-day open-to-open return of each decile, monotonicity (Spearman of decile
  number and return), top-minus-bottom spread, and on non-overlapping 5-session steps the equal
  weight long-short gross return's annualized mean and Sharpe (no costs, no neutralization);
- turnover: 5-session rank autocorrelation and the share of the top decile replaced 5 sessions later;
- crowding proxies: top-decile mean turnover rate over the pool's (attention) and the log
  book-to-price spread between top and bottom deciles (valuation); window mean and the percentile
  of the latest 60-session mean among the window's rolling 60-session means.
Library-wide: each factor's three closest neighbours by the average cross-sectional rank
correlation (sessions sampled every 5) and the clusters at |corr| >= 0.7 (average linkage).
Before RESEARCH section 9's runs this is only run on sessions up to 2019.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from costrank.config import Config
from costrank.data import read_bundle
from costrank.factors import REGISTRY, market
from costrank.signals import daily_rank_ic, newey_west_t, prepare_v04, session_blocks

MIN_NAMES = 30  # three names a decile
LAGS = (5, 10, 20)
WINDOW = 60


def decile_codes(F, min_names=MIN_NAMES):
    """Decile 0..9 of each column by within-session rank; -1 where missing or too few names."""
    pct = pd.DataFrame(F).rank(pct=True, method="first").to_numpy()
    out = np.full(F.shape, -1, dtype=np.int8)
    ok = np.isfinite(pct) & (np.isfinite(F).sum(axis=0) >= min_names)[None, :]
    out[ok] = (np.ceil(pct[ok] * 10) - 1).clip(0, 9).astype(np.int8)
    return out


def leg_means(D, values, deciles):
    """Mean of `values` (one per name) over the names of each decile in `deciles`, per column."""
    finite = np.isfinite(values)
    v = np.where(finite, values, 0.0)[:, None]
    out = []
    for d in deciles:
        mask = (D == d) & finite[:, None]
        n = mask.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(n > 0, (mask * v).sum(axis=0) / n, np.nan))
    return np.array(out)


def rolling_percentile(series, window=WINDOW):
    s = pd.Series(series).dropna()
    if len(s) < window:
        return np.nan, np.nan
    r = s.rolling(window).mean().dropna()
    return float(r.iloc[-1]), float((r <= r.iloc[-1]).mean())


def correlation_summary(samples, names, start, end, threshold=0.7):
    picked = [m for d, m in samples.items() if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not picked:
        return {}, []
    C = np.nanmean(np.stack(picked).astype(float), axis=0)
    np.fill_diagonal(C, 1.0)
    C = np.where(np.isfinite(C), C, 0.0)
    near = {}
    for j, name in enumerate(names):
        order = [k for k in np.argsort(-np.abs(C[j])) if k != j][:3]
        near[name] = [[names[k], round(float(C[j, k]), 4)] for k in order]
    distance = np.clip(1 - np.abs((C + C.T) / 2), 0, None)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"), t=1 - threshold,
                      criterion="distance")
    clusters = [sorted(names[k] for k in np.flatnonzero(labels == c)) for c in np.unique(labels)]
    return near, sorted((c for c in clusters if len(c) > 1), key=len, reverse=True)


def report(bundle, config, start, end):
    panel, signals = prepare_v04(bundle, config, library=True)
    rows = panel.loc[signals.index]
    inside = ((rows.date >= start) & (rows.date <= end) & (rows.label_end <= end)).to_numpy()
    dates, symbols = signals.dates[inside], rows.symbol.to_numpy()[inside]
    X = signals.X[inside]
    target = rows.neutral_target.to_numpy()[inside]
    raw = rows.label.to_numpy(dtype=float)[inside]
    m = market(panel, bundle)
    di = m.c.index.get_indexer(pd.DatetimeIndex(dates))
    si = m.c.columns.get_indexer(symbols)
    turn = m.turn.to_numpy()[di, si]
    with np.errstate(invalid="ignore", divide="ignore"):
        pb = m.pb.to_numpy()[di, si]
        log_bp = np.where(pb > 0, -np.log(pb), np.nan)
    ic = signals.ic.loc[(signals.ic.index >= start) & (signals.ic.index <= end)]
    ic = ic.loc[ic.index.isin(np.unique(dates))]
    t = newey_west_t(ic.to_numpy())
    sessions, blocks = session_blocks(dates)
    k = X.shape[1]
    returns = np.full((len(sessions), 10, k), np.nan)
    attention = np.full((len(sessions), k), np.nan)
    valuation = np.full((len(sessions), k), np.nan)
    replaced, autocorr, previous = [], [], None
    for s, idx in enumerate(blocks):
        D = decile_codes(X[idx])
        returns[s] = leg_means(D, raw[idx], range(10))
        pool_turn = np.nanmean(turn[idx]) if np.isfinite(turn[idx]).any() else np.nan
        attention[s] = leg_means(D, turn[idx], [9])[0] / pool_turn
        top, bottom = leg_means(D, log_bp[idx], [9, 0])
        valuation[s] = top - bottom
        if s % 5 == 0:
            frame = pd.DataFrame(X[idx], index=symbols[idx])
            codes = pd.DataFrame(D, index=symbols[idx])
            if previous is not None:
                common = frame.index.intersection(previous[0].index)
                a, b = previous[0].loc[common].rank(), frame.loc[common].rank()
                autocorr.append(a.corrwith(b).to_numpy())
                was, now = previous[1].loc[common].to_numpy() == 9, codes.loc[common].to_numpy() == 9
                with np.errstate(invalid="ignore", divide="ignore"):
                    replaced.append(1 - (was & now).sum(axis=0) / was.sum(axis=0))
            previous = (frame, codes)
    nan_k = np.full(k, np.nan)
    autocorr = np.nanmean(np.array(autocorr), axis=0) if autocorr else nan_k
    replaced = np.nanmean(np.array(replaced), axis=0) if replaced else nan_k
    # Decay: the factor at a session against the neutral label `lag` sessions later (same names).
    position = pd.Index(sessions).get_indexer(pd.DatetimeIndex(dates))
    keyed = pd.Series(target, index=pd.MultiIndex.from_arrays([position, symbols]))
    sampled = position % 5 == 0
    decay = {}
    for lag in LAGS:
        later = keyed.reindex(pd.MultiIndex.from_arrays([position[sampled] + lag, symbols[sampled]])).to_numpy()
        decay[lag] = np.nanmean(daily_rank_ic(X[sampled], later, dates[sampled]).to_numpy(), axis=0)
    long_short = returns[:, 9, :] - returns[:, 0, :]
    step = long_short[::5]
    years = ic.groupby(ic.index.year).mean()
    near, clusters = correlation_summary(signals.correlations, signals.names, start, end)
    factors = []
    for j, name in enumerate(signals.names):
        curve = np.nanmean(returns[:, :, j], axis=0)
        ls = step[:, j][np.isfinite(step[:, j])]
        sd = ls.std(ddof=1) if len(ls) > 1 else np.nan
        mono = stats.spearmanr(np.arange(10), curve).statistic if np.isfinite(curve).all() else np.nan
        att_mean, att_pct = rolling_percentile(attention[:, j])
        val_mean, val_pct = rolling_percentile(valuation[:, j])
        mean_ic, sd_ic = float(ic[name].mean()), float(ic[name].std())
        factors.append({
            "factor": name, "family": REGISTRY[name][0] if name in REGISTRY else "core",
            "coverage": float(np.isfinite(X[:, j]).mean()), "mean_ic": mean_ic, "ic_std": sd_ic,
            "ic_ir": mean_ic / sd_ic if sd_ic > 0 else np.nan, "nw_t": float(t[j]),
            "positive_share": float((ic[name] > 0).mean()),
            "ic_by_year": {int(y): float(v) for y, v in years[name].items()},
            "ic_decay": {str(lag): float(decay[lag][j]) for lag in LAGS},
            "decile_returns_5d": [float(x) for x in curve], "monotonicity": float(mono),
            "decile_spread_5d": float(curve[9] - curve[0]),
            "long_short": {"periods": int(len(ls)), "annual_return": float(ls.mean() * 252 / 5) if len(ls) else np.nan,
                           "sharpe_annual": float(ls.mean() / sd * np.sqrt(252 / 5)) if sd and sd > 0 else np.nan},
            "rank_autocorr_5": float(autocorr[j]), "top_decile_replaced_5": float(replaced[j]),
            "crowding": {"attention_mean": float(np.nanmean(attention[:, j])), "attention_latest": att_mean,
                         "attention_latest_pct": att_pct, "value_spread_mean": float(np.nanmean(valuation[:, j])),
                         "value_spread_latest": val_mean, "value_spread_latest_pct": val_pct},
            "nearest": near.get(name, [])})
    return {"start": start, "end": end, "sessions": int(len(ic)), "factors": factors,
            "clusters_at_0.7": clusters, "clustered_factors": int(sum(len(c) for c in clusters))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config.load(args.config)
    bundle = read_bundle(args.snapshot, allow_after=config.holdout_start)
    out = dict(snapshot=str(Path(args.snapshot).resolve()), **report(bundle, config, args.start, args.end))
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame([{**{k: f[k] for k in ("factor", "family", "coverage", "mean_ic", "nw_t", "monotonicity",
                                                  "rank_autocorr_5")},
                           "ls_sharpe": f["long_short"]["sharpe_annual"], "ic_lag20": f["ic_decay"]["20"]}
                          for f in out["factors"]]).set_index("factor")
    print(table.sort_values("nw_t", key=abs, ascending=False).head(30).round(4).to_string())
    print("clusters at |corr| >= 0.7:", len(out["clusters_at_0.7"]), "covering", out["clustered_factors"], "factors")


if __name__ == "__main__":
    main()
