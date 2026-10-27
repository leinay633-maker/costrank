"""v0.4 signal pipeline (RESEARCH section 9): factor processing, the neutral label and factor
selection inside each training window. Nothing here looks past a row's own session except the
label, which the folds purge by its maturity date."""
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from .factors import REGISTRY, market, raw_factors
from .features import FEATURES, prepare

MAD_SCALE = 1.4826
CORE = ["core_" + x for x in FEATURES]  # v0.2 features, processed like the library; fallback only


def static_industries(bundle, start, minimum=20):
    """Industry mapping fixed by the first snapshot on or before `start`: CSRC letter, or letter
    plus two digits in manufacturing; groups with fewer than `minimum` stocks in that snapshot
    merge into 'C' (manufacturing) or 'other'. Codes first seen later map to their letter if it is
    a group, else to 'other'."""
    ind = bundle.tables["industries"]
    first = ind.date[ind.date <= pd.Timestamp(start)].max() if (ind.date <= pd.Timestamp(start)).any() else ind.date.min()
    snap = ind[ind.date == first]
    code = lambda c: (c[:3] if c[:1] == "C" else c[:1]) if c[:1].isalpha() else "?"
    counts = snap.industry.map(code).value_counts()
    groups = {k for k, v in counts.items() if v >= minimum and k != "?"}

    def group(industry):
        c = code(industry or "")
        if c in groups:
            return c
        if c.startswith("C"):
            return "C"
        return c[:1] if c[:1] in groups else "other"
    return group


def industry_series(bundle, rows, mapping, mode="monthly"):
    """`mode` "first" (RESEARCH section 9, gate 3 failed) uses the first snapshot for every session."""
    ind = bundle.tables["industries"]
    snaps = np.sort(ind.date.unique())
    if mode == "first":
        ind, snaps = ind[ind.date == snaps[0]], snaps[:1]
    pos = np.searchsorted(snaps, rows.date.to_numpy(), side="right") - 1
    if mode == "first":
        pos = np.zeros(len(rows), dtype=int)
    lookup = pd.Series(ind.industry.to_numpy(), index=pd.MultiIndex.from_arrays([ind.date, ind.symbol]))
    code = lookup.reindex(pd.MultiIndex.from_arrays([snaps[pos.clip(min=0)], rows.symbol])).fillna("").to_numpy(dtype=str)
    code = np.where(pos < 0, "", code)
    return pd.Series([mapping(c) for c in code], index=rows.index)


def industry_groups(bundle, rows, mode="monthly"):
    """Industry of each eligible row under the mapping fixed by the first snapshot of the panel."""
    return industry_series(bundle, rows, static_industries(bundle, rows.date.min()), mode)


def zscore(F):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(F, axis=0), np.nanstd(F, axis=0)
    return (F - mu) / np.where(sd > 0, sd, np.nan)


def session_blocks(dates):
    codes, uniques = pd.factorize(pd.Series(dates), sort=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=len(uniques)))]
    return uniques, [order[bounds[k]:bounds[k + 1]] for k in range(len(uniques))]


def standardize(values, dates, groups, log_size, *, mad=5.0):
    """Per session: winsorize at median +- mad * 1.4826 * MAD, z-score, regress on industry
    dummies and z-scored log float cap using only each column's non-missing names, z-score the
    residual. Missing values stay missing; rows without a float cap are left missing."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    _, blocks = session_blocks(dates)
    groups, log_size = np.asarray(groups), np.asarray(log_size, dtype=float)
    for idx in blocks:
        F = values[idx].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(F, axis=0)
            dev = np.nanmedian(np.abs(F - med), axis=0) * MAD_SCALE
        clip = dev > 0  # flags and other columns with a zero MAD are not winsorized
        F[:, clip] = np.clip(F[:, clip], (med - mad * dev)[clip], (med + mad * dev)[clip])
        Z = zscore(F)
        sized = np.isfinite(log_size[idx])
        dummies = pd.get_dummies(groups[idx]).to_numpy(dtype=float)
        size = np.where(sized, log_size[idx], 0.0)
        size = (size - size[sized].mean()) / (size[sized].std() or 1.0) if sized.any() else size
        X = np.column_stack([dummies, size])
        valid = np.isfinite(Z) & sized[:, None]
        R = np.full(Z.shape, np.nan)
        full = valid.all(axis=0)
        if full.any():
            beta = np.linalg.lstsq(X, Z[:, full], rcond=None)[0]
            R[:, full] = Z[:, full] - X @ beta
        for j in np.flatnonzero(~full & (valid.sum(axis=0) > X.shape[1])):
            v = valid[:, j]
            beta = np.linalg.lstsq(X[v], Z[v, j], rcond=None)[0]
            R[v, j] = Z[v, j] - X[v] @ beta
        out[idx] = zscore(R)
    return out


def cross_rank(values, dates):
    """Within-session percentile rank minus 0.5, missing kept missing."""
    out = np.full(len(values), np.nan)
    _, blocks = session_blocks(dates)
    for idx in blocks:
        v = values[idx]
        ok = np.isfinite(v)
        if ok.any():
            out[idx[ok]] = stats.rankdata(v[ok]) / ok.sum() - 0.5
    return out


def daily_rank_ic(X, target, dates):
    """Spearman per session between each column and the target, over names where both exist."""
    sessions, blocks = session_blocks(dates)
    ic = np.full((len(sessions), X.shape[1]), np.nan)
    for k, idx in enumerate(blocks):
        y = target[idx]
        keep = np.isfinite(y)
        if keep.sum() < 20:
            continue
        F, y = X[idx][keep].astype(float), y[keep]
        M = np.isfinite(F)
        full = M.all(axis=0)
        if full.any():
            R, ry = stats.rankdata(F[:, full], axis=0), stats.rankdata(y)
            R, ry = R - R.mean(axis=0), ry - ry.mean()
            with np.errstate(invalid="ignore", divide="ignore"):
                ic[k, full] = (R.T @ ry) / np.sqrt((R ** 2).sum(axis=0) * (ry @ ry))
        for j in np.flatnonzero(~full & (M.sum(axis=0) >= 20)):
            v = M[:, j]
            ic[k, j] = stats.spearmanr(F[v, j], y[v]).statistic
    return pd.DataFrame(ic, index=pd.DatetimeIndex(sessions))


def correlation_samples(X, dates, every=5):
    """Cross-sectional rank correlation matrices on every `every`-th session (redundancy pruning)."""
    sessions, blocks = session_blocks(dates)
    out = {}
    for k in range(0, len(sessions), every):
        F = X[blocks[k]].astype(float)
        R = stats.rankdata(np.where(np.isfinite(F), F, np.nan), axis=0, nan_policy="omit")
        R = np.where(np.isfinite(R), R - np.nanmean(R, axis=0), 0.0)
        norm = np.sqrt((R ** 2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[pd.Timestamp(sessions[k])] = ((R.T @ R) / np.outer(norm, norm)).astype(np.float32)
    return out


def newey_west_t(x, lag=5):
    """t statistic of the mean of each column with a Bartlett-kernel HAC variance; each column
    uses its own sessions with a defined IC, in order."""
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        v = x[np.isfinite(x[:, j]), j]
        T = len(v)
        if T < 2 * lag + 2:
            continue
        d = v - v.mean()
        var = (d @ d) / T
        for l in range(1, lag + 1):
            var += 2 * (1 - l / (lag + 1)) * (d[l:] @ d[:-l]) / T
        if var > 0:
            out[j] = v.mean() / np.sqrt(var / T)
    return out


def benjamini_hochberg(p, q):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = np.isfinite(p).sum()
    passed = np.zeros(len(p), bool)
    if m == 0:
        return passed
    ranked = p[order]
    below = np.flatnonzero(ranked <= q * np.arange(1, len(p) + 1) / m)
    if len(below):
        passed[order[:below.max() + 1]] = True
    return passed & np.isfinite(p)


@dataclass
class SignalSet:
    """Processed factor matrix on the eligible rows of the panel (row order = panel.loc[eligible])."""
    index: pd.Index
    dates: np.ndarray
    names: list
    X: np.ndarray
    ic: pd.DataFrame
    correlations: dict
    groups: pd.Series
    log_size: np.ndarray
    info: dict = field(default_factory=dict)


def select(signals, train_mask, train_dates, config, library):
    """In-window selection: coverage, Newey-West t of daily rank IC (Bartlett, lag 5), two-sided
    p values, Benjamini-Hochberg at q, then greedy pruning by |t| on sign-aligned correlation."""
    cols = [signals.names.index(n) for n in library]
    coverage = np.isfinite(signals.X[train_mask][:, cols]).mean(axis=0)
    ic = signals.ic.loc[signals.ic.index.isin(train_dates)].to_numpy()[:, cols]
    t = newey_west_t(ic)
    p = 2 * stats.norm.sf(np.abs(t))
    p = np.where(coverage >= config.selection_coverage, p, np.nan)
    passed = benjamini_hochberg(p, config.selection_fdr)
    mean_ic = np.nanmean(ic, axis=0)
    samples = [m for d, m in signals.correlations.items() if d in set(train_dates)]
    corr = np.nanmean(np.stack(samples), axis=0)[np.ix_(cols, cols)] if samples else np.eye(len(cols))
    kept = []
    for j in sorted(np.flatnonzero(passed), key=lambda j: -abs(t[j])):
        aligned = [np.sign(mean_ic[j]) * np.sign(mean_ic[k]) * corr[j, k] for k in kept]
        if not aligned or max(aligned) < config.selection_max_corr:
            kept.append(j)
    return {"selected": [library[j] for j in kept], "ic": {library[j]: float(mean_ic[j]) for j in kept},
            "candidates": int(np.isfinite(p).sum()), "passed_fdr": int(passed.sum()),
            "t": {library[j]: float(t[j]) for j in kept}}


def prepare_v04(bundle, config, *, library=True):
    """features.prepare (eligibility, labels, v0.2 features) plus the neutral label every v0.4 run
    needs and, with `library`, the processed factor matrix."""
    panel = prepare(bundle, config)
    m = market(panel, bundle)
    rows = panel.loc[panel.eligible, ["date", "symbol"]]
    di, si = m.c.index.get_indexer(rows.date), m.c.columns.get_indexer(rows.symbol)
    log_size = np.log(m.mcap.to_numpy()[di, si])
    groups = industry_groups(bundle, rows, config.industry_snapshot)
    label = panel.loc[rows.index, "label"].to_numpy(dtype=float)[:, None]
    residual = standardize(label, rows.date.to_numpy(), groups.to_numpy(), log_size)[:, 0]
    panel["neutral_target"] = np.nan
    panel.loc[rows.index, "neutral_target"] = cross_rank(residual, rows.date.to_numpy())
    if config.label_mode == "neutral_rank":
        panel["target"] = panel.neutral_target
    if not library:
        return panel, SignalSet(rows.index, rows.date.to_numpy(), [], np.zeros((len(rows), 0), np.float32),
                                pd.DataFrame(), {}, groups, log_size)
    names = list(REGISTRY) + CORE
    raw = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    for name, family, frame in raw_factors(m):
        # Gate 2 failed: provider valuation ratios leave the library (price_level uses prices only).
        if family == "value" and name != "price_level" and not config.valuation_fields:
            continue
        raw[:, names.index(name)] = frame.to_numpy()[di, si]
    for name in FEATURES:
        raw[:, names.index("core_" + name)] = panel.loc[rows.index, name].to_numpy()
    X = standardize(raw, rows.date.to_numpy(), groups.to_numpy(), log_size)
    del raw
    ic = daily_rank_ic(X, panel.loc[rows.index, "target"].to_numpy(dtype=float), rows.date.to_numpy())
    ic.columns = names
    return panel, SignalSet(rows.index, rows.date.to_numpy(), names, X, ic,
                            correlation_samples(X, rows.date.to_numpy()), groups, log_size,
                            {"factors": len(REGISTRY),
                             "coverage": dict(zip(names, np.isfinite(X).mean(axis=0).round(4).tolist()))})
