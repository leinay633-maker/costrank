"""Barra-style equity risk model for v0.4 (RESEARCH section 9). Causal throughout.

Exposures at close t: eight styles (MAD-winsorized, float-cap-weighted mean 0, equal-weight
standard deviation 1 over the eligible names) and industry dummies from a mapping fixed by the
first industry snapshot. Factor returns: the return from close t to close t+1 regressed on the
exposures at t, weights sqrt(float cap). Covariance at t uses factor returns dated t or earlier:
zero-mean EWMA, half-life `risk_half_life` sessions, at least `risk_min_history` of them. Specific
variance: EWMA of squared residuals with the same half-life, shrunk toward its size-decile mean.
A return enters the regression only when the stock traded on both closes (a suspended close is a
stale mark). Residuals need only the later close traded: a return after a suspension spans every
session since the last traded close, is taken net of all their factor returns, and enters the
specific variance divided by that span. Listed names outside the eligible pool (holdings that left
it) are covered too: exposures with the pool's winsorization and standardization, residuals against
the pool's factor returns, shrinkage toward the pool's size deciles. They enter portfolio risk,
never the benchmark or the regression.
"""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from .factors import beta_parts, idio_var, mean
from .signals import MAD_SCALE, industry_series, static_industries

STYLES = ["size", "beta", "momentum", "residual_vol", "liquidity", "value", "earnings_yield", "reversal"]


def descriptor_frames(m, valuation=True):
    """Wide raw style descriptors (sessions x symbols); without valuation fields the value and
    earnings-yield exposures are zero."""
    turn = m.turn.fillna(0)
    blank = m.c * np.nan
    return {"size": np.log(m.mcap),
            "beta": beta_parts(m, 252)[0],
            "momentum": m.c.shift(21) / m.c.shift(252) - 1,
            "residual_vol": np.sqrt(idio_var(m, 63)[2]),
            "liquidity": np.log(mean(turn, 21).where(lambda x: x > 0)),
            "value": 1 / m.pb.where(m.pb > 0) if valuation else blank,
            "earnings_yield": 1 / m.pe.where(m.pe > 0) if valuation else blank,
            "reversal": m.c / m.c.shift(21) - 1}


def span_factor_returns(f, carried, t, span):
    """Factor returns over each residual's span: the return ending at close t + 1 covers the closes
    t + 1 - span .. t + 1, so this session's f (close t to t + 1) plus those realized at sessions
    t - span + 2 .. t; carried[j] holds the sum of factor returns realized at sessions before j."""
    return f + carried[t + 1] - carried[np.maximum(t - np.asarray(span) + 2, 0)]


def spans(traded):
    """Sessions covered by each traded close's return since the previous traded close: 1 normally,
    more after a suspension; NaN where the close did not trade or no earlier close traded."""
    pos = pd.DataFrame(np.where(traded.to_numpy(), np.arange(len(traded), dtype=float)[:, None], np.nan),
                       index=traded.index, columns=traded.columns)
    return (pos - pos.ffill().shift(1)).where(traded)


def standardize_styles(raw, cap, other=None):
    """Per session: MAD winsorize, cap-weighted mean 0, equal-weight sd 1; missing -> 0. `other`
    (rows outside the pool) gets the pool's clip bounds, mean and sd; returned second when given."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(raw, axis=0)
        dev = np.nanmedian(np.abs(raw - med), axis=0) * MAD_SCALE
    clip = lambda r: np.where(dev > 0, np.clip(r, med - 5 * dev, med + 5 * dev), r)
    x = clip(raw)
    xo = clip(other) if other is not None else np.zeros((0, raw.shape[1]))
    w = np.where(np.isfinite(cap), cap, 0.0)
    out, out_other = np.zeros_like(x), np.zeros_like(xo)
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        if ok.sum() < 3 or w[ok].sum() <= 0:
            continue
        mu = np.average(x[ok, j], weights=w[ok])
        sd = np.std(x[ok, j])
        if sd > 0:
            out[ok, j] = (x[ok, j] - mu) / sd
            fine = np.isfinite(xo[:, j])
            out_other[fine, j] = (xo[fine, j] - mu) / sd
    return out if other is None else (out, out_other)


@dataclass
class Snapshot:
    """Risk model at one close: names are that session's eligible symbols (the benchmark)."""
    names: pd.Index
    exposures: pd.DataFrame  # names x (styles + industries)
    covariance: pd.DataFrame  # factor covariance, daily units
    specific: pd.Series  # daily specific variance per name
    industry: pd.Series
    # The session's other listed symbols (holdings outside the pool), same columns.
    outside_exposures: pd.DataFrame = None
    outside_specific: pd.Series = None
    outside_industry: pd.Series = None

    def cover(self, symbols):
        """Exposures, daily specific variance and industry of any symbols, in their order; a
        symbol the model does not cover (no bar this session) gets NaN."""
        parts = [(self.exposures, self.specific, self.industry)]
        if self.outside_exposures is not None:
            parts.append((self.outside_exposures, self.outside_specific, self.outside_industry))
        X, d, ind = (pd.concat([p[i] for p in parts]) for i in range(3))
        return X.reindex(symbols), d.reindex(symbols), ind.reindex(symbols)

    def active(self, weights, invested):
        """A book's active weights against `invested` x the equal-weight pool, over the pool and
        the book's covered holdings outside it (a holding without a bar this session is dropped)."""
        held = weights[self.cover(weights.index)[1].notna().to_numpy()]
        union = self.names.union(held.index)
        bench = pd.Series(invested / len(self.names), index=self.names)
        return held.reindex(union).fillna(0.0) - bench.reindex(union).fillna(0.0)

    def exposure(self, active):
        """Factor exposures of active weights over covered symbols, in `covariance` order."""
        return self.cover(active.index)[0].to_numpy().T @ active.to_numpy()

    def daily_variance(self, active):
        """active: Series of active weights over covered symbols; pool names it omits count as
        zero, uncovered symbols are skipped."""
        X, d, _ = self.cover(active.index)
        ok = d.notna().to_numpy()
        a = active.to_numpy()[ok]
        y = X.to_numpy()[ok].T @ a
        return float(y @ self.covariance.to_numpy() @ y + (d.to_numpy()[ok] * a ** 2).sum())


class RiskModel:
    def __init__(self, panel, m, bundle, config, *, keep=None, visit=None):
        """Run the daily regressions over all sessions; keep snapshots for sessions in `keep`.
        visit(date, snapshot), if given, sees every session's snapshot without storing it."""
        self.config = config
        lam = 0.5 ** (1 / config.risk_half_life)
        rows = panel[["date", "symbol"]]
        pool = panel.eligible.fillna(False).to_numpy(dtype=bool)
        mapping = static_industries(bundle, panel.date.min())
        industries = industry_series(bundle, rows, mapping, config.industry_snapshot).to_numpy()
        symbols = rows.symbol.to_numpy()
        raw = descriptor_frames(m, config.valuation_fields)
        dates = m.c.index
        di = dates.get_indexer(rows.date)
        si = m.c.columns.get_indexer(rows.symbol)
        cap = m.mcap.to_numpy()[di, si]
        styles = np.column_stack([raw[s].to_numpy()[di, si] for s in STYLES])
        # Close t to close t+1 where t+1 traded; `span` counts the sessions since the last traded
        # close before t+1 (1 unless the stock was suspended at t). Pool rows traded at t: span 1.
        traded = m.raw_v.gt(0)
        nspan = spans(traded).shift(-1).to_numpy()[di, si]
        nxt = m.ret.shift(-1).where(traded.shift(-1, fill_value=False)).to_numpy()[di, si]
        nxt = np.where(np.isfinite(nspan), nxt, np.nan)
        keep = set(pd.DatetimeIndex(keep)) if keep is not None else set()
        groups = sorted(set(industries[pool]))
        self.factors = STYLES + ["ind_" + g for g in groups]
        k = len(self.factors)
        cov, weight_sum, history = np.zeros((k, k)), 0.0, 0
        spec, spec_w, spec_n = {}, {}, {}
        self.snapshots, self.factor_returns = {}, {}
        order = np.argsort(di, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(di, minlength=len(dates)))]
        pending = None  # regression at t is realized with the t+1 return, i.e. known at close t+1
        carried = np.zeros((len(dates) + 1, k))  # carried[t + 1]: factor returns realized at sessions <= t
        for t in range(len(dates)):
            session = order[bounds[t]:bounds[t + 1]]
            idx, out = session[pool[session]], session[~pool[session]]
            carried[t + 1] = carried[t]
            if pending is not None:
                f, resid_names, squares = pending
                cov = lam * cov + np.outer(f, f)
                weight_sum = lam * weight_sum + 1
                history += 1
                for name, u2 in zip(resid_names, squares):
                    spec[name] = lam * spec.get(name, 0.0) + u2
                    spec_w[name] = lam * spec_w.get(name, 0.0) + 1
                    spec_n[name] = spec_n.get(name, 0) + 1
                self.factor_returns[dates[t]] = f
                carried[t + 1] += f
                pending = None
            if len(idx) == 0:
                continue
            names, outside = symbols[idx], symbols[out]
            col = {g: len(STYLES) + i for i, g in enumerate(groups)}
            X, Xo = np.zeros((len(idx), k)), np.zeros((len(out), k))
            X[:, :len(STYLES)], Xo[:, :len(STYLES)] = standardize_styles(styles[idx], cap[idx], styles[out])
            X[np.arange(len(idx)), [col[g] for g in industries[idx]]] = 1.0
            known = np.array([g in col for g in industries[out]], dtype=bool)  # a group the pool lacks: none
            Xo[np.flatnonzero(known), [col[g] for g in industries[out][known]]] = 1.0
            if (dates[t] in keep or visit is not None) and history >= config.risk_min_history:
                F = cov / weight_sum
                own = lambda group: np.array([spec[n] / spec_w[n] if spec_n.get(n, 0) >= 60 else np.nan
                                              for n in group], dtype=float)
                var, var_out = own(names), own(outside)
                # Names without a float cap sit outside the deciles and shrink toward the overall mean;
                # names outside the pool fall in the pool's deciles by their own cap.
                sized = np.isfinite(cap[idx])
                deciles, deciles_out = np.full(len(idx), -1), np.full(len(out), -1)
                if sized.sum() >= 10:
                    deciles[sized] = pd.qcut(pd.Series(cap[idx][sized]).rank(method="first"), 10, labels=False).to_numpy()
                    edges = [cap[idx][sized][deciles[sized] == d].max() for d in range(9)]
                    fine = np.isfinite(cap[out])
                    deciles_out[fine] = np.searchsorted(edges, cap[out][fine], side="left")
                overall = np.nanmean(var) if np.isfinite(var).any() else 1e-4
                for d in range(-1, 10):
                    group = var[(deciles == d) & np.isfinite(var)]
                    target = group.mean() if len(group) and d >= 0 else overall
                    for v, dec in ((var, deciles), (var_out, deciles_out)):
                        sel = dec == d
                        v[sel] = np.where(np.isfinite(v[sel]), (1 - config.specific_shrink) * v[sel]
                                          + config.specific_shrink * target, target)
                snap = Snapshot(pd.Index(names), pd.DataFrame(X, index=names, columns=self.factors),
                                pd.DataFrame(F, index=self.factors, columns=self.factors),
                                pd.Series(var, index=names), pd.Series(industries[idx], index=names),
                                pd.DataFrame(Xo, index=outside, columns=self.factors),
                                pd.Series(var_out, index=outside), pd.Series(industries[out], index=outside))
                if dates[t] in keep:
                    self.snapshots[dates[t]] = snap
                if visit is not None:
                    visit(dates[t], snap)
            r = nxt[idx]
            ok = np.isfinite(r) & np.isfinite(cap[idx]) & (cap[idx] > 0)
            if ok.sum() > k + 10 and t + 1 < len(dates):
                w = np.sqrt(cap[idx][ok])
                sw = np.sqrt(w)
                f = np.linalg.lstsq(X[ok] * sw[:, None], r[ok] * sw, rcond=None)[0]
                every = np.concatenate([idx, out])
                fine = np.isfinite(nxt[every])
                covered = nspan[every][fine].astype(int)
                net = span_factor_returns(f, carried, t, covered)
                u = nxt[every][fine] - (np.vstack([X, Xo])[fine] * net).sum(axis=1)
                pending = (f, symbols[every][fine], u * u / covered)

    def snapshot(self, day):
        return self.snapshots.get(pd.Timestamp(day))
