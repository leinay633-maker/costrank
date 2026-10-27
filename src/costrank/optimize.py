"""v0.4 mean-variance policy (RESEARCH section 9), solved as a conic program with Clarabel.

At each decision close: maximize alpha'w - kappa * (linear fees + square-root impact), all in
fractions of NAV over one rebalance interval, subject to sum(w) = invested, 0 <= w <= max_weight
(locked exchange shares as floors), industry and style active bounds against the eligible-pool
equal-weight benchmark scaled to `invested`, ex-ante tracking error, and participation.
alpha = IC * daily specific vol * sqrt(horizon) * z (Grinold); IC is the model's mean daily
Spearman against the residual label over the latest matured out-of-sample sessions.
Participation is never relaxed: a name or industry bound it cannot reach within one session (a
holding that left the pool goes to zero) moves to the nearest reachable weight (RESEARCH section 9,
supplement 7). Holdings outside the pool carry their own exposures and specific risk. When the
trading limits admit a portfolio but the risk limits cannot all hold, the industry bands, style
bands and TE widen by the smallest common factor that makes them feasible (supplement 8).
"""
SOLVED = {"Solved", "AlmostSolved"}
import math
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .portfolio import fee_rates
from .risk import STYLES


def matured_ic(predictions, panel):
    """Daily Spearman of one model's scores against the residual label, with the label's maturity."""
    merged = predictions.merge(panel[["date", "symbol", "neutral_target", "label_end"]], on=["date", "symbol"])
    rows = []
    for day, g in merged.groupby("date"):
        g = g.dropna(subset=["score", "neutral_target"])
        if len(g) >= 20:
            rows.append((day, g.label_end.max(), g.score.rank().corr(g.neutral_target.rank())))
    return pd.DataFrame(rows, columns=["date", "label_end", "ic"])


class Planner:
    def __init__(self, config, risk, ic_table, folds, *, objective="mvo"):
        """ic_table: matured_ic() of the model; folds: per-fold dicts with test_start, test_end,
        valid_ic and valid_rho5 of the model."""
        if objective not in {"mvo", "mvo_rank"}:
            raise ValueError("objective must be mvo or mvo_rank")
        self.config, self.risk, self.ic, self.folds, self.objective = config, risk, ic_table, folds, objective
        self.audits = []

    def fold(self, day):
        for f in self.folds:
            if pd.Timestamp(f["test_start"]) <= day <= pd.Timestamp(f["test_end"]):
                return f
        return self.folds[-1]

    def information(self, day):
        mature = self.ic[self.ic.label_end <= day].tail(self.config.alpha_ic_window)
        fold = self.fold(day)
        if len(mature) >= self.config.alpha_ic_min:
            return float(mature.ic.mean()), "matured_oos", fold
        return float(fold.get("valid_ic", np.nan)), "validation", fold

    def __call__(self, day, score, weights, market, nav, locked, initial):
        cfg, started = self.config, time.perf_counter()
        day = pd.Timestamp(day)
        snap = self.risk.snapshot(day)
        previous = weights[weights > 1e-12]
        hold = {"status": "hold_fallback", "target_turnover": 0.0, "compliance_trim": 0.0}
        if snap is None:
            return previous.copy(), dict(hold, reason="no_risk_model")
        names = snap.names
        n_bench = len(names)
        wb_each = cfg.invested / n_bench
        s = score.reindex(names)
        z = ((s - s.mean()) / s.std()).fillna(0.0) if s.std() > 0 else pd.Series(0.0, index=names)
        ic, ic_source, fold = self.information(day)
        ic = max(ic, 0.0) if np.isfinite(ic) else 0.0
        rho = fold.get("valid_rho5", np.nan)
        kappa = cfg.kappa_scale * (1 - rho if np.isfinite(rho) else 1.0)
        sigma = np.sqrt(snap.specific.clip(lower=0))
        alpha = ic * sigma * math.sqrt(cfg.horizon) * z
        industry = snap.industry
        top = alpha.sort_values(ascending=False, kind="stable")
        candidates = set(top.index[:cfg.candidate_top])
        for _, group in top.groupby(industry.reindex(top.index), sort=False):
            candidates.update(group.index[:cfg.industry_top])
        candidates |= set(previous.index) | set(locked.index[locked > 0])
        order = sorted(candidates)
        n = len(order)
        eligible = np.asarray(pd.Index(order).isin(names))
        # Holdings outside the pool keep their own exposures, industry and specific risk.
        X_cover, d_cover, industry_cover = snap.cover(order)
        w0 = weights.reindex(order).fillna(0.0).to_numpy()
        floor = locked.reindex(order).fillna(0.0).to_numpy()
        upper = np.where(eligible, np.maximum(cfg.max_weight, floor), floor)
        adv = market.adv20.reindex(order).fillna(0.0).to_numpy()  # full-session ADV20 (impact)
        room = (market["adv_cap"] if "adv_cap" in market else market.adv20).reindex(order).fillna(0.0).to_numpy()
        part = cfg.participation * np.maximum(room, 0.0) / nav
        # Participation binds every name; one that cannot trade keeps its current weight.
        frozen = part <= 0
        floor = np.where(frozen, w0, floor)
        upper = np.where(frozen, w0, np.maximum(upper, floor))
        # A bound the participation limit cannot reach this session (a holding that left the pool,
        # an overweight after a rally) moves to the nearest reachable weight; later decisions go on.
        unreachable = w0 - part > upper + 1e-12
        upper = np.maximum(upper, w0 - part)
        stamp, transfer = fee_rates(day)
        buy_fee = cfg.commission + transfer + cfg.assumed_slippage_bps / 10_000
        sell_fee = buy_fee + stamp
        d = d_cover.fillna(0.0).clip(lower=0).to_numpy()  # a name without a bar carries no specific risk
        sig = np.where(d_cover.notna().to_numpy(), np.sqrt(d), sigma.median())
        impact = np.where(adv > 0, sig / np.sqrt(np.maximum(adv, 1.0) / nav), 0.0)
        X = X_cover.fillna(0.0).to_numpy()
        wb = np.where(eligible, wb_each, 0.0)
        bench_exposure = snap.exposures.to_numpy().sum(axis=0) * wb_each
        # The pool's industries and those of holdings outside it (benchmark weight 0 there).
        groups = sorted(set(industry) | set(industry_cover[w0 > 0].dropna()))
        G = np.vstack([(industry_cover == g).to_numpy() for g in groups]).astype(float)
        bench_group = industry.value_counts().reindex(groups).fillna(0).to_numpy() * wb_each
        # Industry bounds that locked shares or the participation limit make unreachable move to the
        # nearest reachable industry weight, as the per-name bounds do.
        group_upper = np.maximum(bench_group + cfg.industry_band, G @ np.maximum(floor, w0 - part))
        group_lower = np.minimum(bench_group - cfg.industry_band, G @ np.minimum(upper, w0 + part))
        style, bench_style = X[:, :len(STYLES)].T, bench_exposure[:len(STYLES)]
        F = snap.covariance.to_numpy()
        L = np.linalg.cholesky(F + 1e-12 * np.eye(len(F)))
        rest = ~names.isin(order)
        c0 = float((snap.specific.to_numpy()[rest] * wb_each ** 2).sum())
        te = cfg.te_annual / math.sqrt(252)
        context = {"outside_weight": float(w0[~eligible].sum()), "unreachable_names": int(unreachable.sum()),
                   "relaxed_industry_bounds": int((group_upper > bench_group + cfg.industry_band + 1e-12).sum()
                                                  + (group_lower < bench_group - cfg.industry_band - 1e-12).sum())}

        # Variables x = [w, buy, sell] and, with the impact term, [t, r]: |buy + sell|^1.5 <= t is
        # written as two 3-dim second-order cones, (buy + sell)^2 <= r t and r^2 <= buy + sell. The
        # equivalent power-cone form stalled (InsufficientProgress) on a few percent of synthetic solves.
        use_impact = self.objective == "mvo"
        W, B, S, T, R = (np.arange(n) + k * n for k in range(5))
        q = np.zeros(5 * n if use_impact else 3 * n)
        if self.objective == "mvo":
            q[W] = -alpha.reindex(order).fillna(0.0).to_numpy()
            q[B], q[S] = kappa * buy_fee, kappa * sell_fee
            q[T] = kappa * impact
        else:
            # v0.2 objective (rank utility minus cost_aversion x linear fee) under the same constraints.
            utility = s.rank(pct=True).reindex(order).fillna(0.0).to_numpy() + 0.5
            q[W] = -np.where(eligible, utility, -0.5)
            q[B], q[S] = cfg.cost_aversion * buy_fee + 1e-9, cfg.cost_aversion * sell_fee + 1e-9
        import clarabel
        eye = sp.identity(n, format="csc")
        ng, ns = len(groups), len(STYLES)

        def solve(mode="declared", widen=0.0):
            """mode "declared": the problem, every risk limit widened by `widen` (industry and style
            bands by widen x band, TE by the factor 1 + widen); "trading": only the budget and the
            trading limits (attribution); "phase": minimize the common widening theta, held in the
            last variable, over the trading-feasible portfolios."""
            phase = mode == "phase"
            groups_of_n = 5 if use_impact and not phase else 3
            nx = groups_of_n * n + phase

            def lay(*blocks, theta=None):
                """One row block over every variable group from per-group blocks (None = zero)."""
                height = next(b.shape[0] for b in blocks if b is not None)
                parts = [sp.csc_matrix((height, n)) if b is None else sp.csc_matrix(b) for b in blocks]
                parts += [sp.csc_matrix((height, n))] * (groups_of_n - len(parts))
                if phase:
                    parts.append(sp.csc_matrix((height, 1)) if theta is None else sp.csc_matrix(np.reshape(theta, (-1, 1))))
                return sp.hstack(parts)
            rows, rhs, cones = [], [], []

            def add(block, values, cone):
                rows.append(sp.csc_matrix(block))
                rhs.append(np.asarray(values, float))
                cones.extend(cone if isinstance(cone, list) else [cone])
            # Equalities: fully invested at `invested`; w - buy + sell = w0.
            add(lay(np.ones((1, n))), [cfg.invested], clarabel.ZeroConeT(1))
            add(lay(eye, -eye, eye), w0, clarabel.ZeroConeT(n))
            ineq = [lay(eye), lay(-eye), lay(None, -eye), lay(None, None, -eye), lay(None, eye, eye)]
            bound = [upper, -floor, np.zeros(n), np.zeros(n), part]
            band, sband = cfg.industry_band, cfg.style_band
            if phase:
                ineq += [lay(G, theta=np.full(ng, -band)), lay(-G, theta=np.full(ng, -band)),
                         lay(style, theta=np.full(ns, -sband)), lay(-style, theta=np.full(ns, -sband)),
                         sp.csc_matrix(([-1.0], ([0], [nx - 1])), shape=(1, nx))]
                bound += [group_upper, -group_lower, bench_style + sband, -(bench_style - sband), [0.0]]
            elif mode == "declared":
                ineq += [lay(G), lay(-G), lay(style), lay(-style)]
                bound += [group_upper + band * widen, -(group_lower - band * widen),
                          bench_style + sband * (1 + widen), -(bench_style - sband * (1 + widen))]
            add(sp.vstack(ineq), np.concatenate(bound), clarabel.NonnegativeConeT(sum(len(b) for b in bound)))
            if mode != "trading":
                # Ex-ante tracking error against the scaled equal-weight benchmark, daily units; in the
                # phase problem the bound is te x (1 + theta).
                head = (sp.csc_matrix(([-te], ([0], [nx - 1])), shape=(1, nx)) if phase else sp.csc_matrix((1, nx)))
                soc_A = sp.vstack([head, lay(-L.T @ X.T), lay(sp.diags(-np.sqrt(d))), sp.csc_matrix((1, nx))])
                soc_b = np.concatenate([[te * (1 + widen)], -L.T @ bench_exposure, -np.sqrt(d) * wb, [math.sqrt(c0)]])
                add(soc_A, soc_b, clarabel.SecondOrderConeT(len(soc_b)))
            if groups_of_n == 5:
                # Name i, x = buy + sell: cone (r + t, 2x, r - t) and cone (x + 1, 2r, x - 1); the slack is b - A x.
                one, first = np.ones(n), 6 * np.arange(n)
                second = first + 3
                cone_rows = np.concatenate([first, first, first + 1, first + 1, first + 2, first + 2,
                                            second, second, second + 1, second + 2, second + 2])
                cone_cols = np.concatenate([R, T, B, S, R, T, B, S, R, B, S])
                cone_vals = np.concatenate([-one, -one, -2 * one, -2 * one, -one, one, -one, -one, -2 * one, -one, -one])
                cone_b = np.zeros(6 * n)
                cone_b[second], cone_b[second + 2] = 1.0, -1.0
                add(sp.csc_matrix((cone_vals, (cone_rows, cone_cols)), shape=(6 * n, nx)), cone_b,
                    [clarabel.SecondOrderConeT(3) for _ in range(2 * n)])
            objective = np.r_[np.zeros(nx - 1), 1.0] if phase else q
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            return clarabel.DefaultSolver(sp.csc_matrix((nx, nx)), objective, sp.vstack(rows).tocsc(),
                                          np.concatenate(rhs), cones, settings).solve()

        solution = solve()
        status = declared_status = str(solution.status)
        widen = 0.0
        if status not in SOLVED:
            # Do the budget and trading limits alone admit a portfolio? If not, keep the book.
            trading = str(solve("trading").status)
            failed = dict(hold, reason=status, without_risk_limits=trading, **context)
            if trading not in SOLVED:
                return previous.copy(), failed
            phase = solve("phase")
            if str(phase.status) not in SOLVED:
                return previous.copy(), dict(failed, phase_status=str(phase.status))
            # The smallest common widening, with a margin for the solver's accuracy.
            widen = max(float(np.asarray(phase.x)[-1]), 0.0) * (1 + 1e-3) + 1e-6
            solution = solve(widen=widen)
            status = str(solution.status)
            if status not in SOLVED:
                # `risk_limit_scale` marks solved days only; this attempt is recorded apart.
                return previous.copy(), dict(failed, widened_attempt=1 + widen, widened_status=status)
        x = np.asarray(solution.x)
        self.sigma = pd.Series(sig, index=order)  # the ledger charges impact with the same sigma
        w = np.clip(x[W], 0.0, None)
        w = np.where(w < 1e-7, 0.0, w)
        target = pd.Series(w, index=order)
        # Ex-ante TE of the returned weights as the constraint measures it (Snapshot.daily_variance).
        y = X.T @ w - bench_exposure
        te_ex_ante = math.sqrt(max(float(y @ F @ y + (d * (w - wb) ** 2).sum()) + c0, 0.0) * 252)
        exposure = y[:len(STYLES)]
        held_group = G @ w
        audit = {"status": "optimal", "solver_status": status, "objective_kind": self.objective,
                 "target_turnover": float(np.abs(w - w0).sum()), "compliance_trim": 0.0,
                 "ic": ic, "ic_source": ic_source, "kappa": kappa, "candidates": n, "holdings": int((w > 0).sum()),
                 "ex_ante_te": te_ex_ante, "max_style_active": float(np.abs(exposure).max()),
                 "max_industry_active": float(np.abs(held_group - bench_group).max()),
                 # 1 unless the risk limits were widened (supplement 8); the excesses are over the
                 # bounds actually imposed, reachability and widening included.
                 "risk_limit_scale": 1 + widen, "declared_status": declared_status,
                 "te_excess": max(te_ex_ante - cfg.te_annual * (1 + widen), 0.0),
                 "style_excess": max(float(np.abs(exposure).max()) - cfg.style_band * (1 + widen), 0.0),
                 "industry_excess": max(float(np.max(held_group - group_upper - cfg.industry_band * widen)),
                                        float(np.max(group_lower - cfg.industry_band * widen - held_group)), 0.0),
                 **context,
                 "solve_seconds": time.perf_counter() - started,
                 "primal_residual": float(getattr(solution, "r_prim", np.nan)),
                 "dual_residual": float(getattr(solution, "r_dual", np.nan))}
        if use_impact:
            # Modelled impact in NAV units before kappa: from the weights, and the solver's bound t.
            audit["impact_cost"] = float(impact @ np.abs(w - w0) ** 1.5)
            audit["impact_cost_solver"] = float(impact @ x[T])
        return target, audit
