"""Linear-program portfolio layer; abstain in cash when constraints require it."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from .config import RULES_REVIEWED_THROUGH


def fee_rates(date):
    date = pd.Timestamp(date)
    if not pd.Timestamp("2015-08-01") <= date <= pd.Timestamp(RULES_REVIEWED_THROUGH):
        raise ValueError("Fee schedule outside reviewed historical interval")
    return (0.001 if date < pd.Timestamp("2023-08-28") else 0.0005,
            0.00002 if date < pd.Timestamp("2022-04-29") else 0.00001)


def fees(notional, side, date, config, *, zero=False):
    if notional <= 0 or zero:
        return {"commission": 0.0, "stamp": 0.0, "transfer": 0.0}
    stamp, transfer = fee_rates(date)
    return {"commission": max(config.minimum_commission, config.commission * notional),
            "stamp": notional * stamp if side == "sell" else 0.0,
            "transfer": notional * transfer}


def allocate(scores, previous, volatility, config, policy, date, *, initial=False, locked=None):
    """Return close-t target stock weights. Constraints are not promises of fills.

    LP variables w,buy,sell; w-buy+sell=previous. Cash is residual.
    Risk budget = sum(w_i * trailing daily sigma_i), a conservative volatility
    bound, NOT sector neutrality or a covariance estimate. Scores are ranks, not
    calibrated expected returns; cost_aversion converts estimated fees to rank units.
    `locked` weights (unsettled share-exchange stock) are floors that cannot be sold.
    """
    if policy not in config.policies:
        raise ValueError("Unknown portfolio policy")
    scores = scores.dropna().sort_index().sort_values(ascending=False, kind="stable")
    if config.hold_rank:
        # Hold band: a holding is kept while it ranks within hold_rank; the best names not
        # yet held fill the remaining slots. Cuts churn at the top-k boundary.
        held = set(previous[previous > 1e-12].index)
        kept = [s for s in scores.index[:config.hold_rank] if s in held]
        fresh = [s for s in scores.index if s not in held][:max(0, config.topk - len(kept))]
        candidates = kept + fresh
    else:
        candidates = list(scores.head(config.topk).index)
    names = sorted(set(candidates) | set(previous[previous > 1e-12].index))
    if not names:
        return pd.Series(dtype=float), {"status": "cash", "target_turnover": 0.0}
    prev = previous.reindex(names).fillna(0).to_numpy(float)
    floor = (locked.reindex(names).fillna(0).to_numpy(float) if locked is not None and len(locked)
             else np.zeros(len(names)))
    n = len(names)
    maxw = np.array([config.max_weight if s in candidates else 0.0 for s in names])
    if policy == "topk":
        weights = np.array([min(config.max_weight, config.invested / max(1, len(candidates)))
                            if s in candidates else 0.0 for s in names])
        weights = np.maximum(weights, floor)
        return pd.Series(weights, index=names), {"status": "equal_weight", "target_turnover": float(abs(weights-prev).sum()),
                                                 "compliance_trim": 0.0}
    sigma = volatility.reindex(names).fillna(config.risk_budget * 2).clip(lower=0.001).to_numpy(float)
    # Constant investment utility avoids interpreting a negative rank as a short.
    utility = scores.rank(pct=True).reindex(names).fillna(0).to_numpy(float) + 0.5
    utility = np.array([u if s in candidates else -0.5 for s, u in zip(names, utility)])
    stamp, transfer = fee_rates(date)
    base = config.commission + transfer + config.assumed_slippage_bps / 10_000
    penalty = config.cost_aversion if policy in {"penalty", "cost"} else 0.0
    objective = np.r_[-utility, np.full(n, penalty * base + 1e-9),
                      np.full(n, penalty * (base + stamp) + 1e-9)]
    eq = np.c_[np.eye(n), -np.eye(n), np.eye(n)]
    ub = [np.r_[np.ones(n), np.zeros(2*n)], np.r_[sigma, np.zeros(2*n)]]
    rhs = [config.invested, config.risk_budget]
    # The four LP arms share incumbent feasibility rules; only the penalty and turnover cap vary.
    if config.weight_cap_mode == "enforce":
        # Selling every overweight down to the cap (or to its locked floor) is always feasible
        # because the trim it forces is granted on top of the turnover cap.
        compliance = float(np.maximum(prev - np.maximum(config.max_weight, floor), 0).sum())
        maxw = np.minimum(np.maximum(maxw, prev), config.max_weight)
    else:
        # Existing breaches may remain but cannot be increased.
        compliance = 0.0
        maxw = np.maximum(maxw, prev)
    maxw = np.maximum(maxw, floor)
    if policy in {"turnover", "cost"}:
        # Initial build is explicitly exempt; it still incurs fees and is reported.
        ub.append(np.r_[np.zeros(n), np.ones(2*n)])
        rhs.append(2.0 if initial else config.turnover_cap + compliance)
    rhs[0] = max(config.invested, float(prev.sum()))
    rhs[1] = max(config.risk_budget, float(sigma @ prev))
    solved = linprog(objective, A_ub=np.array(ub), b_ub=np.array(rhs), A_eq=eq,
        b_eq=prev, bounds=[(float(lo), float(x)) for lo, x in zip(floor, maxw)] + [(0, None)]*(2*n), method="highs")
    if not solved.success:
        # No silent switch to unconstrained top-k; preserve holdings and record it.
        return previous.copy(), {"status": "hold_fallback", "reason": solved.message,
                                  "target_turnover": 0.0, "compliance_trim": 0.0}
    w = solved.x[:n]
    residual = float(max(0, np.max(np.array(ub) @ solved.x - rhs)))
    if residual > 1e-7 or abs(eq @ solved.x - prev).max() > 1e-7:
        raise AssertionError("Optimizer returned an infeasible solution")
    return pd.Series(w, index=names), {"status": "optimal", "target_turnover": float(abs(w-prev).sum()),
        "estimated_vol_bound": float(sigma @ w), "constraint_residual": residual,
        "incumbent_risk_breach": bool(sigma @ prev > config.risk_budget + 1e-8),
        "compliance_trim": compliance}
