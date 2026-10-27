"""Paired OOS inference; no IID t-tests on overlapping labels or stock rows."""
import numpy as np
import pandas as pd


def performance(daily):
    r = daily.net_return.to_numpy(float)
    equity = np.r_[1.0, np.cumprod(1+r)]
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    return {"sessions": len(r), "total_return": float(equity[-1]-1),
        "annualized_return": float(equity[-1]**(252/len(r))-1),
        "annualized_volatility": vol*np.sqrt(252),
        "sharpe_rf0": float(r.mean()/vol*np.sqrt(252)) if vol > 0 else np.nan,
        "max_drawdown": float(np.min(equity/np.maximum.accumulate(equity)-1)),
        "mean_two_way_turnover": float(daily.turnover_two_way.mean()),
        "sum_two_way_turnover": float(daily.turnover_two_way.sum()),
        "mean_exposure": float(daily.exposure.mean()),
        "max_single_weight": float(daily.max_weight.max()),
        "gross_same_trades_total_return": float(np.prod(1+daily.gross_same_trades_return)-1),
        **{f"total_{x}": float(daily[x].sum()) for x in ["commission", "stamp", "transfer", "slippage"]}}


def rank_ic(predictions, panel):
    merged = predictions.merge(panel[["date", "symbol", "label"]], on=["date", "symbol"], how="left", validate="many_to_one")
    rows = []
    for (model, day), group in merged.groupby(["model", "date"]):
        observed = group.dropna(subset=["label", "score"])
        ic = np.nan
        if len(observed) >= 5 and observed.score.nunique() > 1 and observed.label.nunique() > 1:
            ic = observed.score.rank().corr(observed.label.rank())
        rows.append({"model": model, "date": day, "rank_ic": ic,
                     "predicted_names": len(group), "observed_labels": len(observed)})
    return pd.DataFrame(rows)


def paired_block_interval(left, right, *, samples=1000, block=20, seed=42):
    """Circular moving-block bootstrap of paired DAILY arithmetic excess returns."""
    if not left.index.equals(right.index):
        raise ValueError("Paired returns must have identical date indices")
    delta = (left-right).to_numpy(float)
    if len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("Insufficient/nonfinite paired returns")
    rng = np.random.default_rng(seed)
    n, block = len(delta), min(block, len(delta))
    estimates = np.empty(samples)
    for i in range(samples):
        starts = rng.integers(0, n, size=(n+block-1)//block)
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        estimates[i] = delta[idx].mean()*252
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    # Two-sided percentile p-value: how often the resampled mean lands across zero.
    p = min(1.0, 2 * max(min((estimates <= 0).mean(), (estimates >= 0).mean()), 1 / samples))
    return {"annualized_mean_difference": float(delta.mean()*252),
            "ci95_low": float(lo), "ci95_high": float(hi), "p_bootstrap": float(p),
            "sessions": n, "block_sessions": block, "bootstrap_samples": samples,
            "method": "paired circular block bootstrap; Holm adjustment across the reported family"}


def holm(pvalues, alpha=0.05):
    """Holm step-down: reject while p_(i) <= alpha / (m - i); flags returned in input order."""
    p = np.asarray(pvalues, float)
    reject = np.zeros(len(p), bool)
    for step, i in enumerate(np.argsort(p, kind="stable")):
        if p[i] > alpha / (len(p) - step):
            break
        reject[i] = True
    return reject


def relative(strategy, benchmark):
    """Daily strategy returns against a benchmark on the same dates (rf=0)."""
    x = pd.concat([strategy, benchmark], axis=1, keys=["s", "b"]).dropna()
    if len(x) < 2:
        return {}
    active = x.s - x.b
    beta = float(np.cov(x.s, x.b, ddof=1)[0, 1] / np.var(x.b, ddof=1))
    annual = lambda r: float(np.prod(1 + r) ** (252 / len(r)) - 1)
    te = float(active.std(ddof=1) * np.sqrt(252))
    return {"benchmark_annualized_return": annual(x.b), "active_annualized_return": annual(x.s) - annual(x.b),
            "beta": beta, "alpha_annual_arith": float((x.s - beta * x.b).mean() * 252),
            "tracking_error": te, "information_ratio": float(active.mean() * 252 / te) if te > 0 else np.nan,
            "correlation": float(x.s.corr(x.b))}


def sharpe(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return x.mean() / sd if sd > 0 else np.nan


def _sharpe_moments(x, y):
    """Delta-method pieces for SR(x) - SR(y) with SR = mu / sqrt(gamma - mu^2)."""
    mx, my, gx, gy = x.mean(), y.mean(), (x ** 2).mean(), (y ** 2).mean()
    grad = np.array([gx / (gx - mx ** 2) ** 1.5, -gy / (gy - my ** 2) ** 1.5,
                     -mx / (2 * (gx - mx ** 2) ** 1.5), my / (2 * (gy - my ** 2) ** 1.5)])
    v = np.column_stack([x - mx, y - my, x ** 2 - gx, y ** 2 - gy])
    diff = mx / np.sqrt(gx - mx ** 2) - my / np.sqrt(gy - my ** 2)
    return diff, grad, v


def _hac(v, bandwidth):
    """Bartlett-kernel long-run covariance of the rows of v (already centered)."""
    T = len(v)
    psi = v.T @ v / T
    for lag in range(1, bandwidth + 1):
        g = v[lag:].T @ v[:-lag] / T
        psi += (1 - lag / (bandwidth + 1)) * (g + g.T)
    return psi


def sharpe_difference_test(x, y, *, block=20, samples=10_000, seed=42, bandwidth=20):
    """Ledoit and Wolf (2008) test of equal Sharpe ratios of two paired daily series: HAC
    (Bartlett) standard error for the observed difference, studentized circular block bootstrap
    with the pair resampled jointly, statistics centered at the observed difference (the null),
    p = (#{|d*| >= |d|} + 1) / (samples + 1). Ratios are arithmetic mean / sd, annualized."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or len(x) < 2 * block or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Need equal-length finite paired series longer than two blocks")
    T = len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff, grad, v = _sharpe_moments(x, y)
        se = np.sqrt(max(grad @ _hac(v, bandwidth) @ grad / T, 0.0)) if np.isfinite(grad).all() else np.nan
    if not (np.isfinite(diff) and np.isfinite(se)):
        # A series without variance has no Sharpe ratio: the test is undefined, not significant.
        return {"sharpe_difference_annual": np.nan, "se_annual": np.nan, "p_value": np.nan,
                "sharpe_left_annual": float(sharpe(x) * np.sqrt(252)), "sharpe_right_annual": float(sharpe(y) * np.sqrt(252)),
                "sessions": T, "block": block, "samples": samples, "method": "undefined: a series has no variance"}
    if se == 0:
        if diff != 0:
            raise ValueError("Zero standard error for a nonzero Sharpe difference")
        return {"sharpe_difference_annual": float(diff * np.sqrt(252)), "se_annual": 0.0, "p_value": 1.0,
                "sharpe_left_annual": float(sharpe(x) * np.sqrt(252)), "sharpe_right_annual": float(sharpe(y) * np.sqrt(252)),
                "sessions": T, "block": block, "samples": samples, "method": "Ledoit-Wolf studentized circular block bootstrap"}
    stat = abs(diff) / se
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(T / block))
    exceed = 0
    for _ in range(samples):
        starts = rng.integers(0, T, size=nblocks)
        idx = ((starts[:, None] + np.arange(block)) % T).ravel()[:T]
        bx, by = x[idx], y[idx]
        d_b, g_b, v_b = _sharpe_moments(bx, by)
        # Block-bootstrap variance: mean outer product of scaled block sums (Goetze-Kuensch).
        usable = (T // block) * block
        sums = v_b[:usable].reshape(-1, block, 4).sum(axis=1) / np.sqrt(block)
        psi_b = sums.T @ sums / len(sums)
        se_b = np.sqrt(g_b @ psi_b @ g_b / T)
        exceed += abs(d_b - diff) / se_b >= stat
    return {"sharpe_difference_annual": float(diff * np.sqrt(252)), "se_annual": float(se * np.sqrt(252)),
            "p_value": float((exceed + 1) / (samples + 1)), "sharpe_left_annual": float(sharpe(x) * np.sqrt(252)),
            "sharpe_right_annual": float(sharpe(y) * np.sqrt(252)), "sessions": T, "block": block,
            "samples": samples, "method": "Ledoit-Wolf studentized circular block bootstrap"}


def minimum_detectable(se, *, alpha=0.05, power=0.80):
    """Smallest true difference a two-sided level-alpha test detects with the given power."""
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def deflated_sharpe(returns, trials, trial_sharpe_variance):
    """Bailey and Lopez de Prado (2014): probability that the observed per-period Sharpe exceeds
    the expected maximum of `trials` unskilled trials whose Sharpe ratios vary by the given
    variance (per-period units)."""
    from scipy.stats import kurtosis, norm, skew
    r = np.asarray(returns, float)
    sr, T = sharpe(r), len(r)
    euler = 0.5772156649015329
    sr0 = np.sqrt(trial_sharpe_variance) * ((1 - euler) * norm.ppf(1 - 1 / trials)
                                            + euler * norm.ppf(1 - 1 / (trials * np.e)))
    denominator = np.sqrt(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr ** 2)
    return {"deflated_sharpe": float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denominator)),
            "expected_max_sharpe_annual": float(sr0 * np.sqrt(252)), "trials": int(trials),
            "sharpe_annual": float(sr * np.sqrt(252))}


def pbo_cscv(returns, segments=16):
    """Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey,
    Borwein, Lopez de Prado and Zhu 2017): returns is sessions x strategies."""
    from itertools import combinations
    r = np.asarray(returns, float)
    T, n = r.shape
    edges = np.linspace(0, T, segments + 1).astype(int)
    parts = [np.arange(edges[i], edges[i + 1]) for i in range(segments)]
    logits = []
    for chosen in combinations(range(segments), segments // 2):
        ins = np.concatenate([parts[i] for i in chosen])
        outs = np.concatenate([parts[i] for i in range(segments) if i not in chosen])
        best = int(np.argmax([sharpe(r[ins, j]) for j in range(n)]))
        oos = np.array([sharpe(r[outs, j]) for j in range(n)])
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1) + 1
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "combinations": len(logits), "segments": segments,
            "strategies": n}
