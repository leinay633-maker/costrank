"""Identical features, labels, folds and eligible rows for every fitted model."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .features import FEATURES


def daily_spearman(scores, target, dates):
    """Mean over sessions of the within-session Spearman correlation (sessions with >= 20 names)."""
    frame = pd.DataFrame({"d": dates, "s": scores, "y": target}).dropna()
    values = [g.s.rank().corr(g.y.rank()) for _, g in frame.groupby("d") if len(g) >= 20]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def rank_autocorrelation(scores, symbols, dates, lag=5):
    """Mean Spearman correlation of scores `lag` sessions apart over common names."""
    frame = pd.DataFrame({"d": dates, "sym": symbols, "s": scores})
    sessions = sorted(frame.d.unique())
    by = {d: g.set_index("sym").s for d, g in frame.groupby("d")}
    values = []
    for a, b in zip(sessions[:-lag], sessions[lag:]):
        x, y = by[a], by[b]
        common = x.index.intersection(y.index)
        if len(common) >= 20:
            values.append(x[common].rank().corr(y[common].rank()))
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


@dataclass
class Fold:
    number: int
    train_start: str
    validation_start: str
    train_label_before: str
    validation_label_before: str
    test_start: str
    test_end: str


def folds(calendar, config, holdout=False):
    c = pd.DatetimeIndex(calendar)
    start, end = ((config.holdout_start, config.holdout_end) if holdout else
                  (config.dev_start, config.dev_end))
    indices = np.flatnonzero((c >= start) & (c <= end))
    if len(indices) == 0:
        raise ValueError("No sessions in requested evaluation interval")
    size = len(indices) if holdout else config.test_days
    out = []
    for number, offset in enumerate(range(0, len(indices), size)):
        i = indices[offset]
        v = i - config.valid_days
        tr = v - config.train_days
        if tr < 0 or v - config.embargo_days < 0:
            raise ValueError("Insufficient prehistory for declared rolling windows; do not silently shorten")
        fmt = lambda j: c[j].strftime("%Y-%m-%d")
        out.append(Fold(number, fmt(tr), fmt(v), fmt(v - config.embargo_days),
                        fmt(i - config.embargo_days), fmt(i),
                        fmt(indices[min(offset + size - 1, len(indices) - 1)])))
    return out


def predict_oos(panel, calendar, config, *, holdout=False, on_fold=None, signals=None, fold_list=None):
    """`signals` (signals.SignalSet) switches to the v0.4 library: per-fold selection on the
    training window, models on the selected processed factors. `fold_list` replaces the folds of
    the configured interval (forward paper trading)."""
    if (config.feature_set == "v04") != (signals is not None):
        raise ValueError("feature_set v04 needs the processed factor library, v02 must not get one")
    xcols = ["x_" + x for x in FEATURES]
    eligible = panel.loc[panel.eligible].copy()
    if signals is not None and not eligible.index.equals(signals.index):
        raise ValueError("Factor rows must be the panel's eligible rows in order")
    results, audits = [], []
    for fold in (fold_list if fold_list is not None else folds(calendar, config, holdout)):
        train_mask = ((eligible.date >= fold.train_start) & (eligible.date < fold.validation_start) &
                      (eligible.label_end < fold.train_label_before) & eligible.target.notna()).to_numpy()
        valid_mask = ((eligible.date >= fold.validation_start) & (eligible.date < fold.test_start) &
                      (eligible.label_end < fold.validation_label_before) & eligible.target.notna()).to_numpy()
        test_mask = ((eligible.date >= fold.test_start) & (eligible.date <= fold.test_end)).to_numpy()
        train, valid, test = eligible[train_mask], eligible[valid_mask], eligible[test_mask]
        if len(train) < 50 or len(valid) < 20 or test.empty:
            raise ValueError(f"Insufficient eligible rows in fold {fold.number}: {len(train)}/{len(valid)}/{len(test)}")
        # Each training date has equal total weight, independent of changing pool size.
        weights = 1 / train.groupby("date").date.transform("size").to_numpy()
        weights *= len(weights) / weights.sum()
        audit = dict(asdict(fold), train_rows=len(train), valid_rows=len(valid), test_rows=len(test),
                     train_max_label_end=str(train.label_end.max()),
                     valid_max_label_end=str(valid.label_end.max()), models={})
        if signals is None:
            design = {"train": train[xcols], "valid": valid[xcols], "test": test[xcols]}
            reversal = {k: -f.x_ret5.to_numpy() for k, f in (("valid", valid), ("test", test))}
            ic_weights = None
        else:
            from .factors import REGISTRY
            from .signals import CORE, select
            chosen = select(signals, train_mask, sorted(train.date.unique()), config, list(REGISTRY))
            columns = chosen["selected"]
            if len(columns) < 5:
                columns, chosen["fallback_core"] = list(CORE), True
                ic = signals.ic.loc[signals.ic.index.isin(train.date.unique()), CORE].mean()
                chosen["ic"] = {k: float(v) for k, v in ic.items()}
            audit["selection"] = chosen
            cols = [signals.names.index(c) for c in columns]
            design = {k: pd.DataFrame(signals.X[np.ix_(mask, cols)], columns=columns, index=f.index)
                      for k, mask, f in (("train", train_mask, train), ("valid", valid_mask, valid),
                                         ("test", test_mask, test))}
            r5 = signals.names.index("ret_5")
            reversal = {k: -np.nan_to_num(signals.X[mask, r5].astype(float))
                        for k, mask in (("valid", valid_mask), ("test", test_mask))}
            ic_weights = pd.Series(chosen["ic"])[columns]
        for name in config.models:
            if name == "factor":
                score, check = reversal["test"], reversal["valid"]  # preregistered short-term reversal
                info = {"formula": "negative 5-session return" + (" (processed, neutralized)" if signals else
                                                                   " cross-sectional rank")}
            elif name == "ridge":
                model = Ridge(alpha=config.ridge_alpha)
                model.fit(design["train"].fillna(0), train.target, sample_weight=weights)
                score, check = model.predict(design["test"].fillna(0)), model.predict(design["valid"].fillna(0))
                info = {"alpha": config.ridge_alpha,
                        "coefficients": dict(zip(design["train"].columns, model.coef_.tolist()))}
            elif name == "ic_linear":
                if ic_weights is None:
                    ic_weights = pd.Series({c: daily_spearman(train[c].to_numpy(), train.target.to_numpy(),
                                                              train.date.to_numpy()) for c in xcols})
                w = ic_weights.fillna(0) / max(ic_weights.abs().sum(), 1e-12)
                score = design["test"].fillna(0).to_numpy() @ w.to_numpy()
                check = design["valid"].fillna(0).to_numpy() @ w.to_numpy()
                info = {"weights": {k: float(v) for k, v in w.items()}}
            else:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(n_estimators=config.trees, learning_rate=0.05,
                    num_leaves=15, max_depth=-1, min_child_samples=100, reg_lambda=10,
                    random_state=config.seed, n_jobs=config.threads, deterministic=True,
                    force_col_wise=True, verbosity=-1)
                model.fit(design["train"], train.target, sample_weight=weights,
                    eval_set=[(design["valid"], valid.target)], eval_metric="l2",
                    callbacks=[lgb.early_stopping(25, verbose=False)])
                score, check = model.predict(design["test"]), model.predict(design["valid"])
                info = {"best_iteration": int(model.best_iteration_),
                        "importance_split": dict(zip(design["train"].columns, model.feature_importances_.tolist()))}
            if "neutral_target" in valid:
                # Inputs to the v0.4 optimizer: validation IC on the residual label and the 5-session
                # rank autocorrelation of scores (cost amortization), both before the test window.
                info["valid_ic"] = daily_spearman(check, valid.neutral_target.to_numpy(), valid.date.to_numpy())
                info["valid_rho5"] = rank_autocorrelation(check, valid.symbol.to_numpy(), valid.date.to_numpy())
            if not np.isfinite(score).all():
                raise ValueError(f"Nonfinite predictions in {name}, fold {fold.number}")
            result = test[["date", "symbol"]].copy()
            result["score"], result["model"], result["fold"] = score, name, fold.number
            results.append(result)
            audit["models"][name] = info
        audits.append(audit)
        if on_fold:
            on_fold(audit)
    predictions = pd.concat(results, ignore_index=True)
    if predictions.duplicated(["date", "symbol", "model"]).any():
        raise AssertionError("Overlapping OOS predictions")
    return predictions, audits
