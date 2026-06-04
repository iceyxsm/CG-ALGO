"""Probability calibration and the cross-asset transfer matrix.

Calibration: LightGBM win probabilities are not guaranteed to mean what they say
("0.8" should win about 80 percent of the time). A monotonic isotonic fit on the
validation split corrects this; it is fit out-of-sample and then frozen, never
fit on test, so it cannot leak.

Transfer matrix: the core referee of the project. Each cell trains on one set of
assets and evaluates on a target asset's temporally held-out test split. The
within-asset diagonal is the performance ceiling; the gap to the off-diagonal
cells is the "generalization tax" that distinguishes a learned market language
from a single asset's dialect. The frozen-deployment experiment (train on an
early period, evaluate on a later period of unseen assets, no retraining) is just
the cell where train and target assets differ and the target test split is later
in time, so it needs no special path.
"""
import numpy as np

from .dataset import build_dataset, temporal_split, embargo_for, SplitConfig
from .features import FeatureConfig
from .labeling import BarrierConfig, WIN, LOSS
from .baseline import train_lightgbm, predict_win_prob
from .metrics import evaluate_strategy, CostConfig


def _isotonic(prob, won):
    """Fit a monotonic step function mapping predicted prob -> empirical rate.

    Pool-adjacent-violators; returns sorted breakpoints (xs, ys) for stepwise
    interpolation. Kept dependency-free so calibration needs no sklearn.
    """
    order = np.argsort(prob, kind="mergesort")
    xs = prob[order].astype(float)
    ys = won[order].astype(float)
    w = np.ones_like(ys)
    i = 0
    # Pool-adjacent-violators on the (xs-sorted) target values.
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            new = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            ys[i] = new
            w[i] += w[i + 1]
            ys = np.delete(ys, i + 1)
            w = np.delete(w, i + 1)
            xs = np.delete(xs, i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    return xs, ys


def fit_calibrator(prob, y_true):
    """Fit an isotonic calibrator on validation predictions (WIN vs not)."""
    won = (np.asarray(y_true) == WIN).astype(float)
    xs, ys = _isotonic(np.asarray(prob), won)
    return lambda p: np.interp(p, xs, ys)


def build_splits(df, feat_cfg, bar_cfg, split_cfg, **kw):
    """Build a dataset and apply embargoed temporal splits in one call."""
    X, y, idx = build_dataset(df, feat_cfg, bar_cfg, split_cfg)
    return temporal_split(X, y, idx, embargo=embargo_for(bar_cfg, split_cfg), **kw)


def _stack(splits_list, part):
    parts = [s[part] for s in splits_list]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _auc(y, p):
    """Mann-Whitney AUC; nan if only one class present."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def skill(y_true, win_prob, eps=1e-12):
    """Predictive skill on every test sample, independent of any trade rule.

    auc       WIN vs (LOSS+TIMEOUT). Inflated by volatility, since whether any
              barrier is touched depends on it, so read dir_auc alongside.
    dir_auc   WIN vs LOSS with TIMEOUT dropped: pure directional skill, the
              honest test of whether geometry predicts which barrier hits first.
    log_loss / base_log_loss  information beyond the class prior.
    """
    yt = np.asarray(y_true)
    y = (yt == WIN).astype(int)
    p = np.clip(np.asarray(win_prob, dtype=float), eps, 1 - eps)
    auc = _auc(y, p)
    resolved = (yt == WIN) | (yt == LOSS)
    dir_auc = _auc((yt[resolved] == WIN).astype(int), p[resolved]) \
        if resolved.any() else float("nan")
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    base = y.sum() / len(y) if len(y) else 0.0
    base_ll = float(-(base * np.log(base) + (1 - base) * np.log(1 - base))) \
        if 0 < base < 1 else 0.0
    return {"auc": float(auc), "dir_auc": float(dir_auc),
            "n_resolved": int(resolved.sum()), "log_loss": ll,
            "base_log_loss": base_ll, "n_test": int(len(y))}


def walk_forward(df, n_folds=5, feat_cfg=FeatureConfig(), bar_cfg=BarrierConfig(),
                 split_cfg=SplitConfig(), fit=train_lightgbm,
                 predict=predict_win_prob):
    """Directional skill across sequential time folds on one asset.

    The full sample sequence is cut into n_folds ordered blocks; fold k trains on
    blocks 0..k-1 and tests on block k, with an embargo trimmed at the seam. A
    signal that is real should hold across folds, not appear only in one lucky
    test window. Returns per-fold dir_auc (WIN vs LOSS, TIMEOUT dropped).
    """
    X, y, idx = build_dataset(df, feat_cfg, bar_cfg, split_cfg)
    emb = embargo_for(bar_cfg, split_cfg)
    bounds = np.linspace(0, len(X), n_folds + 1, dtype=int)
    out = []
    for k in range(1, n_folds):
        tr_hi = bounds[k]
        # embargo: drop train tail whose labels resolve into the test block
        while tr_hi > 0 and idx[tr_hi - 1] + emb >= idx[bounds[k]]:
            tr_hi -= 1
        tr = (X[:tr_hi], y[:tr_hi], idx[:tr_hi])
        te_lo, te_hi = bounds[k], bounds[k + 1]
        va_lo = max(tr_hi - len(X) // 10, 0)
        va = (X[va_lo:tr_hi], y[va_lo:tr_hi], idx[va_lo:tr_hi])
        model = fit(tr, va)
        yte = y[te_lo:te_hi]
        p = predict(model, X[te_lo:te_hi])
        m = (yte == WIN) | (yte == LOSS)
        out.append({"fold": k, "n_resolved": int(m.sum()),
                    "dir_auc": float(_auc((yte[m] == WIN).astype(int), p[m]))
                    if m.any() else float("nan")})
    return out


def transfer_matrix(assets, cells, feat_cfg=FeatureConfig(),
                    bar_cfg=BarrierConfig(), split_cfg=SplitConfig(),
                    cost_cfg=CostConfig(), calibrate=True,
                    fit=train_lightgbm, predict=predict_win_prob):
    """Run a set of train/test cells over named assets.

    `assets`: dict name -> OHLCV DataFrame.
    `cells`: list of (train_names, test_name). Pooled training stacks per-asset
             datasets (never raw series). Each asset is split independently so
             one asset's history cannot leak into another's window.
    Returns a list of result dicts including expectancy and the within/cross flag.
    """
    splits = {name: build_splits(df, feat_cfg, bar_cfg, split_cfg)
              for name, df in assets.items()}
    results = []
    for train_names, test_name in cells:
        tr = _stack([splits[n] for n in train_names], "train")
        va = _stack([splits[n] for n in train_names], "val")
        model = fit(tr, va)

        cal = None
        if calibrate:
            cal = fit_calibrator(predict(model, va[0]), va[1])

        Xte, yte, _ = splits[test_name]["test"]
        prob = predict(model, Xte)
        if cal is not None:
            prob = cal(prob)
        res = evaluate_strategy(yte, prob, bar_cfg.tp, bar_cfg.sl,
                                cost_cfg=cost_cfg)
        res.update(skill(yte, prob))
        res["train"] = "+".join(train_names)
        res["test"] = test_name
        res["within_asset"] = (train_names == [test_name])
        results.append(res)
    return results
