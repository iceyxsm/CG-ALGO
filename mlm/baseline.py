"""LightGBM baseline on flattened candle-geometry windows.

This is the bar a deeper model must beat. The task is framed as binary
WIN-vs-not so the output is a probability of reaching take-profit first, which
feeds directly into the cost-adjusted expectancy in metrics.py.
"""
import numpy as np
import lightgbm as lgb

from .labeling import WIN

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


def train_lightgbm(train, val, params=None, num_round=300, early_stopping=30):
    """Train on (X, y, idx) splits. Multiclass labels are mapped to WIN vs not."""
    Xtr, ytr, _ = train
    Xva, yva, _ = val
    ytr_bin = (ytr == WIN).astype(int)
    yva_bin = (yva == WIN).astype(int)
    dtr = lgb.Dataset(Xtr, label=ytr_bin)
    dva = lgb.Dataset(Xva, label=yva_bin, reference=dtr)
    model = lgb.train(
        params or PARAMS, dtr, num_boost_round=num_round, valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
    )
    return model


def predict_win_prob(model, X):
    """Return the predicted probability of WIN for each row of X."""
    return model.predict(X, num_iteration=model.best_iteration)
