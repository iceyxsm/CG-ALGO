"""Windowing, non-overlapping sampling, and strict temporal splits.

Non-overlapping sampling: after an entry is labeled and resolves at candle r, the
next entry starts at r rather than the next candle. Overlapping windows share
future candles, which induces heavy label autocorrelation and inflates apparent
significance; advancing past the resolution point keeps samples near-independent.

Temporal split: data is divided by time (train -> val -> test) and never
shuffled, so the model is always validated and tested on its future.
"""
from dataclasses import dataclass
import numpy as np

from .features import compute_features, FeatureConfig, FEATURE_COLUMNS
from .labeling import label_entry, BarrierConfig


@dataclass
class SplitConfig:
    window: int = 64        # number of past candles per sample
    non_overlap: bool = True


def embargo_for(bar_cfg, split_cfg):
    """Candles to embargo at each split seam: label horizon + input window."""
    return bar_cfg.horizon + split_cfg.window


def build_dataset(df, feat_cfg=FeatureConfig(), bar_cfg=BarrierConfig(),
                  split_cfg=SplitConfig()):
    """Build a flattened feature matrix X, labels y, and entry indices.

    Each row of X is `window` candles of FEATURE_COLUMNS flattened oldest-first.
    Entries advance to the label resolution index when non_overlap is set.
    """
    feats = compute_features(df, feat_cfg)
    fvals = feats[FEATURE_COLUMNS].to_numpy()
    valid = feats["valid"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    w = split_cfg.window
    start = int(np.argmax(valid)) + w   # first entry with a full valid window
    X, y, idx = [], [], []
    entry = start
    n = len(df)
    while entry < n - 1:
        win = fvals[entry - w:entry]
        if not np.isfinite(win).all():
            entry += 1
            continue
        label, resolve = label_entry(high, low, close, entry, bar_cfg)
        X.append(win.reshape(-1))
        y.append(label)
        idx.append(entry)
        entry = resolve if split_cfg.non_overlap else entry + 1
    return np.asarray(X), np.asarray(y), np.asarray(idx)


def temporal_split(X, y, idx, train=0.7, val=0.15, embargo=0):
    """Split arrays by position into train/val/test without shuffling.

    Purge + embargo: a sample's label can resolve up to `horizon` candles after
    its entry, and the next split's inputs reach back `window` candles. With
    embargo = horizon + window, the earlier split's tail is trimmed so no
    training/validation sample's label-resolution candles ever appear inside a
    later split's input window. This closes the cross-boundary leak that
    non-overlapping sampling alone does not cover.
    """
    n = len(X)
    a = int(n * train)
    b = int(n * (train + val))

    def trim(lo, hi, nxt):
        while hi > lo and nxt < n and idx[hi - 1] + embargo >= idx[nxt]:
            hi -= 1
        return slice(lo, hi)

    g = lambda s: (X[s], y[s], idx[s])
    return {"train": g(trim(0, a, a)), "val": g(trim(a, b, b)),
            "test": g(slice(b, n))}
