"""Candle geometry features with trailing volatility normalization.

Two-stage normalization:
  1. Geometry is expressed as a percentage of the candle open, which removes the
     absolute price level (a BTC candle and a DOGE candle look comparable).
  2. Each geometry feature is divided by a trailing volatility estimate, which
     removes the per-asset and per-regime volatility scale. Without this step a
     BTC-only model would see other coins as pure outliers, confounding the
     cross-asset generalization test.

No look-ahead: the trailing volatility window ends at the previous candle, so a
feature at time t never depends on candle t itself or any future candle.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["body_n", "uwick_n", "lwick_n", "log_vol"]


@dataclass
class FeatureConfig:
    vol_window: int = 100   # trailing candles used for the volatility estimate
    eps: float = 1e-8       # guards divide-by-zero and log(0)


def compute_features(df, cfg=FeatureConfig()):
    """Return a DataFrame of normalized features plus a `valid` mask.

    `valid` is False during the volatility warm-up where the estimate is NaN.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o) / o * 100.0
    upper_wick = (h - np.maximum(o, c)) / o * 100.0   # always a positive magnitude
    lower_wick = (np.minimum(o, c) - l) / o * 100.0   # always a positive magnitude
    rng = (h - l) / o * 100.0                         # used only for the vol estimate

    # Trailing volatility: rolling std of range%, shifted by one so the window
    # covers candles t-vol_window .. t-1 and excludes the current candle.
    vol = rng.shift(1).rolling(cfg.vol_window).std()
    vol_safe = vol.clip(lower=cfg.eps)

    out = pd.DataFrame({
        "body_n": body / vol_safe,
        "uwick_n": upper_wick / vol_safe,
        "lwick_n": lower_wick / vol_safe,
        "log_vol": np.log(vol_safe),
    }, index=df.index)
    out["valid"] = vol.notna()
    return out
