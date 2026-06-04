"""OHLCV loading and a synthetic generator for pipeline testing.

Real data: provide a CSV with columns open, high, low, close (volume optional),
ordered oldest first. The pipeline never consumes absolute price, coin identity,
or indicators; only candle geometry derived in features.py is used downstream.
"""
import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


def load_ohlcv_csv(path, column_map=None):
    """Load an OHLCV CSV into a standardized oldest-first DataFrame."""
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=column_map)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df[REQUIRED].astype(float).reset_index(drop=True)


def generate_synthetic_ohlcv(n=50000, seed=0, start=100.0, drift=0.0, vol=0.002):
    """Random-walk OHLCV for end-to-end pipeline testing only (not real data).

    The `vol` argument controls per-candle volatility; vary it to emulate a
    different asset for a quick cross-asset sanity check.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0.0, vol, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
