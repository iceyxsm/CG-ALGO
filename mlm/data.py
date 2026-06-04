"""OHLCV loading.

Real data: provide a CSV with columns open, high, low, close (volume optional),
ordered oldest first. The pipeline never consumes absolute price, coin identity,
or indicators; only candle geometry derived in features.py is used downstream.
"""
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
