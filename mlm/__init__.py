"""Market Language Model: a price-action research pipeline.

The pipeline learns from candle geometry only. It deliberately excludes
indicators, absolute price levels, and coin identity so that any learned edge
must reflect general market behavior rather than asset-specific memorization.
"""
from .data import load_ohlcv_csv, generate_synthetic_ohlcv
from .features import FeatureConfig, compute_features, FEATURE_COLUMNS
from .labeling import BarrierConfig, label_entry, WIN, LOSS, TIMEOUT
from .dataset import SplitConfig, build_dataset, temporal_split
from .metrics import breakeven_winrate, expectancy, evaluate_strategy
from .baseline import train_lightgbm, predict_win_prob

__all__ = [
    "load_ohlcv_csv", "generate_synthetic_ohlcv",
    "FeatureConfig", "compute_features", "FEATURE_COLUMNS",
    "BarrierConfig", "label_entry", "WIN", "LOSS", "TIMEOUT",
    "SplitConfig", "build_dataset", "temporal_split",
    "breakeven_winrate", "expectancy", "evaluate_strategy",
    "train_lightgbm", "predict_win_prob",
]
