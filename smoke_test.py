"""Smoke test on a small real fetch: verifies no look-ahead, embargoed splits,
and non-overlapping sampling. Requires network access to the Binance mirror."""
import numpy as np
from mlm import (FeatureConfig, BarrierConfig, SplitConfig, build_dataset,
                 temporal_split, embargo_for, load_ohlcv_csv)
from mlm.features import compute_features
from fetch_data import fetch_to_csv

df = load_ohlcv_csv(fetch_to_csv("BTCUSDT", "5m", days=120))
print("candles", len(df))

# 1. Look-ahead: a spike at candle t must not change features at or before t.
base = compute_features(df, FeatureConfig(vol_window=100))
t = 500
df2 = df.copy()
df2.loc[t, "high"] = df2.loc[t, "high"] * 5.0
after = compute_features(df2, FeatureConfig(vol_window=100))
cols = ["body_n", "uwick_n", "lwick_n", "log_vol"]
diff = np.nanmax(np.abs(base[cols].iloc[:t].to_numpy()
                        - after[cols].iloc[:t].to_numpy()))
assert diff == 0.0, f"look-ahead leak: {diff}"
print("look-ahead test passed (max diff up to t-1 =", diff, ")")

# 2. Build, non-overlap, embargo.
feat_cfg, bar_cfg, split_cfg = FeatureConfig(), BarrierConfig(), SplitConfig()
X, y, idx = build_dataset(df, feat_cfg, bar_cfg, split_cfg)
emb = embargo_for(bar_cfg, split_cfg)
splits = temporal_split(X, y, idx, embargo=emb)
print("samples", len(X), "| label dist", np.bincount(y))

assert np.all(np.diff(idx) >= 1)
print("non-overlap entries strictly increasing: ok")

tr_idx, va_idx = splits["train"][2], splits["val"][2]
gap = va_idx[0] - tr_idx[-1]
assert gap > emb, f"embargo gap {gap} <= {emb}"
print("embargo boundary gap", gap, ">", emb, ": ok")
print("OK")
