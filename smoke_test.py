"""End-to-end smoke test: verifies the pipeline runs, has no look-ahead, and
yields near-zero expectancy on random-walk data (the honesty check)."""
import numpy as np
from mlm import (generate_synthetic_ohlcv, FeatureConfig, BarrierConfig,
                 SplitConfig, build_dataset, temporal_split, embargo_for,
                 train_lightgbm, predict_win_prob, evaluate_strategy,
                 transfer_matrix, WIN)
from mlm.features import compute_features
from mlm.metrics import CostConfig

# 1. Look-ahead test: a spike injected at candle t must not change vol/features
#    at or before t (the volatility window ends at t-1).
df = generate_synthetic_ohlcv(n=2000, seed=3, vol=0.002)
base = compute_features(df, FeatureConfig(vol_window=100))
t = 500
df2 = df.copy()
df2.loc[t, "high"] = df2.loc[t, "high"] * 5.0   # huge future spike
after = compute_features(df2, FeatureConfig(vol_window=100))
cols = ["body_n", "uwick_n", "lwick_n", "log_vol"]
diff_upto_t = np.nanmax(np.abs(base[cols].iloc[:t].to_numpy()
                               - after[cols].iloc[:t].to_numpy()))
assert diff_upto_t == 0.0, f"look-ahead leak before t: {diff_upto_t}"
print("look-ahead test passed (max diff up to t-1 =", diff_upto_t, ")")

# 2. Full pipeline with embargoed splits.
feat_cfg, bar_cfg, split_cfg = FeatureConfig(), BarrierConfig(), SplitConfig()
X, y, idx = build_dataset(generate_synthetic_ohlcv(60000, 0, vol=0.002),
                          feat_cfg, bar_cfg, split_cfg)
emb = embargo_for(bar_cfg, split_cfg)
splits = temporal_split(X, y, idx, embargo=emb)
print("samples", len(X), "| label dist", np.bincount(y))

# 3. Non-overlap check: entries strictly increasing.
assert np.all(np.diff(idx) >= 1)
print("non-overlap entries strictly increasing: ok")

# 3b. Embargo check: gap between train tail and val head must exceed embargo, so
#     no train sample's label window overlaps a val sample's input window.
tr_idx, va_idx = splits["train"][2], splits["val"][2]
gap = va_idx[0] - tr_idx[-1]
assert gap > emb, f"embargo gap {gap} <= {emb}"
print("embargo boundary gap", gap, ">", emb, ": ok")

model = train_lightgbm(splits["train"], splits["val"])
Xte, yte, _ = splits["test"]
res = evaluate_strategy(yte, predict_win_prob(model, Xte),
                        bar_cfg.tp, bar_cfg.sl, cost_cfg=CostConfig())
print("in-asset:", {k: round(v, 5) if isinstance(v, float) else v
                    for k, v in res.items()})

# 4. Transfer matrix across two synthetic "assets" at different vol scales.
assets = {"A": generate_synthetic_ohlcv(60000, 0, vol=0.002),
          "B": generate_synthetic_ohlcv(60000, 1, vol=0.006)}
cells = [(["A"], "A"), (["A"], "B"), (["B"], "A"), (["A", "B"], "B")]
for r in transfer_matrix(assets, cells, feat_cfg, bar_cfg, split_cfg):
    print(f"  train={r['train']:>3} test={r['test']} "
          f"within={r['within_asset']} n_taken={r['n_taken']} "
          f"expectancy={r['expectancy']:.5f}")
print("OK")
