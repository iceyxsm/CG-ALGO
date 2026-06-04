# Market Language Model

A research pipeline that tests whether a model can learn generalized crypto
price-action behavior from candle geometry alone, without indicators, absolute
prices, or coin identity. It is built to make the experiment honest (no
look-ahead, no inflated statistics, no volatility confound) and pessimistic
(conservative tie-break, costs included), so a positive result is meaningful.

## Layout

    mlm/
      data.py       OHLCV loading and a synthetic generator for testing
      features.py   candle geometry + trailing volatility normalization
      labeling.py   triple-barrier labels with timeout and conservative tie-break
      dataset.py    windowing, non-overlapping sampling, temporal splits
      metrics.py    breakeven win-rate and cost-adjusted expectancy
      baseline.py   LightGBM baseline
    train.ipynb     orchestrates the full pipeline
    requirements.txt

## Setup

    pip install -r requirements.txt

Then open `train.ipynb`. To use real data, replace `generate_synthetic_ohlcv`
with `load_ohlcv_csv('btc_5m.csv')`. The CSV needs columns open, high, low,
close (oldest first); use `column_map` to rename if needed.

## Design decisions

These map one to one to the methodology risks identified before building.

1. Two-stage normalization. Geometry as a percentage of open removes price
   level; dividing by a trailing volatility estimate removes per-asset and
   per-regime scale. This is what makes the cross-asset test meaningful rather
   than just measuring "DOGE candles are bigger than BTC candles."
2. Range% dropped. It is approximately body + upper wick + lower wick, so it
   carries no extra information. `log_vol` is kept as a channel so the model
   still knows the regime is calm or wild.
3. Wicks are positive magnitudes; only the body carries a sign.
4. Vertical barrier. A horizon caps the hold; unresolved entries are labeled
   TIMEOUT rather than left undefined.
5. Non-overlapping sampling. Entries advance to the label resolution index, so
   samples do not share future candles and statistics are not inflated.
6. No look-ahead. The volatility window ends at the previous candle, so a
   feature at time t never sees candle t or the future.
7. Conservative tie-break. When both barriers fall in one candle, the loss is
   assumed first, biasing results pessimistic.
8. Strict temporal splits. Train, validation, and test are ordered in time and
   never shuffled; a held-out asset provides the cross-asset test.
9. Cost-adjusted expectancy. The no-skill breakeven win-rate is sl/(tp+sl) plus
   costs; success is beating that hurdle, not 50 percent.
10. Baseline first. LightGBM sets the bar. A deeper model (1D-CNN, LSTM, then a
    small transformer with optional masked-candle pretraining) is only worth it
    if it beats this baseline.

## Caveats

The synthetic generator is a random walk with no learnable edge; it exists only
to prove the pipeline runs and to confirm expectancy lands near zero (a check
that nothing is leaking). Conclusions require real OHLCV history. Single OHLC
bars cannot resolve intrabar touch order, which is why the tie-break is
conservative; minute data can refine this later.
