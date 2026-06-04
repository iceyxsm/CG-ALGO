# Market Language Model

A research pipeline that tests whether a model can learn generalized crypto
price-action behavior from candle geometry alone, without indicators, absolute
prices, or coin identity. It is built to make the experiment honest (no
look-ahead, no inflated statistics, no volatility confound) and pessimistic
(conservative tie-break, costs included), so a positive result is meaningful.

## Layout

    mlm/
      data.py       OHLCV loading
      features.py   candle geometry + trailing volatility normalization
      labeling.py   triple-barrier labels with timeout and conservative tie-break
      dataset.py    windowing, non-overlapping sampling, embargoed temporal splits
      metrics.py    breakeven win-rate and cost-adjusted expectancy
      baseline.py   LightGBM baseline
      evaluation.py isotonic calibration + cross-asset transfer matrix
    train.ipynb     orchestrates the full pipeline
    requirements.txt

## Setup

    pip install -r requirements.txt

Then fetch data (below) and open `train.ipynb`. Load any CSV with columns
open, high, low, close (oldest first) via `load_ohlcv_csv('btcusdt_5m.csv')`;
use `column_map` to rename columns if needed.

## Getting data

`fetch_data.py` pulls paginated 5-minute history from Binance public klines into
pipeline-ready CSVs (stdlib only, no extra dependencies):

    python fetch_data.py --symbols BTCUSDT ETHUSDT DOGEUSDT --interval 5m --days 365

Output files like `btcusdt_5m.csv` load directly via `load_ohlcv_csv`. CSVs are
gitignored on purpose; data does not belong in the repo. If api.binance.com is
geo-blocked in your region, pass `--base https://data-api.binance.vision`.

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
8. Strict temporal splits with purge and embargo. Train, validation, and test
   are ordered in time and never shuffled. An embargo of horizon + window
   candles is trimmed at each seam so no earlier sample's label-resolution
   candles appear inside a later split's input window; non-overlap alone does
   not cover this cross-boundary leak. A held-out asset provides the
   cross-asset test.
9. Cost-adjusted expectancy. The no-skill breakeven win-rate is sl/(tp+sl) plus
   costs; success is beating that hurdle, not 50 percent.
10. Calibrated thresholds, selected out-of-sample. Win probabilities are
    isotonic-calibrated on the validation split and frozen before test, and the
    trade threshold defaults to the cost-adjusted breakeven. Thresholds are
    never tuned on test, which is the most common way this experiment fools its
    author.
11. Transfer matrix as referee. The within-asset diagonal is the performance
    ceiling; the gap to off-diagonal and pooled cells is the generalization tax
    that separates a market language from an asset dialect. Frozen deployment
    (train early, test later on unseen assets, no retraining) is just a cell
    where train and target assets differ.
12. Baseline first, but as a diagnostic, not a gate. LightGBM flattens the
    window and cannot represent translation-invariant motifs, so a tree winning
    is informative but a tree losing is not evidence of no edge. The ladder
    (LightGBM, then 1D-CNN for local motifs, then a transformer for long-range
    structure, with optional masked-candle pretraining) has each rung test a
    distinct claim about what carries the signal.

## Caveats

Single OHLC bars cannot resolve intrabar touch order, which is why the tie-break
is conservative; minute data can refine this later. Stage one of the experiment
measures predictive skill (AUC, log-loss) to ask whether the normalized geometry
carries learnable, transferable structure; the cost-adjusted expectancy and any
leverage or position-sizing belong to a later strategy stage and do not gate the
representation question.
