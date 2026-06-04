"""Triple-barrier labeling with a timeout and a conservative tie-break.

Labels:
    1 = WIN     take-profit reached first
    0 = LOSS    stop-loss reached first
    2 = TIMEOUT neither barrier reached within the horizon

Conservative tie-break: when both barriers fall inside the same candle the order
of touches is unknown on a single OHLC bar, so the loss is assumed to hit first.
This biases outcomes to be pessimistic, which is the safe default for a strategy
that might later risk capital.
"""
from dataclasses import dataclass

WIN, LOSS, TIMEOUT = 1, 0, 2


@dataclass
class BarrierConfig:
    tp: float = 0.02       # take-profit as a fraction of entry price (0.02 = +2%)
    sl: float = 0.01       # stop-loss as a fraction of entry price (0.01 = -1%)
    horizon: int = 48      # vertical barrier: max candles to hold


def label_entry(high, low, close, entry, cfg):
    """Label a single entry index. Returns (label, resolve_idx).

    `resolve_idx` is the candle at which the outcome was decided; it is used by
    the dataset builder to advance non-overlapping samples.
    """
    entry_price = close[entry]
    tp_level = entry_price * (1.0 + cfg.tp)
    sl_level = entry_price * (1.0 - cfg.sl)
    last = min(entry + cfg.horizon, len(close) - 1)
    for j in range(entry + 1, last + 1):
        if low[j] <= sl_level:        # check loss first => conservative tie-break
            return LOSS, j
        if high[j] >= tp_level:
            return WIN, j
    return TIMEOUT, last
