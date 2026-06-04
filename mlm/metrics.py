"""Cost-adjusted expectancy and the no-skill breakeven hurdle.

A fixed take-profit/stop-loss scheme has a built-in breakeven win-rate of
sl/(tp+sl); for tp=2%/sl=1% that is about 33%, so the target is roughly zero
expected value before costs. Trading fees and slippage push the real hurdle
higher. Success is beating the cost-adjusted breakeven, not beating 50%.
"""
from dataclasses import dataclass
import numpy as np

from .labeling import WIN, LOSS, TIMEOUT


@dataclass
class CostConfig:
    fee: float = 0.0005       # taker fee per side as a fraction (0.0005 = 0.05%)
    slippage: float = 0.0005  # assumed slippage per side as a fraction


def breakeven_winrate(tp, sl, cost_cfg=CostConfig()):
    """Win-rate at which expectancy is zero, including round-trip costs."""
    cost = 2.0 * (cost_cfg.fee + cost_cfg.slippage)
    return (sl + cost) / (tp + sl)


def expectancy(y_true, taken, tp, sl, cost_cfg=CostConfig(), timeout_return=0.0):
    """Mean return per taken trade, net of round-trip costs.

    `taken` is a boolean mask of entries the strategy chose to trade. WIN earns
    +tp, LOSS earns -sl, TIMEOUT earns `timeout_return`; round-trip cost is
    subtracted from every taken trade.
    """
    taken = np.asarray(taken, dtype=bool)
    if taken.sum() == 0:
        return 0.0
    y = np.asarray(y_true)[taken]
    cost = 2.0 * (cost_cfg.fee + cost_cfg.slippage)
    r = np.where(y == WIN, tp, np.where(y == LOSS, -sl, timeout_return)) - cost
    return float(r.mean())


def evaluate_strategy(y_true, win_prob, tp, sl, threshold=None,
                      cost_cfg=CostConfig(), timeout_return=0.0):
    """Take trades whose predicted win probability clears `threshold`.

    Threshold defaults to the cost-adjusted breakeven win-rate.
    """
    be = breakeven_winrate(tp, sl, cost_cfg)
    thr = be if threshold is None else threshold
    taken = np.asarray(win_prob) >= thr
    return {
        "breakeven_winrate": be,
        "threshold": thr,
        "n_taken": int(taken.sum()),
        "coverage": float(taken.mean()) if len(taken) else 0.0,
        "expectancy": expectancy(y_true, taken, tp, sl, cost_cfg, timeout_return),
    }
