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


def _trade_returns(y_true, taken, tp, sl, cost_cfg, timeout_return):
    """Per-trade net returns for the taken trades (empty array if none)."""
    taken = np.asarray(taken, dtype=bool)
    if taken.sum() == 0:
        return np.empty(0)
    y = np.asarray(y_true)[taken]
    cost = 2.0 * (cost_cfg.fee + cost_cfg.slippage)
    return np.where(y == WIN, tp, np.where(y == LOSS, -sl, timeout_return)) - cost


def expectancy(y_true, taken, tp, sl, cost_cfg=CostConfig(), timeout_return=0.0):
    """Mean return per taken trade, net of round-trip costs.

    `taken` is a boolean mask of entries the strategy chose to trade. WIN earns
    +tp, LOSS earns -sl, TIMEOUT earns `timeout_return`; round-trip cost is
    subtracted from every taken trade.
    """
    r = _trade_returns(y_true, taken, tp, sl, cost_cfg, timeout_return)
    return float(r.mean()) if len(r) else 0.0


def bootstrap_ci(returns, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for mean per-trade return.

    Resamples trade outcomes with replacement to ask whether a positive mean
    expectancy is distinguishable from zero given the trade count. A CI whose
    lower bound stays above zero is the bar; a wide CI straddling zero is the
    small-n mirage that trade count alone only hints at.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def evaluate_strategy(y_true, win_prob, tp, sl, threshold=None,
                      cost_cfg=CostConfig(), timeout_return=0.0, n_boot=2000):
    """Take trades whose predicted win probability clears `threshold`.

    Threshold defaults to the cost-adjusted breakeven win-rate. The bootstrap CI
    on expectancy answers whether a positive result survives the trade count.
    """
    be = breakeven_winrate(tp, sl, cost_cfg)
    thr = be if threshold is None else threshold
    taken = np.asarray(win_prob) >= thr
    r = _trade_returns(y_true, taken, tp, sl, cost_cfg, timeout_return)
    lo, hi = bootstrap_ci(r, n_boot=n_boot)
    return {
        "breakeven_winrate": be,
        "threshold": thr,
        "n_taken": int(taken.sum()),
        "coverage": float(taken.mean()) if len(taken) else 0.0,
        "expectancy": float(r.mean()) if len(r) else 0.0,
        "exp_ci_lo": lo,
        "exp_ci_hi": hi,
    }
