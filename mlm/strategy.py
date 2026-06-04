"""Stage 2: strategy evaluation under futures costs.

Stage 1 settled the representation question: a weak but real, transferable,
predominantly local directional signal exists in normalized candle geometry.
This stage asks the separate, harder question of whether it is harvestable.

The model is used as a directional filter via its confidence ranking. For each
risk-reward and confidence threshold the selected entries are re-labeled against
the actual price path, so the reported win rate and expectancy reflect that exit
rule rather than the single barrier the model trained on. Costs are futures
taker fees (per side) plus an approximate per-trade funding charge. Leverage
scales whatever expectancy survives costs; it is a multiplier, not an edge, and
a stop of sl at leverage L risks L*sl of margin (liquidation when L*sl -> 1).
"""
import numpy as np

from .labeling import label_entry, BarrierConfig, WIN, LOSS


def relabel(df, idx, tp, sl, horizon):
    """Outcome of each entry under a (tp, sl, horizon) barrier on real prices."""
    high, low, close = (df["high"].to_numpy(), df["low"].to_numpy(),
                        df["close"].to_numpy())
    cfg = BarrierConfig(tp=tp, sl=sl, horizon=horizon)
    return np.array([label_entry(high, low, close, int(e), cfg)[0] for e in idx])


def sweep(df, idx, prob, rr_list, conf_fracs, horizon=48, sl=0.01,
          fee=0.0004, funding=0.0003, leverage=1.0):
    """Grid over risk-reward x confidence fraction.

    rr_list: TP/SL ratios (tp = sl * rr).
    conf_fracs: top fraction of predictions to trade (1.0 = trade all).
    Returns one row per combo with selected-trade win rate, the cost-adjusted
    breakeven it must beat, expectancy per trade after costs, and the leveraged
    expectancy. WIN earns +tp, LOSS -sl, TIMEOUT 0 (exit flat, pay only costs).
    """
    prob = np.asarray(prob)
    cost = 2.0 * fee + funding
    rows = []
    for rr in rr_list:
        tp = sl * rr
        labels = relabel(df, idx, tp, sl, horizon)
        be_cost = (sl + cost) / (tp + sl)        # win rate that zeroes EV
        for q in conf_fracs:
            thr = np.quantile(prob, 1 - q)
            sel = prob >= thr
            n = int(sel.sum())
            ls = labels[sel]
            resolved = (ls == WIN) | (ls == LOSS)
            wr = float((ls == WIN).sum() / max(resolved.sum(), 1))
            ret = np.where(ls == WIN, tp, np.where(ls == LOSS, -sl, 0.0)) - cost
            exp = float(ret.mean()) if n else 0.0
            rows.append({
                "rr": rr, "tp": tp, "sl": sl, "conf_frac": q, "n": n,
                "win_rate": round(wr, 4),
                "breakeven_cost": round(be_cost, 4),
                "edge_vs_be": round(wr - be_cost, 4),
                "exp_per_trade": round(exp, 6),
                "exp_levered": round(exp * leverage, 6),
                "liq_risk": leverage * sl >= 1.0,
            })
    return rows
