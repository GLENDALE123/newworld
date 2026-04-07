"""Scalping multi-parameter TBM labeler.

Generates labels for multiple TP/SL ratios and holding periods.
Each combo produces: win/loss label, MAE, MFE, bars_to_hit.
"""

import numpy as np
import pandas as pd
from numba import njit


@njit
def _scalp_tbm(closes, highs, lows, atr, tp_mult, sl_mult, max_hold, direction):
    """Single TBM pass. Returns (label, mae, mfe, bars_to_hit) arrays.

    direction: 1=long, -1=short
    """
    n = len(closes)
    label = np.empty(n)
    mae = np.empty(n)
    mfe = np.empty(n)
    bars = np.empty(n)
    label[:] = np.nan
    mae[:] = np.nan
    mfe[:] = np.nan
    bars[:] = np.nan

    for i in range(n - 1):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        entry = closes[i]
        end = min(i + max_hold, n - 1)
        if end <= i:
            continue

        if direction == 1:
            upper = entry + tp_mult * a
            lower = entry - sl_mult * a
        else:
            upper = entry + sl_mult * a
            lower = entry - tp_mult * a

        # Track MFE/MAE over path
        path_h_max = highs[i + 1]
        path_l_min = lows[i + 1]
        for j in range(i + 2, end + 1):
            if highs[j] > path_h_max:
                path_h_max = highs[j]
            if lows[j] < path_l_min:
                path_l_min = lows[j]

        if direction == 1:
            mfe_val = (path_h_max - entry) / entry
            mae_val = (entry - path_l_min) / entry
        else:
            mfe_val = (entry - path_l_min) / entry
            mae_val = (path_h_max - entry) / entry

        # Find barrier hit
        hit = np.nan
        bars_to_hit = max_hold
        if direction == 1:
            for j in range(i + 1, end + 1):
                if highs[j] >= upper:
                    hit = 1.0
                    bars_to_hit = j - i
                    break
                if lows[j] <= lower:
                    hit = -1.0
                    bars_to_hit = j - i
                    break
        else:
            for j in range(i + 1, end + 1):
                if lows[j] <= lower:
                    hit = 1.0
                    bars_to_hit = j - i
                    break
                if highs[j] >= upper:
                    hit = -1.0
                    bars_to_hit = j - i
                    break

        if np.isnan(hit):
            if direction == 1:
                pnl = closes[end] - entry
            else:
                pnl = entry - closes[end]
            hit = 1.0 if pnl > 0 else -1.0

        label[i] = hit
        mae[i] = mae_val
        mfe[i] = mfe_val
        bars[i] = bars_to_hit

    return label, mae, mfe, bars


@njit
def _compute_atr(high, low, close, period):
    """ATR: True Range → EMA. More stable than returns std for barrier sizing."""
    n = len(close)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    atr = np.empty(n)
    atr[:] = np.nan
    if period < n:
        s = 0.0
        for i in range(1, period + 1):
            s += tr[i]
        atr[period] = s / period
        alpha = 2.0 / (period + 1)
        for i in range(period + 1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


# Default scalping parameter grid
SCALP_PARAMS = [
    # (tp_mult, sl_mult, max_hold_bars)
    (1.5, 1.0, 2),   # tight, fast
    (1.5, 1.0, 3),
    (2.0, 1.0, 2),   # standard
    (2.0, 1.0, 3),
    (2.0, 1.0, 6),   # standard, longer hold
    (3.0, 1.0, 3),   # wide TP
    (3.0, 1.0, 6),
    (3.0, 1.5, 6),   # wide both
    (2.0, 0.5, 3),   # tight SL
]


def generate_scalp_labels(
    ohlcv_5m: pd.DataFrame,
    params: list[tuple] | None = None,
    vol_span: int = 24,
    fee: float = 0.0004,
) -> pd.DataFrame:
    """Generate multi-parameter scalping labels.

    Args:
        ohlcv_5m: 5m OHLCV with timestamp index
        params: list of (tp_mult, sl_mult, max_hold) tuples
        vol_span: EWM span for volatility
        fee: one-way fee (halved round-trip)

    Returns:
        DataFrame with columns:
        - label_{tp}_{sl}_{h}_{dir}: +1/-1 win/loss
        - mae_{tp}_{sl}_{h}_{dir}: max adverse excursion
        - mfe_{tp}_{sl}_{h}_{dir}: max favorable excursion
        - bars_{tp}_{sl}_{h}_{dir}: bars to barrier hit
    """
    if params is None:
        params = SCALP_PARAMS

    c = ohlcv_5m["close"].values.astype(np.float64)
    h = ohlcv_5m["high"].values.astype(np.float64)
    l = ohlcv_5m["low"].values.astype(np.float64)
    # ATR-based volatility (not returns std — prevents label imbalance)
    atr = _compute_atr(h, l, c, vol_span)

    result = {}
    for tp, sl, hold in params:
        for dir_name, direction in [("long", 1), ("short", -1)]:
            suffix = f"{tp}_{sl}_{hold}_{dir_name}"
            lab, mae, mfe, bars = _scalp_tbm(c, h, l, atr, tp, sl, hold, direction)
            result[f"label_{suffix}"] = lab
            result[f"mae_{suffix}"] = mae
            result[f"mfe_{suffix}"] = mfe
            result[f"bars_{suffix}"] = bars

    return pd.DataFrame(result, index=ohlcv_5m.index)
