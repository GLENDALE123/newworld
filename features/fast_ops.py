"""Numba-accelerated feature computation primitives.

Replaces pandas rolling/ewm chains for 5-10x speedup on large datasets.
"""

import numpy as np
from numba import njit


@njit
def rolling_mean(values, window):
    n = len(values)
    out = np.empty(n)
    out[:window - 1] = np.nan
    s = 0.0
    for i in range(window):
        s += values[i]
    out[window - 1] = s / window
    for i in range(window, n):
        s += values[i] - values[i - window]
        out[i] = s / window
    return out


@njit
def rolling_std(values, window):
    n = len(values)
    out = np.empty(n)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        s = 0.0
        s2 = 0.0
        for j in range(i - window + 1, i + 1):
            v = values[j]
            s += v
            s2 += v * v
        mean = s / window
        var = s2 / window - mean * mean
        out[i] = np.sqrt(max(var, 0.0))
    return out


@njit
def rolling_max(values, window):
    n = len(values)
    out = np.empty(n)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        mx = values[i - window + 1]
        for j in range(i - window + 2, i + 1):
            if values[j] > mx:
                mx = values[j]
        out[i] = mx
    return out


@njit
def rolling_min(values, window):
    n = len(values)
    out = np.empty(n)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        mn = values[i - window + 1]
        for j in range(i - window + 2, i + 1):
            if values[j] < mn:
                mn = values[j]
        out[i] = mn
    return out


@njit
def rolling_sum(values, window):
    n = len(values)
    out = np.empty(n)
    out[:window - 1] = np.nan
    s = 0.0
    for i in range(window):
        s += values[i]
    out[window - 1] = s
    for i in range(window, n):
        s += values[i] - values[i - window]
        out[i] = s
    return out


@njit
def ewm_mean(values, span):
    n = len(values)
    out = np.empty(n)
    alpha = 2.0 / (span + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit
def pct_change(values, period):
    n = len(values)
    out = np.empty(n)
    out[:period] = np.nan
    for i in range(period, n):
        prev = values[i - period]
        out[i] = (values[i] - prev) / prev if prev != 0.0 else 0.0
    return out


@njit
def diff(values, period):
    n = len(values)
    out = np.empty(n)
    out[:period] = np.nan
    for i in range(period, n):
        out[i] = values[i] - values[i - period]
    return out


@njit
def safe_div(a, b):
    n = len(a)
    out = np.empty(n)
    for i in range(n):
        if b[i] != 0.0 and np.isfinite(b[i]):
            out[i] = a[i] / b[i]
        else:
            out[i] = 0.0
    return out


@njit
def cumsum(values):
    n = len(values)
    out = np.empty(n)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = out[i - 1] + values[i]
    return out


@njit
def true_range(high, low, close):
    """Compute True Range array."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    return tr


@njit
def price_features_fast(open_, high, low, close, volume, windows, n_pos_windows):
    """Compute all price features in one pass.

    Returns dict-like arrays for: atr, atr_pct, ret, vol, vol_surge, pos, body_ratio, upper_wick
    Each is (n_windows, n) shaped for windowed features.
    """
    n = len(close)
    nw = len(windows)

    # TR
    tr = true_range(high, low, close)

    # Pre-allocate all outputs
    atr = np.empty((nw, n))
    atr_pct = np.empty((nw, n))
    ret = np.empty((nw, n))
    vol = np.empty((nw, n))
    vol_surge = np.empty((nw, n))

    # Returns for volatility calc
    rets = np.empty(n)
    rets[0] = 0.0
    for i in range(1, n):
        rets[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0.0

    for wi in range(nw):
        w = windows[wi]
        # ATR (EWM)
        alpha = 2.0 / (w + 1.0)
        atr[wi, 0] = tr[0]
        for i in range(1, n):
            atr[wi, i] = alpha * tr[i] + (1.0 - alpha) * atr[wi, i-1]
        # ATR %
        for i in range(n):
            atr_pct[wi, i] = atr[wi, i] / close[i] if close[i] != 0 else 0.0
        # Returns
        ret[wi, :w] = np.nan
        for i in range(w, n):
            ret[wi, i] = (close[i] - close[i-w]) / close[i-w] if close[i-w] != 0 else 0.0
        # Rolling std of returns (volatility)
        vol[wi, :w-1] = np.nan
        for i in range(w-1, n):
            s = 0.0; s2 = 0.0
            for j in range(i-w+1, i+1):
                s += rets[j]; s2 += rets[j]*rets[j]
            m = s/w
            vol[wi, i] = np.sqrt(max(s2/w - m*m, 0.0))
        # Volume surge
        vol_surge[wi, :w-1] = np.nan
        vm = rolling_mean(volume, w)
        for i in range(n):
            if i < w-1 or vm[i] == 0:
                vol_surge[wi, i] = np.nan
            else:
                vol_surge[wi, i] = volume[i] / vm[i]

    # Body ratio, upper wick
    body_ratio = np.empty(n)
    upper_wick = np.empty(n)
    for i in range(n):
        rng = high[i] - low[i]
        if rng > 0:
            body_ratio[i] = (close[i] - open_[i]) / rng
            upper_wick[i] = (high[i] - max(close[i], open_[i])) / rng
        else:
            body_ratio[i] = 0.0
            upper_wick[i] = 0.0

    return atr, atr_pct, ret, vol, vol_surge, body_ratio, upper_wick
