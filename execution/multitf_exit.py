"""
Multi-TF Consensus Exit

Instead of trailing stop on 1m (too noisy), use higher-TF signals:

Resolution hierarchy (해상도를 올리면서 판단):
  1h:  Overall trend intact? (SMA20 direction)
  15m: Momentum reversal? (3-bar momentum flip)
  5m:  Confirmation (price crosses 5m SMA10)

Exit rules:
  - Hold at least min_hold bars (avoid premature exit)
  - After min_hold: exit if 15m momentum reverses AND 5m confirms
  - Always exit at max_hold (backstop)
  - Stop loss: still use ATR-based fixed SL (prevents catastrophic loss)

This captures trend continuation while protecting against reversals
at a resolution that filters out 1m noise.
"""

import numpy as np
import pandas as pd


def simulate_multitf_exit(
    entry_bar_15m: int,
    direction: int,
    entry_price: float,
    close_5m: np.ndarray,
    close_15m: np.ndarray,
    high_15m: np.ndarray,
    low_15m: np.ndarray,
    ts_5m: np.ndarray,   # timestamps for 5m
    ts_15m: np.ndarray,  # timestamps for 15m
    max_hold_15m: int,
    atr: float,
    sl_atr_mult: float = 1.0,
    min_hold_15m: int = 2,
    fee: float = 0.0008,
) -> dict:
    """Simulate exit using multi-TF consensus.

    Args:
        entry_bar_15m: Index in 15m arrays where trade starts
        direction: 1=long, -1=short
        entry_price: Entry price
        close_5m: Full 5m close array
        close_15m: Full 15m close array
        high_15m, low_15m: Full 15m high/low arrays
        ts_5m, ts_15m: Timestamp arrays for index alignment
        max_hold_15m: Maximum bars to hold (in 15m bars)
        atr: ATR at entry for stop loss
        sl_atr_mult: Stop loss multiplier
        min_hold_15m: Minimum bars before exit allowed
        fee: Trading fee

    Returns:
        dict with exit details
    """
    n15 = len(close_15m)
    sl_level = entry_price - direction * sl_atr_mult * atr

    end_bar = min(entry_bar_15m + max_hold_15m, n15 - 1)
    exit_price = None
    exit_bar = None
    exit_reason = "timeout"

    # Pre-compute 5m SMA10 aligned to 15m timestamps
    # Each 15m bar corresponds to 3 5m bars
    sma10_5m = pd.Series(close_5m).rolling(10).mean().values

    # Pre-compute 15m momentum (3-bar rate of change)
    mom_15m = np.zeros(n15)
    mom_15m[3:] = close_15m[3:] - close_15m[:-3]

    for i in range(entry_bar_15m + 1, end_bar + 1):
        # Check stop loss first (always active)
        if direction == 1:
            if low_15m[i] <= sl_level:
                exit_price = sl_level
                exit_bar = i
                exit_reason = "stop_loss"
                break
        else:
            if high_15m[i] >= sl_level:
                exit_price = sl_level
                exit_bar = i
                exit_reason = "stop_loss"
                break

        # Don't check early exit before min_hold
        if i - entry_bar_15m < min_hold_15m:
            continue

        # Must be in profit to consider early exit
        unrealized = direction * (close_15m[i] - entry_price) / entry_price
        if unrealized <= 0:
            continue

        # ── Multi-TF consensus check ──

        # 15m: momentum reversal (direction flip in last 3 bars)
        if direction == 1:
            mom_reversed = mom_15m[i] < 0 and mom_15m[i-1] >= 0
        else:
            mom_reversed = mom_15m[i] > 0 and mom_15m[i-1] <= 0

        if not mom_reversed:
            continue

        # 5m: price below/above SMA10 (confirmation)
        # Find corresponding 5m bar index
        ts_current = ts_15m[i]
        idx_5m = np.searchsorted(ts_5m, ts_current)
        if idx_5m >= len(close_5m):
            idx_5m = len(close_5m) - 1

        if direction == 1:
            confirmed_5m = close_5m[idx_5m] < sma10_5m[idx_5m]
        else:
            confirmed_5m = close_5m[idx_5m] > sma10_5m[idx_5m]

        if not confirmed_5m:
            continue

        # Both conditions met → exit
        exit_price = close_15m[i]
        exit_bar = i
        exit_reason = "multitf_reversal"
        break

    if exit_price is None:
        exit_price = close_15m[end_bar]
        exit_bar = end_bar

    pnl = direction * (exit_price - entry_price) / entry_price
    net = pnl - fee
    hold_time = exit_bar - entry_bar_15m

    # Compare with fixed exit
    simple_exit = close_15m[end_bar]
    simple_pnl = direction * (simple_exit - entry_price) / entry_price - fee

    return {
        "exit_price": exit_price,
        "exit_bar": exit_bar,
        "exit_reason": exit_reason,
        "net_pct": round(net * 100, 3),
        "hold_time_15m": hold_time,
        "simple_net_pct": round(simple_pnl * 100, 3),
        "improvement_pct": round((net - simple_pnl) * 100, 3),
    }
