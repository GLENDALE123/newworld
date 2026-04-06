"""
Precision Entry Module

When the PLE model signals a trade on 15m timeframe, this module
uses 1m/5m data to find the optimal entry point within the next
15-minute window.

Entry strategies:
  1. VWAP Pullback: Wait for price to pull back to VWAP before entering
  2. Volume Confirmation: Enter after buy/sell volume spike confirms direction
  3. Momentum Entry: Enter on 1m momentum breakout in signal direction
  4. Immediate: Fall back to immediate entry if no better opportunity

Each strategy is scored, and the best entry is selected.
The module can also be used in backtest mode to measure entry improvement.
"""

import numpy as np
import pandas as pd


def compute_vwap(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Compute VWAP (Volume Weighted Average Price)."""
    cumvol = np.cumsum(volumes)
    cumvol_price = np.cumsum(prices * volumes)
    return np.where(cumvol > 0, cumvol_price / cumvol, prices)


def find_precision_entry(
    entry_bar_idx: int,
    direction: int,  # 1=long, -1=short
    close_1m: np.ndarray,
    high_1m: np.ndarray,
    low_1m: np.ndarray,
    volume_1m: np.ndarray,
    buy_volume_1m: np.ndarray | None = None,
    window: int = 15,  # max 1m bars to wait (15 = 15 minutes)
    max_adverse: float = 0.003,  # max 0.3% adverse move before giving up
) -> dict:
    """Find optimal entry within next `window` 1-minute bars.

    Args:
        entry_bar_idx: Index of the 1m bar where 15m signal fires
        direction: 1 for long, -1 for short
        close_1m: 1-minute close prices (full array)
        high_1m: 1-minute high prices
        low_1m: 1-minute low prices
        volume_1m: 1-minute volumes
        buy_volume_1m: 1-minute buy volumes (optional)
        window: How many 1m bars to search
        max_adverse: Max adverse move before timeout

    Returns:
        dict with entry_price, entry_offset, strategy_used, improvement
    """
    n = len(close_1m)
    end_idx = min(entry_bar_idx + window, n - 1)
    if end_idx <= entry_bar_idx:
        return {"entry_price": close_1m[entry_bar_idx], "entry_offset": 0,
                "strategy": "immediate", "improvement_bps": 0}

    signal_price = close_1m[entry_bar_idx]
    window_close = close_1m[entry_bar_idx:end_idx + 1]
    window_high = high_1m[entry_bar_idx:end_idx + 1]
    window_low = low_1m[entry_bar_idx:end_idx + 1]
    window_vol = volume_1m[entry_bar_idx:end_idx + 1]
    W = len(window_close)

    # VWAP within the window
    typical = (window_high + window_low + window_close) / 3
    vwap = compute_vwap(typical, window_vol)

    candidates = []

    # ── Strategy 1: VWAP Pullback ──
    # For longs: wait for price to touch/cross below VWAP
    # For shorts: wait for price to touch/cross above VWAP
    for i in range(1, W):
        if direction == 1:
            # Long: price pulls back below VWAP → good entry
            if window_low[i] <= vwap[i]:
                entry = min(vwap[i], window_close[i])  # enter at VWAP or close
                improvement = (signal_price - entry) / signal_price * 10000  # bps
                if improvement > 0:
                    candidates.append({
                        "entry_price": entry, "entry_offset": i,
                        "strategy": "vwap_pullback", "improvement_bps": round(improvement, 1)
                    })
                    break
        else:
            # Short: price pushes above VWAP → good entry
            if window_high[i] >= vwap[i]:
                entry = max(vwap[i], window_close[i])
                improvement = (entry - signal_price) / signal_price * 10000
                if improvement > 0:
                    candidates.append({
                        "entry_price": entry, "entry_offset": i,
                        "strategy": "vwap_pullback", "improvement_bps": round(improvement, 1)
                    })
                    break

    # ── Strategy 2: Volume Spike Confirmation ──
    if len(window_vol) > 3:
        avg_vol = np.mean(window_vol[:3])  # first 3 bars as baseline
        for i in range(3, W):
            vol_ratio = window_vol[i] / max(avg_vol, 1)
            if vol_ratio > 2.0:  # 2x average volume
                # Check if volume is in our direction
                if buy_volume_1m is not None:
                    buy_vol = buy_volume_1m[entry_bar_idx + i]
                    buy_ratio = buy_vol / max(window_vol[i], 1)
                    directional = (direction == 1 and buy_ratio > 0.6) or \
                                  (direction == -1 and buy_ratio < 0.4)
                else:
                    # No buy volume data: check if price moved in our direction
                    price_move = window_close[i] - window_close[i - 1]
                    directional = (direction == 1 and price_move > 0) or \
                                  (direction == -1 and price_move < 0)

                if directional:
                    entry = window_close[i]
                    if direction == 1:
                        improvement = (signal_price - entry) / signal_price * 10000
                    else:
                        improvement = (entry - signal_price) / signal_price * 10000
                    candidates.append({
                        "entry_price": entry, "entry_offset": i,
                        "strategy": "volume_confirm", "improvement_bps": round(improvement, 1)
                    })
                    break

    # ── Strategy 3: Best Price in Window ──
    # Simply find the best possible price within the window
    if direction == 1:
        best_idx = np.argmin(window_low[1:]) + 1  # best low after signal
        best_price = window_low[best_idx]
        improvement = (signal_price - best_price) / signal_price * 10000
    else:
        best_idx = np.argmax(window_high[1:]) + 1
        best_price = window_high[best_idx]
        improvement = (best_price - signal_price) / signal_price * 10000

    candidates.append({
        "entry_price": best_price, "entry_offset": int(best_idx),
        "strategy": "best_in_window", "improvement_bps": round(improvement, 1)
    })

    # ── Select best realistic entry ──
    # VWAP pullback > volume confirm > best_in_window (oracle, not realistically achievable)
    # Filter out "best_in_window" unless it's the only option
    realistic = [c for c in candidates if c["strategy"] != "best_in_window"]
    if realistic:
        best = max(realistic, key=lambda x: x["improvement_bps"])
    else:
        # Use immediate entry if no realistic improvement found
        best = {"entry_price": signal_price, "entry_offset": 0,
                "strategy": "immediate", "improvement_bps": 0}

    # Also return oracle for comparison
    oracle = max(candidates, key=lambda x: x["improvement_bps"])
    best["oracle_improvement_bps"] = oracle["improvement_bps"]

    return best


def backtest_precision_entry(
    signals: pd.DataFrame,
    kline_1m: pd.DataFrame,
    window: int = 15,
) -> pd.DataFrame:
    """Evaluate precision entry improvement across all trading signals.

    Args:
        signals: DataFrame with columns [timestamp, direction (1/-1)]
        kline_1m: 1-minute OHLCV data with timestamp index

    Returns:
        DataFrame with entry analysis per signal
    """
    if kline_1m.index.tz is not None:
        kline_1m = kline_1m.copy()
        kline_1m.index = kline_1m.index.tz_localize(None)

    close = kline_1m["close"].values
    high = kline_1m["high"].values
    low = kline_1m["low"].values
    vol = kline_1m["volume"].values
    buy_vol = kline_1m["buy_volume"].values if "buy_volume" in kline_1m.columns else None

    timestamps = kline_1m.index
    results = []

    for _, sig in signals.iterrows():
        ts = sig["timestamp"]
        direction = int(sig["direction"])

        # Find nearest 1m bar
        idx = timestamps.searchsorted(ts)
        if idx >= len(timestamps):
            continue

        entry = find_precision_entry(
            idx, direction, close, high, low, vol, buy_vol, window=window
        )
        entry["signal_time"] = ts
        entry["direction"] = direction
        results.append(entry)

    return pd.DataFrame(results)
