"""
Grid Entry + Trailing Take-Profit

Grid Entry (거미줄 진입):
  Split position into 3 tranches:
    T1: 40% at signal price (immediate)
    T2: 30% at VWAP pullback (within 15 bars)
    T3: 30% at deeper pullback (-0.1% from signal, within 15 bars)
  If T2/T3 don't fill, remaining size enters at timeout.

Trailing Take-Profit (전략적 익절):
  Phase 1: Hold until profit reaches 1×ATR → activate trailing
  Phase 2: Trail stop at 50% of max unrealized profit
  Phase 3: If profit exceeds 2×ATR → tighten trail to 70% of max

This captures larger moves while protecting gains.
"""

import numpy as np


def simulate_grid_entry(
    entry_bar: int,
    direction: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    hold_bars: int,
    fee: float = 0.0008,
    grid_window: int = 15,
    pullback_pct: float = 0.001,
) -> dict:
    """Simulate grid entry with 3 tranches on 1m data.

    Returns dict with avg_entry_price, fill_ratio, improvement_bps.
    """
    n = len(close)
    signal_price = close[entry_bar]
    end_entry = min(entry_bar + grid_window, n - 1)

    # Grid levels
    if direction == 1:  # long
        levels = [
            signal_price,                          # T1: immediate
            signal_price * (1 - pullback_pct / 2), # T2: small pullback
            signal_price * (1 - pullback_pct),     # T3: deeper pullback
        ]
    else:  # short
        levels = [
            signal_price,
            signal_price * (1 + pullback_pct / 2),
            signal_price * (1 + pullback_pct),
        ]

    weights = [0.4, 0.3, 0.3]
    fills = [signal_price] * 3  # default: all fill at signal
    filled = [True, False, False]

    # Check if T2/T3 fill within window
    for i in range(entry_bar + 1, end_entry + 1):
        if direction == 1:
            if not filled[1] and low[i] <= levels[1]:
                fills[1] = levels[1]
                filled[1] = True
            if not filled[2] and low[i] <= levels[2]:
                fills[2] = levels[2]
                filled[2] = True
        else:
            if not filled[1] and high[i] >= levels[1]:
                fills[1] = levels[1]
                filled[1] = True
            if not filled[2] and high[i] >= levels[2]:
                fills[2] = levels[2]
                filled[2] = True

    # Unfilled tranches enter at timeout price
    timeout_price = close[end_entry]
    for j in range(3):
        if not filled[j]:
            fills[j] = timeout_price

    avg_entry = sum(w * f for w, f in zip(weights, fills))

    if direction == 1:
        improvement = (signal_price - avg_entry) / signal_price * 10000
    else:
        improvement = (avg_entry - signal_price) / signal_price * 10000

    return {
        "avg_entry": avg_entry,
        "signal_price": signal_price,
        "fills": fills,
        "filled": filled,
        "fill_ratio": sum(filled) / 3,
        "improvement_bps": round(improvement, 1),
    }


def simulate_trailing_tp(
    entry_bar: int,
    direction: int,
    entry_price: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    hold_bars: int,
    atr: float,
    fee: float = 0.0008,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    trail_start_atr: float = 1.0,
    trail_pct_1: float = 0.5,
    trail_pct_2: float = 0.7,
    trail_upgrade_atr: float = 2.0,
) -> dict:
    """Simulate trailing take-profit exit on 1m data.

    Phases:
      - Before trail_start_atr × ATR profit: use fixed TP/SL
      - After: trail stop at trail_pct_1 of max profit
      - After trail_upgrade_atr × ATR: tighten to trail_pct_2

    Returns exit details.
    """
    n = len(close)
    if atr <= 0:
        atr = entry_price * 0.005  # fallback 0.5%

    tp_level = entry_price + direction * tp_atr_mult * atr
    sl_level = entry_price - direction * sl_atr_mult * atr
    trail_activate = trail_start_atr * atr
    trail_upgrade = trail_upgrade_atr * atr

    max_profit = 0.0
    trailing_active = False
    trail_pct = trail_pct_1
    exit_price = None
    exit_bar = None
    exit_reason = "timeout"

    end_bar = min(entry_bar + hold_bars, n - 1)

    for i in range(entry_bar + 1, end_bar + 1):
        if direction == 1:
            unrealized = high[i] - entry_price
            adverse = entry_price - low[i]
            current_profit = close[i] - entry_price
        else:
            unrealized = entry_price - low[i]
            adverse = high[i] - entry_price
            current_profit = entry_price - close[i]

        max_profit = max(max_profit, unrealized)

        # Check fixed SL
        if adverse >= sl_atr_mult * atr:
            exit_price = sl_level
            exit_bar = i
            exit_reason = "stop_loss"
            break

        # Check trailing
        if max_profit >= trail_activate:
            trailing_active = True

        if trailing_active:
            # Upgrade trail tightness
            if max_profit >= trail_upgrade:
                trail_pct = trail_pct_2

            # Trail stop level
            trail_stop_dist = max_profit * (1 - trail_pct)
            if direction == 1:
                trail_stop = entry_price + max_profit - trail_stop_dist
            else:
                trail_stop = entry_price - max_profit + trail_stop_dist

            # Check if trail stop hit
            if direction == 1 and low[i] <= trail_stop:
                exit_price = trail_stop
                exit_bar = i
                exit_reason = "trailing_tp"
                break
            elif direction == -1 and high[i] >= trail_stop:
                exit_price = trail_stop
                exit_bar = i
                exit_reason = "trailing_tp"
                break

        # Check fixed TP (only if trailing not active)
        if not trailing_active:
            if direction == 1 and high[i] >= tp_level:
                exit_price = tp_level
                exit_bar = i
                exit_reason = "take_profit"
                break
            elif direction == -1 and low[i] <= tp_level:
                exit_price = tp_level
                exit_bar = i
                exit_reason = "take_profit"
                break

    if exit_price is None:
        exit_price = close[end_bar]
        exit_bar = end_bar

    pnl = direction * (exit_price - entry_price) / entry_price
    net = pnl - fee
    hold_time = exit_bar - entry_bar

    # Compare with simple fixed exit (close at hold_bars)
    simple_exit = close[end_bar]
    simple_pnl = direction * (simple_exit - entry_price) / entry_price - fee

    return {
        "exit_price": exit_price,
        "exit_bar": exit_bar,
        "exit_reason": exit_reason,
        "pnl_pct": round(pnl * 100, 3),
        "net_pct": round(net * 100, 3),
        "hold_time": hold_time,
        "max_profit_pct": round(max_profit / entry_price * 100, 3),
        "trailing_activated": trailing_active,
        "simple_net_pct": round(simple_pnl * 100, 3),
        "improvement_pct": round((net - simple_pnl) * 100, 3),
    }
