"""Scalping labeler v3 — optimal action label.

매 바마다 hindsight optimal action을 역산:
  +1 = LONG (forward MFE long > fee, and better than short)
  -1 = SHORT (forward MFE short > fee, and better than long)
   0 = HOLD (둘 다 fee 못 넘음, 또는 MAE가 MFE보다 큼)

핵심: TP/SL 파라미터 없음. 순수하게 미래 price path에서 최적 행동 역산.
"""

import numpy as np
import pandas as pd
from numba import njit


@njit
def _optimal_action(closes, highs, lows, max_hold, fee_pct, min_rr):
    """Compute optimal action label for each bar.

    Args:
        closes, highs, lows: price arrays
        max_hold: max bars to look forward
        fee_pct: round-trip fee (e.g. 0.0008)
        min_rr: minimum reward/risk ratio to qualify as opportunity
                (MFE / MAE must exceed this)

    Returns:
        action: +1/0/-1 array
        mfe_long, mae_long: MFE/MAE for long direction
        mfe_short, mae_short: MFE/MAE for short direction
        edge_long, edge_short: net edge (MFE - MAE - fee) per direction
        bars_to_peak_long, bars_to_peak_short: bars to MFE peak
    """
    n = len(closes)
    action = np.zeros(n)
    mfe_long = np.empty(n); mfe_long[:] = np.nan
    mae_long = np.empty(n); mae_long[:] = np.nan
    mfe_short = np.empty(n); mfe_short[:] = np.nan
    mae_short = np.empty(n); mae_short[:] = np.nan
    edge_long = np.empty(n); edge_long[:] = np.nan
    edge_short = np.empty(n); edge_short[:] = np.nan
    bars_peak_long = np.empty(n); bars_peak_long[:] = np.nan
    bars_peak_short = np.empty(n); bars_peak_short[:] = np.nan

    for i in range(n - max_hold):
        entry = closes[i]
        if entry <= 0:
            continue

        # Scan forward path
        best_high = highs[i + 1]
        worst_low = lows[i + 1]
        bar_best_high = 1
        bar_worst_low = 1

        # Also track: worst before best (for realistic MAE)
        # For long: MAE = max drawdown before MFE peak
        # For short: MAE = max adverse rally before MFE trough

        path_highs = np.empty(max_hold)
        path_lows = np.empty(max_hold)

        for j in range(max_hold):
            idx = i + 1 + j
            if idx >= n:
                path_highs[j] = path_highs[j-1] if j > 0 else entry
                path_lows[j] = path_lows[j-1] if j > 0 else entry
                continue
            path_highs[j] = highs[idx]
            path_lows[j] = lows[idx]

        # Long MFE: best high in window
        cummax_high = path_highs[0]
        bar_mfe_l = 0
        for j in range(1, max_hold):
            if path_highs[j] > cummax_high:
                cummax_high = path_highs[j]
                bar_mfe_l = j

        # Long MAE: worst low BEFORE MFE peak (realistic: you'd exit at peak)
        cummin_low_before_peak = path_lows[0]
        for j in range(1, bar_mfe_l + 1):
            if path_lows[j] < cummin_low_before_peak:
                cummin_low_before_peak = path_lows[j]

        mfe_l = (cummax_high - entry) / entry
        mae_l = (entry - cummin_low_before_peak) / entry
        if mae_l < 0:
            mae_l = 0.0

        # Short MFE: best low in window
        cummin_low = path_lows[0]
        bar_mfe_s = 0
        for j in range(1, max_hold):
            if path_lows[j] < cummin_low:
                cummin_low = path_lows[j]
                bar_mfe_s = j

        # Short MAE: worst high BEFORE MFE trough
        cummax_high_before_trough = path_highs[0]
        for j in range(1, bar_mfe_s + 1):
            if path_highs[j] > cummax_high_before_trough:
                cummax_high_before_trough = path_highs[j]

        mfe_s = (entry - cummin_low) / entry
        mae_s = (cummax_high_before_trough - entry) / entry
        if mae_s < 0:
            mae_s = 0.0

        # Net edge = MFE - fee (이게 양수여야 기회)
        # Risk-adjusted: MFE - MAE - fee
        el = mfe_l - mae_l - fee_pct
        es = mfe_s - mae_s - fee_pct

        mfe_long[i] = mfe_l
        mae_long[i] = mae_l
        mfe_short[i] = mfe_s
        mae_short[i] = mae_s
        edge_long[i] = el
        edge_short[i] = es
        bars_peak_long[i] = bar_mfe_l + 1
        bars_peak_short[i] = bar_mfe_s + 1

        # Decision: optimal action
        long_ok = (mfe_l > fee_pct) and (el > 0) and (mfe_l / (mae_l + 1e-10) > min_rr)
        short_ok = (mfe_s > fee_pct) and (es > 0) and (mfe_s / (mae_s + 1e-10) > min_rr)

        if long_ok and short_ok:
            # Both viable → pick better edge
            if el >= es:
                action[i] = 1.0
            else:
                action[i] = -1.0
        elif long_ok:
            action[i] = 1.0
        elif short_ok:
            action[i] = -1.0
        else:
            action[i] = 0.0  # HOLD

    return action, mfe_long, mae_long, mfe_short, mae_short, edge_long, edge_short, bars_peak_long, bars_peak_short


def generate_scalp_labels_v3(
    ohlcv_5m: pd.DataFrame,
    max_holds: list[int] | None = None,
    fee: float = 0.0008,
    min_rr: float = 1.5,
) -> pd.DataFrame:
    """Generate v3 optimal action labels.

    Args:
        ohlcv_5m: 5m OHLCV
        max_holds: list of forward windows to test (bars)
        fee: round-trip fee
        min_rr: minimum MFE/MAE ratio to qualify

    Returns:
        DataFrame with per-window columns:
        - action_{h}: +1/0/-1 (long/hold/short)
        - mfe_long_{h}, mae_long_{h}: long direction MFE/MAE
        - mfe_short_{h}, mae_short_{h}: short direction MFE/MAE
        - edge_long_{h}, edge_short_{h}: net edge per direction
        - bars_peak_long_{h}, bars_peak_short_{h}: bars to MFE peak
    """
    if max_holds is None:
        max_holds = [3, 4, 6, 8, 12]  # 15min, 20min, 30min, 40min, 1h

    c = ohlcv_5m["close"].values.astype(np.float64)
    h = ohlcv_5m["high"].values.astype(np.float64)
    lo = ohlcv_5m["low"].values.astype(np.float64)

    result = {}
    for hold in max_holds:
        act, mfe_l, mae_l, mfe_s, mae_s, el, es, bp_l, bp_s = _optimal_action(
            c, h, lo, hold, fee, min_rr
        )
        tag = str(hold)
        result[f"action_{tag}"] = act
        result[f"mfe_long_{tag}"] = mfe_l
        result[f"mae_long_{tag}"] = mae_l
        result[f"mfe_short_{tag}"] = mfe_s
        result[f"mae_short_{tag}"] = mae_s
        result[f"edge_long_{tag}"] = el
        result[f"edge_short_{tag}"] = es
        result[f"bars_peak_long_{tag}"] = bp_l
        result[f"bars_peak_short_{tag}"] = bp_s

    return pd.DataFrame(result, index=ohlcv_5m.index)
