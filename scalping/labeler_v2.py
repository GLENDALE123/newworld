"""Scalping labeler v2 — fee-aware, dynamic barriers, asymmetric TP/SL.

Changes from v1:
1. Fee-aware: win/loss는 수수료 차감 후 순수익 기준
2. Dynamic barriers: vol regime별 ATR 배수 자동 조정
   - squeeze → 타이트 배리어 (작은 움직임도 의미 있음)
   - expansion → 넓은 배리어 (노이즈 필터)
3. Asymmetric TP/SL: mean-reversion 방향으로 유리한 비율
4. 3-class label: +1(win), 0(no-trade/chop), -1(loss)
   - 수수료도 못 넘는 움직임 = 0 (모델이 "패스"를 배움)
"""

import numpy as np
import pandas as pd
from numba import njit


@njit
def _compute_atr(high, low, close, period):
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


@njit
def _vol_squeeze_ratio(close, short_w, long_w):
    """Rolling vol ratio (short/long) — squeeze detection."""
    n = len(close)
    ret = np.empty(n)
    ret[0] = 0.0
    for i in range(1, n):
        ret[i] = (close[i] - close[i-1]) / close[i-1]

    ratio = np.empty(n)
    ratio[:] = np.nan

    for i in range(long_w, n):
        # Short window std
        s_sum = 0.0; s_sq = 0.0
        for j in range(i - short_w, i):
            s_sum += ret[j]; s_sq += ret[j] * ret[j]
        s_var = s_sq / short_w - (s_sum / short_w) ** 2
        s_std = s_var ** 0.5 if s_var > 0 else 1e-10

        # Long window std
        l_sum = 0.0; l_sq = 0.0
        for j in range(i - long_w, i):
            l_sum += ret[j]; l_sq += ret[j] * ret[j]
        l_var = l_sq / long_w - (l_sum / long_w) ** 2
        l_std = l_var ** 0.5 if l_var > 0 else 1e-10

        ratio[i] = s_std / l_std

    return ratio


@njit
def _scalp_tbm_v2(closes, highs, lows, atr, vol_ratio,
                   base_tp, base_sl, max_hold, direction, fee_pct):
    """Fee-aware TBM with dynamic barriers.

    Dynamic barrier logic:
    - vol_ratio < 0.7 (squeeze): multiply TP/SL by 0.6 (tighter)
    - vol_ratio 0.7-1.3 (normal): use base TP/SL
    - vol_ratio > 1.3 (expansion): multiply TP/SL by 1.5 (wider)

    3-class label:
    - +1: net P&L after fee > 0
    - -1: net P&L after fee < 0
    -  0: barrier not hit AND |exit P&L| < fee (chop/no-trade)
    """
    n = len(closes)
    label = np.empty(n); label[:] = np.nan
    mae = np.empty(n); mae[:] = np.nan
    mfe = np.empty(n); mfe[:] = np.nan
    bars = np.empty(n); bars[:] = np.nan
    net_pnl = np.empty(n); net_pnl[:] = np.nan

    for i in range(n - 1):
        a = atr[i]
        vr = vol_ratio[i]
        if np.isnan(a) or a <= 0 or np.isnan(vr):
            continue

        entry = closes[i]
        end = min(i + max_hold, n - 1)
        if end <= i:
            continue

        # Dynamic barrier scaling
        if vr < 0.7:
            scale = 0.6
        elif vr > 1.3:
            scale = 1.5
        else:
            scale = 1.0

        tp_dist = base_tp * a * scale
        sl_dist = base_sl * a * scale

        if direction == 1:  # long
            upper = entry + tp_dist
            lower = entry - sl_dist
        else:  # short
            upper = entry + sl_dist
            lower = entry - tp_dist

        # Track MFE/MAE
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
        hit_type = 0  # 0=no hit, 1=TP, -1=SL
        bars_to_hit = max_hold
        if direction == 1:
            for j in range(i + 1, end + 1):
                if highs[j] >= upper:
                    hit_type = 1
                    bars_to_hit = j - i
                    break
                if lows[j] <= lower:
                    hit_type = -1
                    bars_to_hit = j - i
                    break
        else:
            for j in range(i + 1, end + 1):
                if lows[j] <= lower:
                    hit_type = 1
                    bars_to_hit = j - i
                    break
                if highs[j] >= upper:
                    hit_type = -1
                    bars_to_hit = j - i
                    break

        # Compute net P&L
        if hit_type == 1:
            raw_pnl = tp_dist / entry  # TP hit
        elif hit_type == -1:
            raw_pnl = -sl_dist / entry  # SL hit
        else:
            # Time exit
            if direction == 1:
                raw_pnl = (closes[end] - entry) / entry
            else:
                raw_pnl = (entry - closes[end]) / entry

        net = raw_pnl - fee_pct  # fee 차감

        # 3-class label
        if net > 0:
            lab = 1.0
        elif net < -fee_pct:
            lab = -1.0
        else:
            lab = 0.0  # chop — fee 근처, 의미 없는 움직임

        label[i] = lab
        mae[i] = mae_val
        mfe[i] = mfe_val
        bars[i] = bars_to_hit
        net_pnl[i] = net

    return label, mae, mfe, bars, net_pnl


# v2 parameter grid — mean-reversion 기반 비대칭
SCALP_PARAMS_V2 = [
    # (tp_mult, sl_mult, max_hold_bars)
    # 비대칭: TP > SL (mean-reversion은 반전폭이 크고 추세 계속은 제한적)
    (2.0, 1.0, 4),   # standard asymmetric
    (2.5, 1.0, 4),   # wider TP
    (2.0, 1.0, 6),   # longer hold
    (2.5, 1.0, 6),
    (3.0, 1.0, 6),   # wide TP, standard SL
    (2.0, 0.7, 4),   # tight SL (빠른 손절)
    (2.5, 0.7, 6),
    # 대칭 비교용
    (1.5, 1.5, 4),   # symmetric
    (2.0, 2.0, 6),
]


def generate_scalp_labels_v2(
    ohlcv_5m: pd.DataFrame,
    params: list[tuple] | None = None,
    atr_period: int = 24,
    vol_short: int = 20,
    vol_long: int = 60,
    fee: float = 0.0008,  # round-trip fee
) -> pd.DataFrame:
    """Generate v2 scalping labels.

    Returns DataFrame with columns per param:
    - label_{suffix}: +1/0/-1 (win/chop/loss, fee-aware)
    - mae_{suffix}: max adverse excursion
    - mfe_{suffix}: max favorable excursion
    - bars_{suffix}: bars to barrier hit
    - net_{suffix}: net P&L after fee
    """
    if params is None:
        params = SCALP_PARAMS_V2

    c = ohlcv_5m["close"].values.astype(np.float64)
    h = ohlcv_5m["high"].values.astype(np.float64)
    lo = ohlcv_5m["low"].values.astype(np.float64)
    atr = _compute_atr(h, lo, c, atr_period)
    vol_ratio = _vol_squeeze_ratio(c, vol_short, vol_long)

    result = {}
    for tp, sl, hold in params:
        for dir_name, direction in [("long", 1), ("short", -1)]:
            suffix = f"{tp}_{sl}_{hold}_{dir_name}"
            lab, mae, mfe, bars, net = _scalp_tbm_v2(
                c, h, lo, atr, vol_ratio,
                tp, sl, hold, direction, fee
            )
            result[f"label_{suffix}"] = lab
            result[f"mae_{suffix}"] = mae
            result[f"mfe_{suffix}"] = mfe
            result[f"bars_{suffix}"] = bars
            result[f"net_{suffix}"] = net

    return pd.DataFrame(result, index=ohlcv_5m.index)
