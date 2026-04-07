"""
Dynamic TBM — 매 bar에서 최적 TP/SL/hold를 라벨로 생성

기존 TBM: 고정 TP=2×ATR, SL=1×ATR → 32개 binary labels
Dynamic TBM: 매 bar마다 최적 조합 탐색 → regression labels

출력 라벨:
  direction: +1 (long) or -1 (short)
  tp_mult: 최적 TP (ATR 배수, 0.3~3.0)
  sl_mult: 최적 SL (ATR 배수, 0.3~2.0)
  hold_bars: 최적 보유기간 (bars)
  rar: 최적 조합의 risk-adjusted return
"""

import numpy as np
from numba import njit


@njit
def _find_best_tbm(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    fee_pct: float = 0.0008,
) -> tuple:
    """For each bar, find the best TP/SL/hold/direction combination.

    Returns arrays: direction, tp_mult, sl_mult, hold_bars, best_rar
    """
    n = len(closes)

    # Search grid
    tp_mults = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    sl_mults = np.array([0.3, 0.5, 1.0])
    holds = np.array([4, 8, 16, 32])  # 1h, 2h, 4h, 8h in 15m bars

    out_dir = np.zeros(n)
    out_tp = np.zeros(n)
    out_sl = np.zeros(n)
    out_hold = np.zeros(n, dtype=np.int64)
    out_rar = np.full(n, np.nan)

    for i in range(n - 32):  # need space for max hold
        entry = closes[i]
        a = atr[i]
        if np.isnan(a) or a <= 0 or entry <= 0:
            continue

        best_rar = -999.0
        best_dir = 0
        best_tp = 0.0
        best_sl = 0.0
        best_hold = 0

        for direction in (1, -1):
            for tp_idx in range(len(tp_mults)):
                tp_m = tp_mults[tp_idx]
                for sl_idx in range(len(sl_mults)):
                    sl_m = sl_mults[sl_idx]
                    for h_idx in range(len(holds)):
                        max_hold = holds[h_idx]
                        end = min(i + max_hold, n - 1)
                        if end <= i:
                            continue

                        if direction == 1:
                            upper = entry + tp_m * a
                            lower = entry - sl_m * a
                        else:
                            upper = entry + sl_m * a
                            lower = entry - tp_m * a

                        # Simulate
                        exit_price = closes[end]
                        for j in range(i + 1, end + 1):
                            if direction == 1:
                                hit_tp = highs[j] >= upper
                                hit_sl = lows[j] <= lower
                                if hit_tp and hit_sl:
                                    exit_price = lower
                                    break
                                if hit_tp:
                                    exit_price = upper
                                    break
                                if hit_sl:
                                    exit_price = lower
                                    break
                            else:
                                hit_tp = lows[j] <= lower
                                hit_sl = highs[j] >= upper
                                if hit_tp and hit_sl:
                                    exit_price = upper
                                    break
                                if hit_tp:
                                    exit_price = lower
                                    break
                                if hit_sl:
                                    exit_price = upper
                                    break

                        if direction == 1:
                            realized = (exit_price - entry) / entry
                        else:
                            realized = (entry - exit_price) / entry

                        net = realized - fee_pct
                        sigma = a / entry
                        rar = net / sigma if sigma > 0 else 0.0

                        if rar > best_rar:
                            best_rar = rar
                            best_dir = direction
                            best_tp = tp_m
                            best_sl = sl_m
                            best_hold = max_hold

        if best_rar > -999:
            out_dir[i] = best_dir
            out_tp[i] = best_tp
            out_sl[i] = best_sl
            out_hold[i] = best_hold
            out_rar[i] = best_rar

    return out_dir, out_tp, out_sl, out_hold, out_rar


def generate_dynamic_tbm(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    fee_pct: float = 0.0008,
) -> dict[str, np.ndarray]:
    """Generate dynamic TBM labels.

    Returns dict with: direction, tp_mult, sl_mult, hold_bars, rar
    """
    direction, tp, sl, hold, rar = _find_best_tbm(
        high.astype(np.float64),
        low.astype(np.float64),
        close.astype(np.float64),
        atr.astype(np.float64),
        fee_pct,
    )

    return {
        "direction": direction,
        "tp_mult": tp,
        "sl_mult": sl,
        "hold_bars": hold,
        "rar": rar,
    }
