"""
Multi-Label TBM v2: ATR-based Dynamic Barriers + Realistic Strategy Profiles

3 strategy styles × 2 directions × 4 regimes = 24 core labels × 4 types = 96 labels

Strategy profiles (ATR-based):
  Scalping:     TP = 2×ATR(5m),   SL = 1×ATR(5m),   T = 15min
  Day Trading:  TP = 2×ATR(1h),   SL = 1×ATR(1h),   T = 4h~12h
  Swing:        TP = 3×ATR(4h),   SL = 1.5×ATR(4h), T = 2~7 days

Each uses ATR at its OWN timeframe, not a global volatility.
"""

import numpy as np
import pandas as pd
from numba import njit


# ── ATR Computation (numba) ─────────────────────────────────────────────────

@njit
def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
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


# ── Regime Detection ────────────────────────────────────────────────────────

def detect_regimes(close: pd.Series, atr: np.ndarray, window: int = 24) -> np.ndarray:
    """Classify each bar: surge, dump, range, volatile. Fully vectorized."""
    n = len(close)
    closes = close.values
    ret = np.zeros(n)
    ret[window:] = (closes[window:] - closes[:-window]) / closes[:-window]

    atr_s = pd.Series(atr)
    vol = atr_s.rolling(window).mean().values
    vol_p90 = atr_s.rolling(window * 4).quantile(0.9).values
    ret_std = pd.Series(ret).rolling(window * 4).std().values

    valid = ~(np.isnan(vol_p90) | np.isnan(ret_std))
    regime = np.full(n, "range", dtype=object)
    regime[valid & (vol > vol_p90)] = "volatile"
    # surge/dump only where not already volatile
    not_vol = valid & (vol <= vol_p90)
    regime[not_vol & (ret > ret_std)] = "surge"
    regime[not_vol & (ret < -ret_std)] = "dump"
    return regime


# ── Strategy Profiles ───────────────────────────────────────────────────────

STRATEGIES = {
    "scalp": {
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.0,
        "atr_period": 14,
        "source_tf": "5m",
    },
    "intraday": {
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.0,
        "atr_period": 14,
        "source_tf": "15m",
    },
    "daytrade": {
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.0,
        "atr_period": 14,
        "source_tf": "1h",
    },
    "swing": {
        "tp_atr_mult": 3.0,
        "sl_atr_mult": 1.5,
        "atr_period": 14,
        "source_tf": "4h",
    },
}

HOLD_PERIODS = {
    "scalp": 3,       # 3 bars × 5m  = 15 min
    "intraday": 4,    # 4 bars × 15m = 1 hour
    "daytrade": 12,   # 12 bars × 1h = 12 hours
    "swing": 42,      # 42 bars × 4h = 7 days
}

DIRECTIONS = ["long", "short"]
REGIMES = ["surge", "dump", "range", "volatile"]
REGIMES_V3 = ["surge", "dump", "range", "volatile", "deleverage"]  # with OI divergence

FEE_PCT = 0.0008  # 0.08% round trip


def detect_oi_divergence(
    close: pd.Series,
    oi: pd.Series,
    lookback: int = 96,  # 1 day at 15m
    oi_threshold: float = 0.05,
    price_threshold: float = 0.02,
) -> np.ndarray:
    """Detect OI-Price divergence regime.

    deleverage_bull: OI↓ + Price↑ → trend likely continues (+0.71% next day)
    deleverage_bear: OI↑ + Price↓ → shorts accumulating

    Returns array of booleans (True = deleverage event).
    """
    oi_chg = oi.pct_change(lookback).values
    price_chg = close.pct_change(lookback).values
    n = len(close)

    is_deleverage = np.zeros(n, dtype=bool)
    # OI dropping + price rising = bullish deleverage
    is_deleverage |= (oi_chg < -oi_threshold) & (price_chg > price_threshold)
    # OI rising + price dropping = bearish leverage buildup
    is_deleverage |= (oi_chg > oi_threshold) & (price_chg < -price_threshold)

    return is_deleverage


# ── Single Strategy TBM (numba) ───────────────────────────────────────────

@njit
def _compute_strategy_tbm(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_hold: int,
    direction: int,
    fee_pct: float,
) -> tuple:
    n = len(closes)
    tbm = np.empty(n)
    mae = np.empty(n)
    mfe = np.empty(n)
    rar = np.empty(n)
    weight = np.empty(n)
    tbm[:] = np.nan
    mae[:] = np.nan
    mfe[:] = np.nan
    rar[:] = np.nan
    weight[:] = np.nan

    for i in range(n - 1):
        entry = closes[i]
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        if direction == 1:
            upper = entry + tp_mult * a
            lower = entry - sl_mult * a
        else:
            upper = entry + sl_mult * a
            lower = entry - tp_mult * a

        end = min(i + max_hold, n - 1)
        if end <= i:
            continue

        # MFE / MAE over the path
        path_h_max = highs[i + 1]
        path_l_min = lows[i + 1]
        for j in range(i + 2, end + 1):
            if highs[j] > path_h_max:
                path_h_max = highs[j]
            if lows[j] < path_l_min:
                path_l_min = lows[j]

        if direction == 1:
            mfe_val = (path_h_max - entry) / entry
            mae_val = (path_l_min - entry) / entry
        else:
            mfe_val = (entry - path_l_min) / entry
            mae_val = (entry - path_h_max) / entry

        hit = np.nan
        exit_price = closes[end]
        bars_to_hit = max_hold

        if direction == 1:
            for j in range(i + 1, end + 1):
                if highs[j] >= upper:
                    hit = 1.0
                    exit_price = upper
                    bars_to_hit = j - i
                    break
                if lows[j] <= lower:
                    hit = -1.0
                    exit_price = lower
                    bars_to_hit = j - i
                    break
        else:
            for j in range(i + 1, end + 1):
                if lows[j] <= lower:
                    hit = 1.0
                    exit_price = lower
                    bars_to_hit = j - i
                    break
                if highs[j] >= upper:
                    hit = -1.0
                    exit_price = upper
                    bars_to_hit = j - i
                    break

        if np.isnan(hit):
            if direction == 1:
                pnl = closes[end] - entry
            else:
                pnl = entry - closes[end]
            hit = 1.0 if pnl > 0 else -1.0
            exit_price = closes[end]

        if direction == 1:
            realized = (exit_price - entry) / entry
        else:
            realized = (entry - exit_price) / entry

        net = realized - fee_pct
        sigma = a / entry
        rar_val = net / sigma if sigma > 0 else 0.0

        magnitude_w = abs(net) / max(tp_mult * sigma, 1e-8)
        speed_w = (max_hold - bars_to_hit) / max_hold
        mag_clamp = magnitude_w if magnitude_w > 0 else 0.0
        spd_clamp = speed_w if speed_w > 0 else 0.0
        sample_w = np.sqrt(mag_clamp * spd_clamp)

        if hit == 1.0 and net > 0:
            sample_w = 1.0 + sample_w
        else:
            sample_w = 1.0

        tbm[i] = hit
        mae[i] = mae_val
        mfe[i] = mfe_val
        rar[i] = rar_val
        weight[i] = sample_w

    return tbm, mae, mfe, rar, weight


# ── Multi-Label Generator ──────────────────────────────────────────────────

def generate_multi_tbm_v2(
    kline_data: dict[str, pd.DataFrame],
    fee_pct: float = FEE_PCT,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate multi-label TBM matrix from multi-timeframe kline data.

    Args:
        kline_data: {"5m": df, "1h": df, "4h": df} each with timestamp index, OHLCV columns
        fee_pct: Round-trip trading fee
        progress: Print progress

    Returns:
        Dict of DataFrames per strategy timeframe with label columns.
    """
    label_matrix = {}
    done = 0

    for strat_name, strat_cfg in STRATEGIES.items():
        source_tf = strat_cfg["source_tf"]
        if source_tf not in kline_data:
            if progress:
                print(f"  Skipping {strat_name}: {source_tf} data not available")
            continue

        df = kline_data[source_tf].copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()

        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)

        atr = compute_atr(h, l, c, period=strat_cfg["atr_period"])
        max_hold = HOLD_PERIODS[strat_name]

        regimes = detect_regimes(df["close"], atr, window=max(24, max_hold))

        if progress:
            valid_atr = np.sum(~np.isnan(atr) & (atr > 0))
            avg_atr_pct = np.nanmean(atr / c) * 100
            print(f"\n  {strat_name} ({source_tf}): {len(df)} bars, "
                  f"ATR valid={valid_atr}, avg ATR={avg_atr_pct:.3f}%, "
                  f"TP={strat_cfg['tp_atr_mult']}×ATR, SL={strat_cfg['sl_atr_mult']}×ATR, "
                  f"hold={max_hold} bars")

        for dir_name in DIRECTIONS:
            direction = 1 if dir_name == "long" else -1

            raw_tbm, raw_mae, raw_mfe, raw_rar, raw_weight = _compute_strategy_tbm(
                h, l, c, atr,
                tp_mult=strat_cfg["tp_atr_mult"],
                sl_mult=strat_cfg["sl_atr_mult"],
                max_hold=max_hold,
                direction=direction,
                fee_pct=fee_pct,
            )

            for regime in REGIMES:
                mask = regimes == regime
                base = f"{strat_name}_{dir_name}_{regime}"

                label_matrix[f"tbm_{base}"] = np.where(mask, raw_tbm, np.nan)
                label_matrix[f"mae_{base}"] = np.where(mask, raw_mae, np.nan)
                label_matrix[f"mfe_{base}"] = np.where(mask, raw_mfe, np.nan)
                label_matrix[f"rar_{base}"] = np.where(mask, raw_rar, np.nan)
                label_matrix[f"wgt_{base}"] = np.where(mask, raw_weight, np.nan)

                done += 1

        if progress:
            tbm_cols_strat = [k for k in label_matrix if k.startswith(f"tbm_{strat_name}")]
            vals = np.array([label_matrix[k] for k in tbm_cols_strat])
            pos = np.nansum(vals == 1)
            neg = np.nansum(vals == -1)
            total = pos + neg
            rar_cols_strat = [k for k in label_matrix if k.startswith(f"rar_{strat_name}")]
            rar_vals = np.array([label_matrix[k] for k in rar_cols_strat])
            pct_positive = np.nansum(rar_vals > 0) / max(np.sum(~np.isnan(rar_vals)), 1) * 100
            print(f"    TBM: +1={pos} -1={neg} ({pos/max(total,1)*100:.1f}% win)")
            print(f"    RAR>0 (after fees): {pct_positive:.1f}%")

    results = {}
    for strat_name, strat_cfg in STRATEGIES.items():
        source_tf = strat_cfg["source_tf"]
        if source_tf not in kline_data:
            continue
        df = kline_data[source_tf]
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()

        cols = {k: v for k, v in label_matrix.items() if strat_name in k}
        results[strat_name] = pd.DataFrame(cols, index=df.index)

    if progress:
        total_labels = sum(r.shape[1] for r in results.values())
        print(f"\n  Total: {total_labels} label columns across {len(results)} strategy timeframes")

    return results
