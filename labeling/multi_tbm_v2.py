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


# ── ATR Computation ─────────────────────────────────────────────────────────

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """True Range → Exponential Moving Average = ATR."""
    n = len(close)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    # EMA of TR
    atr = np.zeros(n)
    atr[:period] = np.nan
    if period < n:
        atr[period] = np.mean(tr[1:period + 1])
        alpha = 2.0 / (period + 1)
        for i in range(period + 1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


# ── Regime Detection ────────────────────────────────────────────────────────

def detect_regimes(close: pd.Series, atr: np.ndarray, window: int = 24) -> np.ndarray:
    """Classify each bar: surge, dump, range, volatile."""
    n = len(close)
    closes = close.values
    ret = np.zeros(n)
    ret[window:] = (closes[window:] - closes[:-window]) / closes[:-window]

    vol = pd.Series(atr).rolling(window).mean().values
    vol_p50 = pd.Series(atr).rolling(window * 4).median().values
    vol_p90 = pd.Series(atr).rolling(window * 4).quantile(0.9).values

    ret_std = pd.Series(ret).rolling(window * 4).std().values

    regime = np.full(n, "range", dtype=object)
    for i in range(n):
        if np.isnan(vol_p90[i]) or np.isnan(ret_std[i]):
            continue
        if vol[i] > vol_p90[i]:
            regime[i] = "volatile"
        elif ret[i] > ret_std[i]:
            regime[i] = "surge"
        elif ret[i] < -ret_std[i]:
            regime[i] = "dump"
    return regime


# ── Strategy Profiles ───────────────────────────────────────────────────────

STRATEGIES = {
    "scalp": {
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.0,
        "atr_period": 14,
        "max_hold_bars": None,  # set per timeframe
        "source_tf": "5m",
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

# Max holding periods per strategy
HOLD_PERIODS = {
    "scalp": 3,       # 3 bars of source_tf (5m) = 15 min
    "daytrade": 12,   # 12 bars of source_tf (1h) = 12 hours
    "swing": 42,      # 42 bars of source_tf (4h) = 7 days
}

DIRECTIONS = ["long", "short"]
REGIMES = ["surge", "dump", "range", "volatile"]

FEE_PCT = 0.0008  # 0.08% round trip


# ── Single Strategy TBM ────────────────────────────────────────────────────

def _compute_strategy_tbm(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_hold: int,
    direction: int,
    fee_pct: float = FEE_PCT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TBM + MAE + MFE + fee-adjusted RAR using ATR-based barriers."""
    n = len(closes)
    tbm = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    rar = np.full(n, np.nan)

    for i in range(n - 1):
        entry = closes[i]
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        tp_dist = tp_mult * a / entry   # as percentage
        sl_dist = sl_mult * a / entry

        if direction == 1:
            upper = entry + tp_mult * a
            lower = entry - sl_mult * a
        else:
            upper = entry + sl_mult * a  # SL for short
            lower = entry - tp_mult * a  # TP for short

        end = min(i + max_hold, n - 1)
        if end <= i:
            continue

        # Path tracking
        path_h = highs[i + 1:end + 1]
        path_l = lows[i + 1:end + 1]

        if direction == 1:
            mfe_val = (path_h.max() - entry) / entry
            mae_val = (path_l.min() - entry) / entry
        else:
            mfe_val = (entry - path_l.min()) / entry
            mae_val = (entry - path_h.max()) / entry

        # Barrier check
        hit = np.nan
        exit_price = closes[end]

        if direction == 1:
            for j in range(i + 1, end + 1):
                if highs[j] >= upper:
                    hit = 1.0; exit_price = upper; break
                if lows[j] <= lower:
                    hit = -1.0; exit_price = lower; break
        else:
            for j in range(i + 1, end + 1):
                if lows[j] <= lower:
                    hit = 1.0; exit_price = lower; break
                if highs[j] >= upper:
                    hit = -1.0; exit_price = upper; break

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
        sigma = a / entry  # ATR as pct for RAR
        rar_val = net / sigma if sigma > 0 else 0.0

        tbm[i] = hit
        mae[i] = mae_val
        mfe[i] = mfe_val
        rar[i] = rar_val

    return tbm, mae, mfe, rar


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
        DataFrame indexed by finest available timeframe with 96 label columns:
        3 strategies × 2 directions × 4 regimes × 4 types (tbm/mae/mfe/rar)
    """
    label_matrix = {}
    n_combos = len(STRATEGIES) * len(DIRECTIONS) * len(REGIMES)
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

        # Regime detection using this timeframe's data
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

            raw_tbm, raw_mae, raw_mfe, raw_rar = _compute_strategy_tbm(
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

    # Build result per strategy's own timeframe
    # For now, return per-strategy DataFrames keyed by source_tf
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
