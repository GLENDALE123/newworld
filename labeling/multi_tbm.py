"""
Multi-label TBM Matrix Generator

100 labels = 5 timeframes x 5 risk/reward ratios x 4 market regimes

Timeframes (max holding as tick bar count):
  3m, 5m, 15m, 1h, 4h

Risk/Reward ratios (pt_mult : sl_mult):
  1:1   (pt=1.0, sl=1.0)
  1:2   (pt=2.0, sl=1.0) - wider TP
  2:1   (pt=1.0, sl=2.0) - wider SL (high win rate target)
  tight (pt=0.002/sigma, sl=0.002/sigma) - ~0.2% fixed
  wide  (pt=0.02/sigma, sl=0.02/sigma)  - ~2% fixed

Market regimes:
  surge     - strong uptrend (ret_24h > +1 sigma)
  dump      - strong downtrend (ret_24h < -1 sigma)
  range     - low volatility sideways (vol_24h < median)
  volatile  - high volatility explosion (vol_24h > 90th percentile)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Regime Detection ─────────────────────────────────────────────────────────

def detect_regimes(
    close: pd.Series,
    window: int = 24,
    vol_window: int = 24,
) -> pd.Series:
    """Classify each bar into a market regime.

    Returns Series with values: 'surge', 'dump', 'range', 'volatile'
    """
    ret = close.pct_change(window)
    vol = close.pct_change().rolling(vol_window).std()

    ret_std = ret.rolling(window * 4).std()
    vol_median = vol.rolling(window * 4).median()
    vol_p90 = vol.rolling(window * 4).quantile(0.9)

    regime = pd.Series("range", index=close.index)

    # Volatile first (takes priority)
    regime[vol > vol_p90] = "volatile"
    # Surge/dump override range but not volatile
    regime[(ret > ret_std) & (regime != "volatile")] = "surge"
    regime[(ret < -ret_std) & (regime != "volatile")] = "dump"

    return regime


# ── TBM Configuration ────────────────────────────────────────────────────────

TIMEFRAMES = {
    "3m":  {"minutes": 3},
    "5m":  {"minutes": 5},
    "15m": {"minutes": 15},
    "1h":  {"minutes": 60},
    "4h":  {"minutes": 240},
}

RR_RATIOS = {
    "1to1":  {"pt_mult": 1.0,  "sl_mult": 1.0},
    "1to2":  {"pt_mult": 2.0,  "sl_mult": 1.0},
    "2to1":  {"pt_mult": 1.0,  "sl_mult": 2.0},
    "tight": {"fixed_pct": 0.002},
    "wide":  {"fixed_pct": 0.02},
}

REGIMES = ["surge", "dump", "range", "volatile"]


def _estimate_holding_bars(tick_bar_df: pd.DataFrame, minutes: int) -> int:
    """Estimate how many tick bars correspond to N minutes."""
    median_interval = tick_bar_df["timestamp"].diff().dt.total_seconds().median()
    if median_interval <= 0:
        median_interval = 20.0
    bars = int((minutes * 60) / median_interval)
    return max(1, bars)


# ── Single TBM Label ─────────────────────────────────────────────────────────

def _compute_single_tbm(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volatility: np.ndarray,
    max_holding: int,
    pt_mult: float | None = None,
    sl_mult: float | None = None,
    fixed_pct: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TBM + MAE + MFE + Risk-Adjusted Return for a single parameter set.

    Returns:
        (tbm_labels, mae, mfe, rar) each np.ndarray of length n
        - tbm: +1/-1 direction
        - mae: max adverse excursion (worst drawdown during trade, as pct, always <= 0)
        - mfe: max favorable excursion (best unrealized profit, as pct, always >= 0)
        - rar: risk-adjusted return = realized_return / sigma_t
    """
    n = len(closes)
    tbm = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    rar = np.full(n, np.nan)

    for i in range(n - 1):
        entry = closes[i]
        sigma = volatility[i]

        if np.isnan(sigma) or sigma <= 0:
            continue

        # Determine barriers
        if fixed_pct is not None:
            upper = entry * (1 + fixed_pct)
            lower = entry * (1 - fixed_pct)
        else:
            upper = entry * (1 + pt_mult * sigma)
            lower = entry * (1 - sl_mult * sigma)

        end = min(i + max_holding, n - 1)
        if end <= i:
            continue

        # Track path: MAE and MFE
        path_highs = highs[i + 1:end + 1]
        path_lows = lows[i + 1:end + 1]

        max_price = path_highs.max()
        min_price = path_lows.min()

        mfe_val = (max_price - entry) / entry   # best unrealized profit
        mae_val = (min_price - entry) / entry    # worst unrealized loss

        # Check barriers (which hit first)
        hit = np.nan
        exit_price = closes[end]
        for j in range(i + 1, end + 1):
            if highs[j] >= upper:
                hit = 1.0
                exit_price = upper
                break
            if lows[j] <= lower:
                hit = -1.0
                exit_price = lower
                break

        if np.isnan(hit):
            pnl = closes[end] - entry
            hit = 1.0 if pnl > 0 else -1.0
            exit_price = closes[end]

        realized_ret = (exit_price - entry) / entry
        risk_adj = realized_ret / sigma if sigma > 0 else 0.0

        tbm[i] = hit
        mae[i] = mae_val
        mfe[i] = mfe_val
        rar[i] = risk_adj

    return tbm, mae, mfe, rar


# ── Multi-Label Matrix ───────────────────────────────────────────────────────

def generate_multi_tbm(
    tick_bar_df: pd.DataFrame,
    vol_span: int = 500,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate 400-label matrix from tick bar data.

    100 combinations x 4 label types:
      - tbm_{tf}_{rr}_{regime}: direction (+1/-1)
      - mae_{tf}_{rr}_{regime}: max adverse excursion (worst drawdown, <= 0)
      - mfe_{tf}_{rr}_{regime}: max favorable excursion (best profit, >= 0)
      - rar_{tf}_{rr}_{regime}: risk-adjusted return (return / volatility)

    Args:
        tick_bar_df: DataFrame with [timestamp, open, high, low, close, volume].
        vol_span: EWM span for volatility.
        progress: Print progress.

    Returns:
        DataFrame with 400 columns. NaN where regime doesn't match or insufficient data.
    """
    df = tick_bar_df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()

    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)

    ret = pd.Series(closes).pct_change()
    volatility = ret.ewm(span=vol_span).std().values

    close_series = pd.Series(closes, index=df.index)
    bars_per_hour = _estimate_holding_bars(tick_bar_df, 60)
    regimes = detect_regimes(close_series, window=bars_per_hour, vol_window=bars_per_hour)
    regime_values = regimes.values

    holding_bars = {}
    for tf_name, tf_cfg in TIMEFRAMES.items():
        holding_bars[tf_name] = _estimate_holding_bars(tick_bar_df, tf_cfg["minutes"])

    if progress:
        print(f"  Tick bars: {len(df)}, Bars/hour: {bars_per_hour}")
        for tf, hb in holding_bars.items():
            print(f"    {tf}: {hb} bars holding")

    label_matrix = {}
    total_combos = len(TIMEFRAMES) * len(RR_RATIOS)
    done = 0

    for tf_name in TIMEFRAMES:
        max_hold = holding_bars[tf_name]

        for rr_name, rr_cfg in RR_RATIOS.items():
            if "fixed_pct" in rr_cfg:
                raw_tbm, raw_mae, raw_mfe, raw_rar = _compute_single_tbm(
                    highs, lows, closes, volatility, max_hold,
                    fixed_pct=rr_cfg["fixed_pct"],
                )
            else:
                raw_tbm, raw_mae, raw_mfe, raw_rar = _compute_single_tbm(
                    highs, lows, closes, volatility, max_hold,
                    pt_mult=rr_cfg["pt_mult"],
                    sl_mult=rr_cfg["sl_mult"],
                )

            for regime in REGIMES:
                regime_mask = regime_values == regime
                base = f"{tf_name}_{rr_name}_{regime}"

                label_matrix[f"tbm_{base}"] = np.where(regime_mask, raw_tbm, np.nan)
                label_matrix[f"mae_{base}"] = np.where(regime_mask, raw_mae, np.nan)
                label_matrix[f"mfe_{base}"] = np.where(regime_mask, raw_mfe, np.nan)
                label_matrix[f"rar_{base}"] = np.where(regime_mask, raw_rar, np.nan)

            done += 1
            if progress and done % 5 == 0:
                print(f"  [{done}/{total_combos}] combos done...")

    result = pd.DataFrame(label_matrix, index=df.index)

    if progress:
        tbm_cols = [c for c in result.columns if c.startswith("tbm_")]
        rar_cols = [c for c in result.columns if c.startswith("rar_")]
        n_valid = result[tbm_cols].notna().sum()
        print(f"\n  Label matrix: {result.shape[0]} rows x {result.shape[1]} cols")
        print(f"  TBM valid/col (mean): {n_valid.mean():.0f}")
        pos = (result[tbm_cols] == 1).sum().mean()
        neg = (result[tbm_cols] == -1).sum().mean()
        if pos + neg > 0:
            print(f"  TBM balance: +1={pos:.0f} -1={neg:.0f} ({pos/(pos+neg):.1%})")
        rar_mean = result[rar_cols].mean().mean()
        print(f"  RAR mean: {rar_mean:.4f}")
        mae_cols = [c for c in result.columns if c.startswith("mae_")]
        mfe_cols = [c for c in result.columns if c.startswith("mfe_")]
        print(f"  MAE mean: {result[mae_cols].mean().mean():.4f}")
        print(f"  MFE mean: {result[mfe_cols].mean().mean():.4f}")

    return result
