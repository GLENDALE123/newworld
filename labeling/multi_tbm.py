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
) -> np.ndarray:
    """Compute TBM labels for a single parameter set. Vectorized inner loop."""
    n = len(closes)
    labels = np.full(n, np.nan)

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

        # Check barriers
        hit = np.nan
        for j in range(i + 1, end + 1):
            if highs[j] >= upper:
                hit = 1.0
                break
            if lows[j] <= lower:
                hit = -1.0
                break

        if np.isnan(hit):
            # Vertical barrier: use close at end
            pnl = closes[end] - entry
            hit = 1.0 if pnl > 0 else -1.0

        labels[i] = hit

    return labels


# ── Multi-Label Matrix ───────────────────────────────────────────────────────

def generate_multi_tbm(
    tick_bar_df: pd.DataFrame,
    vol_span: int = 500,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate 100-label TBM matrix from tick bar data.

    Args:
        tick_bar_df: DataFrame with columns [timestamp, open, high, low, close, volume]
                     Must be sorted by timestamp.
        vol_span: EWM span for volatility calculation (in tick bars).
        progress: Print progress.

    Returns:
        DataFrame with tick bar index and 100 label columns.
        Column names: tbm_{timeframe}_{rr}_{regime}
        Values: +1 (take profit hit), -1 (stop loss hit), NaN (insufficient data)
    """
    df = tick_bar_df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp").sort_index()

    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)

    # Volatility (EWM std of returns)
    ret = pd.Series(closes).pct_change()
    volatility = ret.ewm(span=vol_span).std().values

    # Detect regimes (using close prices resampled to ~1h for stability)
    close_series = pd.Series(closes, index=df.index)
    # Use tick-bar based window: ~180 bars ≈ 1 hour at 20sec/bar
    bars_per_hour = _estimate_holding_bars(tick_bar_df, 60)
    regimes = detect_regimes(close_series, window=bars_per_hour, vol_window=bars_per_hour)
    regime_values = regimes.values

    # Pre-compute holding bars for each timeframe
    holding_bars = {}
    for tf_name, tf_cfg in TIMEFRAMES.items():
        holding_bars[tf_name] = _estimate_holding_bars(tick_bar_df, tf_cfg["minutes"])

    if progress:
        print(f"  Tick bars: {len(df)}")
        print(f"  Bars/hour estimate: {bars_per_hour}")
        for tf, hb in holding_bars.items():
            print(f"  {tf}: {hb} bars holding")

    # Generate all 100 labels
    label_matrix = {}
    total = len(TIMEFRAMES) * len(RR_RATIOS) * len(REGIMES)
    done = 0

    for tf_name, tf_cfg in TIMEFRAMES.items():
        max_hold = holding_bars[tf_name]

        for rr_name, rr_cfg in RR_RATIOS.items():
            # Compute raw TBM for this timeframe + risk/reward
            if "fixed_pct" in rr_cfg:
                raw_labels = _compute_single_tbm(
                    highs, lows, closes, volatility, max_hold,
                    fixed_pct=rr_cfg["fixed_pct"],
                )
            else:
                raw_labels = _compute_single_tbm(
                    highs, lows, closes, volatility, max_hold,
                    pt_mult=rr_cfg["pt_mult"],
                    sl_mult=rr_cfg["sl_mult"],
                )

            # Split by regime
            for regime in REGIMES:
                col_name = f"tbm_{tf_name}_{rr_name}_{regime}"
                regime_mask = regime_values == regime
                regime_labels = np.where(regime_mask, raw_labels, np.nan)
                label_matrix[col_name] = regime_labels

                done += 1
                if progress and done % 20 == 0:
                    print(f"  [{done}/{total}] labels generated...")

    result = pd.DataFrame(label_matrix, index=df.index)

    if progress:
        n_valid = result.notna().sum()
        print(f"\n  Label matrix: {result.shape}")
        print(f"  Valid labels per column (mean): {n_valid.mean():.0f}")
        print(f"  Label balance (+1/-1) per column (mean):")
        pos = (result == 1).sum().mean()
        neg = (result == -1).sum().mean()
        print(f"    +1: {pos:.0f}  -1: {neg:.0f}  ratio: {pos/(pos+neg):.2%}")

    return result
