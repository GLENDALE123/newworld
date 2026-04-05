"""
Automatic Feature Factory

Generates thousands of combination features from raw data.
No hand-crafting. Let the model find what works.

Feature categories:
  1. Temporal: returns, momentum, rolling stats at multiple windows
  2. Cross-feature: ratios, products, differences between any two raw inputs
  3. Statistical: z-scores, percentile ranks, rolling correlations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations


WINDOWS = [5, 10, 20, 50, 100, 200]


# ── Temporal Features ─────────────────────────────────────────────────────────

def _temporal_features(series: pd.Series, name: str, windows: list[int]) -> dict[str, pd.Series]:
    """Generate temporal features for a single series."""
    feats = {}

    for w in windows:
        feats[f"{name}_ret_{w}"] = series.pct_change(w)
        feats[f"{name}_mom_{w}"] = series / series.shift(w) - 1
        feats[f"{name}_rmean_{w}"] = series.rolling(w).mean()
        feats[f"{name}_rstd_{w}"] = series.rolling(w).std()
        feats[f"{name}_rmin_{w}"] = series.rolling(w).min()
        feats[f"{name}_rmax_{w}"] = series.rolling(w).max()

        # Z-score: how far is current value from rolling mean
        rmean = series.rolling(w).mean()
        rstd = series.rolling(w).std()
        feats[f"{name}_zscore_{w}"] = (series - rmean) / rstd.replace(0, np.nan)

        # Percentile rank within rolling window
        feats[f"{name}_rank_{w}"] = series.rolling(w).rank(pct=True)

    return feats


# ── Cross Features ────────────────────────────────────────────────────────────

def _cross_features(
    series_a: pd.Series, name_a: str,
    series_b: pd.Series, name_b: str,
    windows: list[int],
) -> dict[str, pd.Series]:
    """Generate cross-features between two series."""
    feats = {}

    # Ratio
    feats[f"{name_a}_div_{name_b}"] = series_a / series_b.replace(0, np.nan)

    # Difference
    feats[f"{name_a}_sub_{name_b}"] = series_a - series_b

    # Product (normalized)
    a_norm = (series_a - series_a.rolling(50).mean()) / series_a.rolling(50).std().replace(0, np.nan)
    b_norm = (series_b - series_b.rolling(50).mean()) / series_b.rolling(50).std().replace(0, np.nan)
    feats[f"{name_a}_x_{name_b}"] = a_norm * b_norm

    # Rolling correlation
    for w in [20, 50, 100]:
        if w in windows:
            feats[f"{name_a}_corr_{name_b}_{w}"] = series_a.rolling(w).corr(series_b)

    # Divergence: change in A vs change in B
    for w in [5, 20, 50]:
        if w in windows:
            chg_a = series_a.pct_change(w)
            chg_b = series_b.pct_change(w)
            feats[f"{name_a}_div_chg_{name_b}_{w}"] = chg_a - chg_b

    return feats


# ── Volume/Trade Features ────────────────────────────────────────────────────

def _volume_features(df: pd.DataFrame, windows: list[int]) -> dict[str, pd.Series]:
    """Features from volume and trade data."""
    feats = {}

    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        total_vol = df["buy_volume"] + df["sell_volume"]
        buy_ratio = df["buy_volume"] / total_vol.replace(0, np.nan)
        cvd = (df["buy_volume"] - df["sell_volume"]).cumsum()

        feats["buy_ratio"] = buy_ratio
        feats["cvd"] = cvd

        for w in windows:
            feats[f"buy_ratio_rmean_{w}"] = buy_ratio.rolling(w).mean()
            feats[f"cvd_chg_{w}"] = cvd.diff(w)
            feats[f"cvd_zscore_{w}"] = (cvd - cvd.rolling(w).mean()) / cvd.rolling(w).std().replace(0, np.nan)

    if "volume" in df.columns:
        vol = df["volume"]
        for w in windows:
            feats[f"vol_surge_{w}"] = vol / vol.rolling(w).mean().replace(0, np.nan)

    if "trade_count" in df.columns:
        tc = df["trade_count"]
        for w in windows:
            feats[f"trades_surge_{w}"] = tc / tc.rolling(w).mean().replace(0, np.nan)

    return feats


# ── Main Factory ─────────────────────────────────────────────────────────────

def generate_features(
    tick_bar_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
    windows: list[int] | None = None,
    max_cross_pairs: int = 50,
    progress: bool = True,
) -> pd.DataFrame:
    """Auto-generate combination features from raw data.

    Args:
        tick_bar_df: Tick bar DataFrame with timestamp index and OHLCV + buy/sell volume.
        metrics_df: Optional metrics DataFrame (OI, taker ratio, LS ratio) aligned to same index.
        funding_df: Optional funding rate DataFrame aligned to same index.
        windows: Rolling window sizes. Defaults to [5, 10, 20, 50, 100, 200].
        max_cross_pairs: Max number of cross-feature pairs to generate.
        progress: Print progress.

    Returns:
        DataFrame of auto-generated features. Expect 500-2000+ columns.
    """
    if windows is None:
        windows = WINDOWS

    all_feats: dict[str, pd.Series] = {}

    # ── 1. Price temporal features ──
    if progress:
        print("  [1/5] Price temporal features...")

    price = tick_bar_df["close"]
    all_feats.update(_temporal_features(price, "price", windows))

    # OHLC derived
    hl_range = tick_bar_df["high"] - tick_bar_df["low"]
    oc_range = tick_bar_df["close"] - tick_bar_df["open"]
    body_ratio = oc_range / hl_range.replace(0, np.nan)
    upper_shadow = tick_bar_df["high"] - tick_bar_df[["open", "close"]].max(axis=1)
    lower_shadow = tick_bar_df[["open", "close"]].min(axis=1) - tick_bar_df["low"]

    all_feats.update(_temporal_features(hl_range, "hl_range", windows))
    all_feats["body_ratio"] = body_ratio
    all_feats["upper_shadow_ratio"] = upper_shadow / hl_range.replace(0, np.nan)
    all_feats["lower_shadow_ratio"] = lower_shadow / hl_range.replace(0, np.nan)

    # ── 2. Volume/trade features ──
    if progress:
        print("  [2/5] Volume/trade features...")
    all_feats.update(_volume_features(tick_bar_df, windows))

    # ── 3. Metrics features (OI, taker, LS ratio) ──
    raw_series: dict[str, pd.Series] = {"price": price}

    if metrics_df is not None and not metrics_df.empty:
        if progress:
            print("  [3/5] Metrics features...")
        for col in metrics_df.columns:
            s = metrics_df[col]
            short_name = col.replace("sum_", "").replace("count_", "cnt_")
            all_feats.update(_temporal_features(s, short_name, windows))
            raw_series[short_name] = s
    else:
        if progress:
            print("  [3/5] Metrics features... (skipped, no data)")

    # ── 4. Funding features ──
    if funding_df is not None and not funding_df.empty:
        if progress:
            print("  [4/5] Funding features...")
        for col in funding_df.columns:
            s = funding_df[col]
            all_feats.update(_temporal_features(s, col, windows))
            raw_series[col] = s
    else:
        if progress:
            print("  [4/5] Funding features... (skipped, no data)")

    # ── 5. Cross features ──
    if progress:
        print(f"  [5/5] Cross features ({len(raw_series)} raw series)...")

    series_names = list(raw_series.keys())
    pairs = list(combinations(series_names, 2))
    if len(pairs) > max_cross_pairs:
        # Prioritize price-based crosses
        price_pairs = [(a, b) for a, b in pairs if "price" in a or "price" in b]
        other_pairs = [(a, b) for a, b in pairs if "price" not in a and "price" not in b]
        pairs = price_pairs + other_pairs[:max_cross_pairs - len(price_pairs)]

    for name_a, name_b in pairs:
        cross = _cross_features(
            raw_series[name_a], name_a,
            raw_series[name_b], name_b,
            windows,
        )
        all_feats.update(cross)

    result = pd.DataFrame(all_feats, index=tick_bar_df.index)

    # Replace inf with NaN, then clean
    result = result.replace([np.inf, -np.inf], np.nan)

    # Remove columns that are all NaN or all constant
    result = result.loc[:, result.nunique() > 1]
    result = result.loc[:, result.notna().sum() > len(result) * 0.5]

    if progress:
        print(f"\n  Generated {result.shape[1]} features from {len(tick_bar_df)} rows")
        n_inf = np.isinf(result.select_dtypes(include=[np.number]).values).sum()
        print(f"  Inf values: {n_inf}")

    return result
