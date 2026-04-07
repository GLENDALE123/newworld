"""
Temporal Context Features — Recent bar patterns for regime awareness

The PLE model is point-in-time: it sees features at time T but has no explicit
memory of recent bars. These features encode recent market behavior as extra inputs.

Categories:
  1. Recent return profile (last 4/8/16 bars)
  2. Momentum consistency (how aligned are recent bars?)
  3. Volume profile evolution
  4. Volatility trajectory (expanding/contracting?)
  5. Micro-regime signals (squeeze detection, breakout proximity)
"""

import numpy as np
import pandas as pd


def generate_temporal_features(
    df: pd.DataFrame,
    windows: list[int] = [4, 8, 16, 32],
) -> dict[str, pd.Series]:
    """Generate temporal context features from OHLCV data.

    These features capture recent sequential patterns that a point-in-time
    model would otherwise miss.
    """
    f = {}
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    ret = c.pct_change()

    for w in windows:
        # 1. Return trajectory — is the trend accelerating or decelerating?
        # Split window into halves: compare first half vs second half
        half = w // 2
        first_half_ret = c.diff(half).shift(half) / c.shift(w)
        second_half_ret = c.diff(half) / c.shift(half)
        f[f"tc_ret_accel_{w}"] = second_half_ret - first_half_ret

        # 2. Momentum consistency — what fraction of recent bars are positive?
        f[f"tc_pos_ratio_{w}"] = ret.rolling(w).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )

        # 3. Consecutive direction — longest streak of same-direction bars
        # Encoded as: positive streak count (+) or negative streak count (-)
        signs = np.sign(ret.values)
        streak = np.zeros(len(signs))
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1] and signs[i] != 0:
                streak[i] = streak[i-1] + signs[i]
            else:
                streak[i] = signs[i]
        f[f"tc_streak_{w}"] = pd.Series(streak, index=c.index).rolling(w).apply(
            lambda x: x[-1], raw=True
        )

        # 4. High-low range trend — is range expanding or contracting?
        bar_range = h - l
        range_ma_short = bar_range.rolling(half).mean()
        range_ma_long = bar_range.rolling(w).mean()
        f[f"tc_range_ratio_{w}"] = range_ma_short / range_ma_long.replace(0, np.nan)

        # 5. Volume trajectory
        vol_short = v.rolling(half).mean()
        vol_long = v.rolling(w).mean()
        f[f"tc_vol_trend_{w}"] = vol_short / vol_long.replace(0, np.nan)

        # 6. Volatility regime — realized vol vs longer window
        rvol_short = ret.rolling(half).std()
        rvol_long = ret.rolling(w).std()
        f[f"tc_vol_regime_{w}"] = rvol_short / rvol_long.replace(0, np.nan)

    # 7. Squeeze detection — ATR narrowing relative to longer history
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14).mean()
    atr_median_96 = atr_14.rolling(96).median()
    f["tc_squeeze_ratio"] = atr_14 / atr_median_96.replace(0, np.nan)

    # 8. Price position velocity — how fast is price moving within range?
    for w in [20, 50]:
        rmax = c.rolling(w).max()
        rmin = c.rolling(w).min()
        pos = (c - rmin) / (rmax - rmin).replace(0, np.nan)
        f[f"tc_pos_velocity_{w}"] = pos.diff(4)  # how fast position changes

    # 9. Gap analysis — open vs previous close
    gap = df["open"] - c.shift(1)
    gap_pct = gap / c.shift(1)
    f["tc_gap_pct"] = gap_pct
    f["tc_gap_zscore"] = (gap_pct - gap_pct.rolling(96).mean()) / gap_pct.rolling(96).std().replace(0, np.nan)

    # 10. Directional movement balance
    up_move = (h - h.shift(1)).clip(lower=0)
    down_move = (l.shift(1) - l).clip(lower=0)
    for w in [8, 16, 32]:
        up_sum = up_move.rolling(w).sum()
        down_sum = down_move.rolling(w).sum()
        total = up_sum + down_sum
        f[f"tc_dm_balance_{w}"] = (up_sum - down_sum) / total.replace(0, np.nan)

    return f


def add_temporal_features(
    features_df: pd.DataFrame,
    kline_df: pd.DataFrame,
    target_tf: str = "15m",
) -> pd.DataFrame:
    """Add temporal context features to existing feature DataFrame."""
    temporal = generate_temporal_features(kline_df)
    temporal_df = pd.DataFrame(temporal)

    # Resample if needed
    if target_tf != "15m":
        temporal_df = temporal_df.resample(target_tf).last()

    # Align and merge
    common_idx = features_df.index.intersection(temporal_df.index)
    result = features_df.loc[common_idx].copy()
    for col in temporal_df.columns:
        result[col] = temporal_df[col].reindex(common_idx)

    # Clean
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
