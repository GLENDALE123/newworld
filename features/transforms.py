"""Feature quality transforms: normalization, nonlinear extraction, interactions.

These transforms make existing features more accessible to the model
without adding new data sources.
"""

import numpy as np
import pandas as pd
from numba import njit


# ═══════════════════════════════════════════════════════════════════════════
# 1. Rolling Percentile Transform
# ═══════════════════════════════════════════════════════════════════════════

@njit
def _rolling_pctl(values, window):
    """Numba-compiled rolling percentile rank (0~1)."""
    n = len(values)
    out = np.empty(n)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        val = values[i]
        if np.isnan(val):
            out[i] = np.nan
            continue
        count_below = 0
        count_valid = 0
        for j in range(i - window + 1, i + 1):
            v = values[j]
            if np.isnan(v):
                continue
            count_valid += 1
            if v < val:
                count_below += 1
        out[i] = count_below / count_valid if count_valid > 0 else np.nan
    return out


def rolling_percentile(series: pd.Series, window: int = 100) -> pd.Series:
    """Transform raw values to rolling percentile rank (0~1).

    Linearizes nonlinear features — makes extreme values explicit.
    """
    result = _rolling_pctl(series.values.astype(np.float64), window)
    return pd.Series(result, index=series.index, name=f"{series.name}_pctl")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cross-Sectional Rank (multi-coin)
# ═══════════════════════════════════════════════════════════════════════════

def cross_sectional_rank(
    feature_by_symbol: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Rank a feature across symbols at each timestamp.

    Returns percentile rank (0~1) for each symbol at each time.
    Handles missing data and different listing dates.
    """
    df = pd.DataFrame(feature_by_symbol)
    ranked = df.rank(axis=1, pct=True)
    return {col: ranked[col] for col in ranked.columns}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Regime-Aware Z-Score
# ═══════════════════════════════════════════════════════════════════════════

@njit
def _regime_zscore(values, vol_proxy, window, vol_threshold_pctl):
    """Z-score that adapts to volatility regime.

    In high-vol regime: use shorter lookback (fast adaptation)
    In low-vol regime: use longer lookback (stable estimates)
    """
    n = len(values)
    out = np.empty(n)
    out[:] = np.nan

    for i in range(window, n):
        # Determine regime from vol_proxy percentile
        vol_sum = 0.0
        vol_count = 0
        for j in range(max(0, i - window * 4), i):
            v = vol_proxy[j]
            if not np.isnan(v):
                vol_sum += v
                vol_count += 1
        vol_mean = vol_sum / vol_count if vol_count > 0 else 0.0
        is_high_vol = vol_proxy[i] > vol_mean * 1.5

        # Adaptive window
        w = window // 2 if is_high_vol else window

        # Compute z-score
        s = 0.0
        s2 = 0.0
        cnt = 0
        start = max(0, i - w)
        for j in range(start, i):
            v = values[j]
            if not np.isnan(v):
                s += v
                s2 += v * v
                cnt += 1
        if cnt < 5:
            continue
        mean = s / cnt
        var = s2 / cnt - mean * mean
        std = np.sqrt(max(var, 1e-10))
        out[i] = (values[i] - mean) / std

    return out


def regime_zscore(
    series: pd.Series,
    vol_proxy: pd.Series,
    window: int = 100,
) -> pd.Series:
    """Z-score that adapts lookback to volatility regime."""
    result = _regime_zscore(
        series.values.astype(np.float64),
        vol_proxy.values.astype(np.float64),
        window,
        0.75,
    )
    return pd.Series(result, index=series.index, name=f"{series.name}_rz")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Interaction Features
# ═══════════════════════════════════════════════════════════════════════════

def build_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful feature interactions based on domain knowledge.

    Only creates interactions between features known to have synergy:
    - Order flow × volatility (flow matters more in calm markets)
    - OI × price (divergence = regime shift signal)
    - Volatility × position (breakout from range)
    """
    f = {}

    # Helper: safe get column or return None
    def _get(name):
        return features[name] if name in features.columns else None

    # ── Flow × Volatility ───────────────────────────────────────────
    cvd = _get("flow_cvd")
    buy_ratio = _get("flow_buy_ratio")

    for vol_col in ["15m_vol_20", "15m_vol_50", "1h_vol_20"]:
        vol = _get(vol_col)
        if vol is None:
            continue
        vol_inv = 1.0 / vol.replace(0, np.nan)  # inverse vol = calm market

        if cvd is not None:
            # CVD in calm market (more meaningful)
            f[f"ix_cvd_x_calm_{vol_col}"] = cvd * vol_inv.clip(upper=vol_inv.quantile(0.95))

        if buy_ratio is not None:
            # Buy pressure in calm market
            f[f"ix_buyratio_x_calm_{vol_col}"] = (buy_ratio - 0.5) * vol_inv.clip(upper=vol_inv.quantile(0.95))

    # ── OI × Price Divergence ───────────────────────────────────────
    for oi_col in ["deriv_oi_chg_5", "deriv_oi_chg_10", "deriv_oi_chg_20"]:
        oi_chg = _get(oi_col)
        if oi_chg is None:
            continue
        for ret_col in ["15m_ret_5", "15m_ret_10", "15m_ret_20"]:
            ret = _get(ret_col)
            if ret is None:
                continue
            # Same window only (5-5, 10-10, 20-20)
            if oi_col.split("_")[-1] != ret_col.split("_")[-1]:
                continue
            w = oi_col.split("_")[-1]
            # Divergence: OI goes one way, price goes opposite
            f[f"ix_oi_price_div_{w}"] = oi_chg * (-ret)  # positive = divergence
            # Confirmation: OI and price same direction
            f[f"ix_oi_price_conf_{w}"] = oi_chg * ret  # positive = confirmation

    # ── Volatility × Price Position (breakout signal) ────────────────
    for pos_col in ["15m_pos_20", "15m_pos_50"]:
        pos = _get(pos_col)
        if pos is None:
            continue
        for vol_col in ["15m_vol_20", "15m_vol_50"]:
            vol = _get(vol_col)
            if vol is None:
                continue
            if pos_col.split("_")[-1] != vol_col.split("_")[-1]:
                continue
            w = pos_col.split("_")[-1]
            # Price at extreme + rising volatility = breakout
            extreme = (pos - 0.5).abs()  # 0 at middle, 0.5 at extremes
            f[f"ix_breakout_{w}"] = extreme * vol

    # ── Funding × OI ────────────────────────────────────────────────
    fund = _get("fund_rate")
    oi = _get("deriv_oi")
    if fund is not None and oi is not None:
        oi_chg = oi.pct_change(20)
        # High funding + rising OI = crowded trade
        f["ix_crowded"] = fund.abs() * oi_chg.clip(lower=0)
        # Extreme negative funding + dropping OI = capitulation
        f["ix_capitulation"] = (-fund).clip(lower=0) * (-oi_chg).clip(lower=0)

    # ── Long/Short Ratio × Volume Surge ──────────────────────────────
    ls = _get("deriv_ls_ratio")
    top_ls = _get("deriv_top_ls_ratio")
    vol_surge = _get("15m_vol_surge_20")
    if ls is not None and vol_surge is not None:
        # Extreme L/S ratio during volume surge
        ls_extreme = (ls - ls.rolling(100).mean()).abs()
        f["ix_ls_vol_surge"] = ls_extreme * vol_surge
    if top_ls is not None and vol_surge is not None:
        top_extreme = (top_ls - top_ls.rolling(100).mean()).abs()
        f["ix_topls_vol_surge"] = top_extreme * vol_surge

    result = pd.DataFrame(f, index=features.index)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main Transform Pipeline
# ═══════════════════════════════════════════════════════════════════════════

# Features worth transforming to percentile (strong nonlinear signal)
PCTL_FEATURES = [
    "flow_cvd", "flow_buy_ratio",
    "flow_delta_sum_5", "flow_delta_sum_10", "flow_delta_sum_20",
    "flow_cvd_chg_5", "flow_cvd_chg_10", "flow_cvd_chg_20",
    "deriv_oi", "deriv_top_ls_ratio", "deriv_ls_ratio",
    "deriv_taker_ratio",
    "15m_vol_20", "15m_vol_50", "15m_vol_100",
    "1h_vol_20", "1h_vol_50",
    "fund_rate",
]

# Features for regime-aware z-score
RZSCORE_FEATURES = [
    "flow_cvd", "flow_delta_sum_20",
    "deriv_oi", "deriv_top_ls_ratio",
    "fund_rate",
]


def apply_transforms(
    features: pd.DataFrame,
    pctl_window: int = 200,
    rz_window: int = 100,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """Apply all quality transforms to a feature DataFrame.

    Adds new columns (doesn't remove originals):
    - *_pctl: rolling percentile versions
    - *_rz: regime-aware z-scores
    - ix_*: interaction features
    """
    parts = [features]

    # 1. Rolling percentile
    pctl_dict = {}
    for col in PCTL_FEATURES:
        if col in features.columns:
            pctl_dict[f"{col}_pctl"] = rolling_percentile(features[col], pctl_window).values
    if pctl_dict:
        parts.append(pd.DataFrame(pctl_dict, index=features.index))

    # 2. Regime-aware z-score
    vol_proxy = features.get("15m_vol_20")
    if vol_proxy is not None:
        rz_dict = {}
        for col in RZSCORE_FEATURES:
            if col in features.columns:
                rz_dict[f"{col}_rz"] = regime_zscore(
                    features[col], vol_proxy, rz_window,
                ).values
        if rz_dict:
            parts.append(pd.DataFrame(rz_dict, index=features.index))

    # 3. Interactions
    if include_interactions:
        ix = build_interaction_features(features)
        if not ix.empty:
            parts.append(ix)

    result = pd.concat(parts, axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
