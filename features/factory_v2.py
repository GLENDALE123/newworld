"""
Feature Factory v2: ALL data sources, multi-timeframe, cross-asset

Input: raw DataFrames from data/merged/
Output: aligned feature DataFrame per target timeframe

Data sources utilized:
  1. kline (5m/15m/1h) → multi-TF price + ATR + volatility structure
  2. tick_bar → order flow (CVD, buy ratio, trade intensity)
  3. metrics → derivatives (OI, funding, long/short ratios)
  4. book_ticker → microstructure (spread, OBI)
  5. cross-asset (ETH, SOL) → lead-lag, correlation, rotation
"""

import numpy as np
import pandas as pd


WINDOWS = [5, 10, 20, 50, 100]


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return np.where(np.isfinite(r), r, 0.0)


# ── 1. Multi-Timeframe Price Features ────────────────────────────────────────

def _price_features(df: pd.DataFrame, prefix: str, windows: list[int]) -> dict[str, pd.Series]:
    """ATR + returns + vol + price structure from OHLCV."""
    f = {}
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    for w in windows:
        atr = tr.ewm(span=w).mean()
        f[f"{prefix}_atr_{w}"] = atr
        f[f"{prefix}_atr_pct_{w}"] = atr / c

    # Returns
    for w in windows:
        f[f"{prefix}_ret_{w}"] = c.pct_change(w)

    # Volatility
    ret = c.pct_change()
    for w in windows:
        f[f"{prefix}_vol_{w}"] = ret.rolling(w).std()

    # Price position (where is price relative to recent range)
    for w in [20, 50, 100]:
        rmax = c.rolling(w).max()
        rmin = c.rolling(w).min()
        f[f"{prefix}_pos_{w}"] = _safe_div((c - rmin).values, (rmax - rmin).values)
        f[f"{prefix}_pos_{w}"] = pd.Series(f[f"{prefix}_pos_{w}"], index=c.index)

    # Candle structure
    body = c - df["open"]
    wick_up = h - pd.concat([c, df["open"]], axis=1).max(axis=1)
    wick_dn = pd.concat([c, df["open"]], axis=1).min(axis=1) - l
    rng = h - l
    f[f"{prefix}_body_ratio"] = _safe_div(body.values, rng.values)
    f[f"{prefix}_body_ratio"] = pd.Series(f[f"{prefix}_body_ratio"], index=c.index)
    f[f"{prefix}_upper_wick"] = _safe_div(wick_up.values, rng.values)
    f[f"{prefix}_upper_wick"] = pd.Series(f[f"{prefix}_upper_wick"], index=c.index)

    # Volume
    for w in windows:
        f[f"{prefix}_vol_surge_{w}"] = v / v.rolling(w).mean()

    return f


# ── 2. Order Flow (from tick_bar) ────────────────────────────────────────────

def _order_flow_features(tick_bar: pd.DataFrame, target_tf: str) -> dict[str, pd.Series]:
    """CVD, buy ratio, trade intensity resampled to target timeframe."""
    f = {}
    tb = tick_bar.copy()

    total_vol = tb["buy_volume"] + tb["sell_volume"]
    tb["buy_ratio"] = tb["buy_volume"] / total_vol.replace(0, np.nan)
    tb["delta"] = tb["buy_volume"] - tb["sell_volume"]
    tb["cvd"] = tb["delta"].cumsum()

    # Resample to target
    agg = {
        "buy_ratio": "mean",
        "delta": "sum",
        "cvd": "last",
        "volume": "sum",
        "trade_count": "sum",
    }
    resampled = tb[list(agg.keys())].resample(target_tf).agg(agg)
    # Shift by one bar so only completed bars contribute to the current row.
    resampled = resampled.shift(1)

    for w in WINDOWS:
        f[f"flow_buy_ratio_{w}"] = resampled["buy_ratio"].rolling(w).mean()
        f[f"flow_delta_sum_{w}"] = resampled["delta"].rolling(w).sum()
        f[f"flow_cvd_chg_{w}"] = resampled["cvd"].diff(w)
        f[f"flow_cvd_zscore_{w}"] = (
            (resampled["cvd"] - resampled["cvd"].rolling(w).mean()) /
            resampled["cvd"].rolling(w).std().replace(0, np.nan)
        )
        f[f"flow_intensity_{w}"] = resampled["trade_count"] / resampled["trade_count"].rolling(w).mean()

    f["flow_buy_ratio"] = resampled["buy_ratio"]
    f["flow_cvd"] = resampled["cvd"]

    return f


# ── 3. Derivatives (from metrics) ───────────────────────────────────────────

def _derivatives_features(metrics: pd.DataFrame, target_tf: str) -> dict[str, pd.Series]:
    """OI, long/short ratios, taker ratio."""
    f = {}
    m = metrics.resample(target_tf).last().ffill()

    cols = {
        "sum_open_interest_value": "oi",
        "sum_taker_long_short_vol_ratio": "taker_ratio",
        "count_long_short_ratio": "ls_ratio",
        "sum_toptrader_long_short_ratio": "top_ls_ratio",
    }

    for raw_col, name in cols.items():
        if raw_col not in m.columns:
            continue
        s = m[raw_col]
        f[f"deriv_{name}"] = s

        for w in WINDOWS:
            f[f"deriv_{name}_chg_{w}"] = s.pct_change(w)
            f[f"deriv_{name}_zscore_{w}"] = (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)

    # OI-price divergence (if close price available)
    if "oi" in [v for v in cols.values()] and "sum_open_interest_value" in m.columns:
        for w in [10, 20, 50]:
            f[f"deriv_oi_chg_{w}"] = m["sum_open_interest_value"].pct_change(w)

    return f


# ── 4. Funding Rate ─────────────────────────────────────────────────────────

def _funding_features(funding: pd.DataFrame, target_tf: str) -> dict[str, pd.Series]:
    """Funding rate features."""
    f = {}
    fr = funding[["funding_rate"]].resample(target_tf).ffill()["funding_rate"]

    f["fund_rate"] = fr
    f["fund_abs"] = fr.abs()

    for w in [10, 20, 50, 100]:
        f[f"fund_zscore_{w}"] = (fr - fr.rolling(w).mean()) / fr.rolling(w).std().replace(0, np.nan)
        f[f"fund_extreme_{w}"] = (fr.abs() > fr.abs().rolling(w).quantile(0.9)).astype(float)

    return f


# ── 5. Microstructure (from book_ticker) ─────────────────────────────────────

def _microstructure_features(book_ticker: pd.DataFrame, target_tf: str) -> dict[str, pd.Series]:
    """Spread, OBI features."""
    f = {}
    bt = book_ticker[["spread_bps", "obi"]].resample(target_tf).mean()

    f["micro_spread"] = bt["spread_bps"]
    f["micro_obi"] = bt["obi"]

    for w in [10, 20, 50]:
        f[f"micro_spread_zscore_{w}"] = (
            (bt["spread_bps"] - bt["spread_bps"].rolling(w).mean()) /
            bt["spread_bps"].rolling(w).std().replace(0, np.nan)
        )
        f[f"micro_obi_ma_{w}"] = bt["obi"].rolling(w).mean()

    return f


# ── 6. Cross-Asset ──────────────────────────────────────────────────────────

def _cross_asset_features(
    btc_close: pd.Series,
    other_closes: dict[str, pd.Series],
    windows: list[int],
) -> dict[str, pd.Series]:
    """Correlation, lead-lag, relative strength vs BTC."""
    f = {}

    for name, other in other_closes.items():
        # Align (strip timezone if mismatch)
        btc = btc_close.copy()
        oth = other.copy()
        if hasattr(btc.index, 'tz') and btc.index.tz is not None:
            btc.index = btc.index.tz_localize(None)
        if hasattr(oth.index, 'tz') and oth.index.tz is not None:
            oth.index = oth.index.tz_localize(None)
        aligned = pd.DataFrame({"btc": btc, name: oth}).dropna()
        if len(aligned) < 50:
            continue

        btc_ret = aligned["btc"].pct_change()
        other_ret = aligned[name].pct_change()

        # Rolling correlation
        for w in [20, 50]:
            f[f"xasset_{name}_corr_{w}"] = btc_ret.rolling(w).corr(other_ret)

        # Relative strength: other outperforming BTC?
        for w in [10, 20, 50]:
            btc_cum = btc_ret.rolling(w).sum()
            other_cum = other_ret.rolling(w).sum()
            f[f"xasset_{name}_rs_{w}"] = other_cum - btc_cum

        # Lead-lag: does other move first?
        f[f"xasset_{name}_lead1"] = other_ret.shift(1)  # other's return 1 bar ago
        f[f"xasset_{name}_lead2"] = other_ret.shift(2)

    return f


# ── 7. Cross-Timeframe ──────────────────────────────────────────────────────

def _cross_tf_features(
    price_features_by_tf: dict[str, dict[str, pd.Series]],
    target_tf: str,
) -> dict[str, pd.Series]:
    """Volatility ratios and trend alignment across timeframes."""
    f = {}
    tf_order = ["5m", "15m", "1h", "4h"]

    for i, tf1 in enumerate(tf_order):
        for tf2 in tf_order[i+1:]:
            key1_vol = f"{tf1}_vol_20"
            key2_vol = f"{tf2}_vol_20"

            if tf1 in price_features_by_tf and tf2 in price_features_by_tf:
                pf1 = price_features_by_tf[tf1]
                pf2 = price_features_by_tf[tf2]

                if key1_vol in pf1 and key2_vol in pf2:
                    v1 = pf1[key1_vol].resample(target_tf).last()
                    v2 = pf2[key2_vol].resample(target_tf).last()
                    aligned = pd.DataFrame({"v1": v1, "v2": v2}).ffill().dropna()
                    if len(aligned) > 0:
                        f[f"xtf_vol_{tf1}_vs_{tf2}"] = aligned["v1"] / aligned["v2"].replace(0, np.nan)

                # Trend alignment
                key1_ret = f"{tf1}_ret_20"
                key2_ret = f"{tf2}_ret_20"
                if key1_ret in pf1 and key2_ret in pf2:
                    r1 = pf1[key1_ret].resample(target_tf).last()
                    r2 = pf2[key2_ret].resample(target_tf).last()
                    aligned = pd.DataFrame({"r1": r1, "r2": r2}).ffill().dropna()
                    if len(aligned) > 0:
                        # Same direction = aligned trend
                        f[f"xtf_trend_{tf1}_vs_{tf2}"] = np.sign(aligned["r1"]) * np.sign(aligned["r2"])
                        f[f"xtf_trend_{tf1}_vs_{tf2}"] = pd.Series(
                            f[f"xtf_trend_{tf1}_vs_{tf2}"].values, index=aligned.index)

    return f


# ── Main Factory ─────────────────────────────────────────────────────────────

def generate_features_v2(
    kline_data: dict[str, pd.DataFrame],
    tick_bar: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    book_ticker: pd.DataFrame | None = None,
    cross_asset_closes: dict[str, pd.Series] | None = None,
    target_tf: str = "1h",
    progress: bool = True,
) -> pd.DataFrame:
    """Generate comprehensive features aligned to target timeframe.

    Args:
        kline_data: {"5m": df, "15m": df, "1h": df, "4h": df}
        tick_bar: tick bar DataFrame with buy/sell volume
        metrics: OI, long/short ratios
        funding: funding rate
        book_ticker: spread, OBI
        cross_asset_closes: {"ETH": Series, "SOL": Series}
        target_tf: target timeframe for alignment
        progress: print progress

    Returns:
        DataFrame aligned to target_tf with all features. Inf replaced with NaN.
    """
    all_features = {}

    # 1. Multi-TF price features
    # NOTE: features from timeframes longer than target_tf must be shifted
    # to avoid lookahead bias. E.g., 1h features resampled to 15m would
    # otherwise contain the current (incomplete) candle's data.
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "15min": 15, "1h": 60, "4h": 240}
    target_minutes = tf_minutes.get(target_tf, 15)

    if progress:
        print("  [1/7] Multi-TF price features...")
    price_by_tf = {}
    for tf, df in kline_data.items():
        pf = _price_features(df, tf, WINDOWS)
        price_by_tf[tf] = pf
        source_minutes = tf_minutes.get(tf, 15)
        needs_shift = source_minutes > target_minutes
        # Resample to target
        for name, series in pf.items():
            resampled = series.resample(target_tf).last()
            if needs_shift:
                # Shift by 1 source period to use only completed candles
                shift_bars = source_minutes // target_minutes
                resampled = resampled.shift(shift_bars)
            all_features[name] = resampled

    # 2. Order flow
    if tick_bar is not None and len(tick_bar) > 0:
        if progress:
            print("  [2/7] Order flow features...")
        flow = _order_flow_features(tick_bar, target_tf)
        all_features.update(flow)
    elif progress:
        print("  [2/7] Order flow... (skipped)")

    # 3. Derivatives
    if metrics is not None and len(metrics) > 0:
        if progress:
            print("  [3/7] Derivatives features...")
        deriv = _derivatives_features(metrics, target_tf)
        all_features.update(deriv)
    elif progress:
        print("  [3/7] Derivatives... (skipped)")

    # 4. Funding
    if funding is not None and len(funding) > 0:
        if progress:
            print("  [4/7] Funding features...")
        fund = _funding_features(funding, target_tf)
        all_features.update(fund)
    elif progress:
        print("  [4/7] Funding... (skipped)")

    # 5. Microstructure
    if book_ticker is not None and len(book_ticker) > 0:
        if progress:
            print("  [5/7] Microstructure features...")
        micro = _microstructure_features(book_ticker, target_tf)
        all_features.update(micro)
    elif progress:
        print("  [5/7] Microstructure... (skipped)")

    # 6. Cross-asset
    if cross_asset_closes and len(cross_asset_closes) > 0:
        if progress:
            print(f"  [6/7] Cross-asset features ({len(cross_asset_closes)} assets)...")
        target_close = kline_data.get(target_tf, list(kline_data.values())[0])["close"]
        xasset = _cross_asset_features(target_close, cross_asset_closes, [10, 20, 50])
        all_features.update(xasset)
    elif progress:
        print("  [6/7] Cross-asset... (skipped)")

    # 7. Cross-timeframe
    if len(price_by_tf) > 1:
        if progress:
            print("  [7/7] Cross-timeframe features...")
        xtf = _cross_tf_features(price_by_tf, target_tf)
        all_features.update(xtf)
    elif progress:
        print("  [7/7] Cross-TF... (skipped)")

    # Build DataFrame
    result = pd.DataFrame(all_features)
    result = result.replace([np.inf, -np.inf], np.nan)

    # Drop all-NaN or constant columns
    result = result.loc[:, result.nunique() > 1]
    result = result.loc[:, result.notna().sum() > len(result) * 0.3]

    if progress:
        print(f"\n  Total: {result.shape[1]} features, {len(result)} rows")

    return result
