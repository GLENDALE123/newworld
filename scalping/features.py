"""Scalping feature builder — 31 validated features + proper lookahead prevention."""

import numpy as np
import pandas as pd


def build_scalp_features(
    kline_5m: pd.DataFrame,
    kline_15m: pd.DataFrame,
    tick_bar: pd.DataFrame,
    book_ticker: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build scalping features from 5m + 15m + tick_bar + book_ticker.

    All features are lookahead-free:
    - 5m features: use current and past bars (no shift needed)
    - tick_bar/book_ticker: resampled to 5m with shift(1)
    - 15m features: resampled to 5m with shift(3)
    - funding: shift(1) after resample

    Returns DataFrame indexed by 5m timestamps.
    """
    f = {}
    c = kline_5m["close"]
    h = kline_5m["high"]
    l = kline_5m["low"]
    v = kline_5m["volume"]
    o = kline_5m["open"]

    # ── 5m Price (no shift needed — uses past data) ──
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

    f["5m_vol_100"] = c.pct_change(1).rolling(100).std()

    # ── Cross-TF Volatility Ratios (strongest features) ──
    # Use 5m rolling vol at different windows as proxy for different TFs
    vol_15m = c.pct_change(1).rolling(60).std()    # ~15m equivalent (12 bars × 5m)
    vol_1h = c.pct_change(1).rolling(240).std()     # ~1h equivalent (48 bars × 5m)
    vol_4h = c.pct_change(1).rolling(960).std()     # ~4h equivalent (192 bars × 5m)

    f["xtf_vol_15m_vs_1h"] = vol_15m / vol_1h.replace(0, np.nan)
    f["xtf_vol_15m_vs_4h"] = vol_15m / vol_4h.replace(0, np.nan)
    f["xtf_vol_1h_vs_4h"] = vol_1h / vol_4h.replace(0, np.nan)

    # ── Cross-TF Trend Alignment ──
    ret_5m = c.pct_change(5)
    ret_15m = c.pct_change(15)
    ret_1h = c.pct_change(60)
    ret_4h = c.pct_change(240)

    f["xtf_trend_5m_vs_15m"] = np.sign(ret_5m) * np.sign(ret_15m)
    f["xtf_trend_5m_vs_1h"] = np.sign(ret_5m) * np.sign(ret_1h)
    f["xtf_trend_5m_vs_4h"] = np.sign(ret_5m) * np.sign(ret_4h)
    f["xtf_trend_15m_vs_1h"] = np.sign(ret_15m) * np.sign(ret_1h)
    f["xtf_trend_15m_vs_4h"] = np.sign(ret_15m) * np.sign(ret_4h)
    f["xtf_trend_1h_vs_4h"] = np.sign(ret_1h) * np.sign(ret_4h)

    # ── 15m Context (shift 3 = wait for completed 15m bar) ──
    c15 = kline_15m["close"]
    h15 = kline_15m["high"]
    l15 = kline_15m["low"]
    o15 = kline_15m["open"]

    ctx = {}
    ctx["15m_body_ratio"] = (c15 - o15) / (h15 - l15).replace(0, np.nan)
    ctx["15m_upper_wick"] = (h15 - pd.concat([c15, o15], axis=1).max(axis=1)) / (h15 - l15).replace(0, np.nan)
    ctx["15m_ret_5"] = c15.pct_change(5)
    ctx["15m_ret_10"] = c15.pct_change(10)
    ctx["15m_ret_20"] = c15.pct_change(20)

    rmax20 = c15.rolling(20).max()
    rmin20 = c15.rolling(20).min()
    ctx["15m_pos_20"] = (c15 - rmin20) / (rmax20 - rmin20).replace(0, np.nan)
    rmax50 = c15.rolling(50).max()
    rmin50 = c15.rolling(50).min()
    ctx["15m_pos_50"] = (c15 - rmin50) / (rmax50 - rmin50).replace(0, np.nan)
    rmax100 = c15.rolling(100).max()
    rmin100 = c15.rolling(100).min()
    ctx["15m_pos_100"] = (c15 - rmin100) / (rmax100 - rmin100).replace(0, np.nan)

    ctx_df = pd.DataFrame(ctx, index=kline_15m.index).resample("5min").ffill().shift(3)
    for col in ctx_df.columns:
        f[col] = ctx_df[col].reindex(kline_5m.index)

    # ── Order Flow (tick_bar → 5m, shift 1) ──
    tb = tick_bar.copy()
    tb["delta"] = tb["buy_volume"] - tb["sell_volume"]
    tb["cvd"] = tb["delta"].cumsum()

    t5 = tb.resample("5min").agg({
        "delta": "sum", "cvd": "last",
        "volume": "sum", "trade_count": "sum",
    }).shift(1)

    cvd = t5["cvd"].reindex(kline_5m.index)
    f["flow_cvd"] = cvd

    # CVD lags
    for lag in range(1, 8):
        f[f"flow_cvd_lag{lag}"] = cvd.shift(lag)

    # ── Book Ticker (shift 1) ──
    if book_ticker is not None and len(book_ticker) > 0:
        b5 = book_ticker.resample("5min").agg({
            "spread_bps": "mean", "obi": "mean",
        }).shift(1).reindex(kline_5m.index)

        # Spread volatility (new discovery)
        for w in [3, 5]:
            f[f"spread_vol_{w}"] = b5["spread_bps"].rolling(w).std()

    # ── Funding (shift 1) ──
    if funding is not None and len(funding) > 0:
        fr = funding[["funding_rate"]].resample("5min").ffill().shift(1)
        fr = fr.reindex(kline_5m.index)
        f["fund_rate"] = fr["funding_rate"]
        f["fund_abs"] = fr["funding_rate"].abs()

    # ── Time ──
    hour = kline_5m.index.hour
    f["is_asia"] = pd.Series(((hour >= 0) & (hour < 8)).astype(np.float32), index=kline_5m.index)
    f["is_us"] = pd.Series(((hour >= 16) & (hour < 24)).astype(np.float32), index=kline_5m.index)
    f["dow_sin"] = pd.Series(np.sin(2 * np.pi * kline_5m.index.dayofweek / 7).astype(np.float32), index=kline_5m.index)

    # ── Combinations ──
    f["ix_asia_cvd"] = f["is_asia"] * cvd
    f["ix_us_cvd"] = f["is_us"] * cvd

    result = pd.DataFrame(f)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
