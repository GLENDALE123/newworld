"""BTC 5m 스캘핑 방향 예측력 분석 — opportunity zone 내에서."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ── 1. 데이터 로딩 ──
print("=== Direction Predictability Analysis ===\n")
kline = pd.read_parquet("data/merged/BTCUSDT/kline_5m.parquet")
kline = kline.set_index("timestamp").sort_index()
tick = pd.read_parquet("data/merged/BTCUSDT/tick_bar.parquet")
tick = tick.set_index("timestamp").sort_index()
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)

# Funding rate
import os
fr_path = "data/merged/BTCUSDT/funding_rate.parquet"
funding = None
if os.path.exists(fr_path):
    funding = pd.read_parquet(fr_path).set_index("timestamp").sort_index()
    if funding.index.tz is None and kline.index.tz is not None:
        funding.index = funding.index.tz_localize(kline.index.tz)

c = kline["close"]; o = kline["open"]; h = kline["high"]; l = kline["low"]; v = kline["volume"]

# ── 2. Forward return (target) ──
kline["fwd_6"] = c.shift(-6) / c - 1
# Also compute best within 3-6 bars for MFE
for n in [3, 4, 5, 6]:
    kline[f"_fwd_{n}"] = c.shift(-n) / c - 1

# ── 3. Tick bar aggregation ──
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({
    "buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"
})
kline = kline.drop(columns=["trade_count"], errors="ignore")
kline = kline.join(tick_5m[["buy_volume", "sell_volume", "cvd_raw", "trade_count"]], how="left")

# ── 4. Feature engineering ──
kline["ret"] = c.pct_change()

# Volatility
kline["vol_20"] = kline["ret"].rolling(20).std()
kline["vol_60"] = kline["ret"].rolling(60).std()
kline["vol_squeeze"] = kline["vol_20"] / kline["vol_60"]

# ATR
kline["atr_12"] = (h - l).rolling(12).mean()
kline["atr_ratio"] = (h - l) / kline["atr_12"]

# Volume ratio
kline["vol_ma_12"] = v.rolling(12).mean()
kline["vol_ratio"] = v / kline["vol_ma_12"]

# === PREDICTIVE FEATURES ===

# 1. CVD slope (change over last 3 and 5 bars)
kline["cvd_cumsum"] = kline["cvd_raw"].cumsum()
kline["cvd_slope_3"] = kline["cvd_cumsum"] - kline["cvd_cumsum"].shift(3)
kline["cvd_slope_5"] = kline["cvd_cumsum"] - kline["cvd_cumsum"].shift(5)

# 2. Buy ratio trend (5-bar rolling vs 12-bar rolling)
kline["buy_ratio"] = kline["buy_volume"] / (kline["buy_volume"] + kline["sell_volume"])
kline["buy_ratio_ma5"] = kline["buy_ratio"].rolling(5).mean()
kline["buy_ratio_ma12"] = kline["buy_ratio"].rolling(12).mean()
kline["buy_ratio_trend"] = kline["buy_ratio_ma5"] - kline["buy_ratio_ma12"]

# 3. Prior 3-bar return (momentum)
kline["mom_3"] = c / c.shift(3) - 1

# 4. Upper TF trend (MA slopes)
kline["ma_20"] = c.rolling(20).mean()
kline["ma_50"] = c.rolling(50).mean()
kline["ma_20_slope"] = (kline["ma_20"] - kline["ma_20"].shift(3)) / kline["ma_20"].shift(3)
kline["ma_50_slope"] = (kline["ma_50"] - kline["ma_50"].shift(5)) / kline["ma_50"].shift(5)
kline["ma_20_50_cross"] = kline["ma_20"] / kline["ma_50"] - 1  # positive = bullish

# 5. Funding rate
if funding is not None and "funding_rate" in funding.columns:
    # Forward fill to 5m (funding is 8h)
    kline["funding_rate"] = funding["funding_rate"].reindex(kline.index, method="ffill")
    kline["funding_direction"] = kline["funding_rate"].diff().rolling(3).mean()
else:
    kline["funding_rate"] = np.nan
    kline["funding_direction"] = np.nan

# 6. Taker buy ratio (from kline itself)
kline["taker_buy_pct"] = kline["taker_buy_volume"] / kline["volume"]
kline["taker_buy_pct_ma5"] = kline["taker_buy_pct"].rolling(5).mean()

# 7. Candle body ratio of last 3 bars
kline["body_ratio"] = (c - o) / (h - l + 1e-10)
kline["body_ratio_3avg"] = kline["body_ratio"].rolling(3).mean()

# 8. Additional: OBV slope, price position
kline["obv"] = (np.sign(kline["ret"]) * v).cumsum()
kline["obv_slope_5"] = kline["obv"] - kline["obv"].shift(5)
kline["obv_slope_norm"] = kline["obv_slope_5"] / (v.rolling(5).sum() + 1e-10)

# Range position
kline["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)

# VWAP distance
kline["vwap_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()
kline["vwap_dist"] = c / kline["vwap_20"] - 1

# Session
kline["hour"] = kline.index.hour

# ── 5. Opportunity zone filter ──
df = kline.iloc[60:].copy()
df = df.dropna(subset=["fwd_6"])

# Opportunity zone: vol expansion + volume surge
opp = df[(df["vol_squeeze"] > 1.0) & (df["vol_ratio"] > 1.5)]
print(f"Total bars: {len(df):,}")
print(f"Opportunity zone (vol_squeeze>1.0 & vol_ratio>1.5): {len(opp):,} ({len(opp)/len(df)*100:.1f}%)")

FEE = 0.0008

# ── 6. Feature-by-feature predictive power ──
features_to_test = {
    "cvd_slope_3": "CVD slope (3-bar)",
    "cvd_slope_5": "CVD slope (5-bar)",
    "buy_ratio_trend": "Buy ratio trend (5 vs 12 MA)",
    "buy_ratio": "Buy ratio (current bar)",
    "mom_3": "Momentum (3-bar return)",
    "ma_20_slope": "MA20 slope (3-bar)",
    "ma_50_slope": "MA50 slope (5-bar)",
    "ma_20_50_cross": "MA20/MA50 cross",
    "funding_rate": "Funding rate level",
    "funding_direction": "Funding rate direction",
    "taker_buy_pct": "Taker buy % (current bar)",
    "taker_buy_pct_ma5": "Taker buy % (5-bar avg)",
    "body_ratio_3avg": "Body ratio (3-bar avg)",
    "body_ratio": "Body ratio (current bar)",
    "obv_slope_norm": "OBV slope (normalized)",
    "range_pos_20": "Range position (20-bar)",
    "vwap_dist": "VWAP distance",
}

target = opp["fwd_6"]
results = []

print(f"\n{'='*80}")
print(f"{'Feature':<35s} {'Corr':>7s} {'IC':>7s} {'WR_hi':>7s} {'WR_lo':>7s} {'Δ_WR':>7s} {'n_hi':>7s}")
print(f"{'-'*80}")

for feat_col, feat_name in features_to_test.items():
    x = opp[feat_col]
    y = target
    valid = ~(x.isna() | y.isna())
    if valid.sum() < 500:
        print(f"  {feat_name:<33s} SKIP (n={valid.sum()})")
        continue

    xv = x[valid].values
    yv = y[valid].values

    # Pearson correlation
    corr = np.corrcoef(xv, yv)[0, 1]

    # IC (Spearman rank correlation)
    ic, _ = spearmanr(xv, yv)

    # Median split: win rate
    med = np.median(xv)
    hi_mask = xv >= med
    lo_mask = xv < med

    # "Win" = fwd_6 > fee for long
    wr_hi = (yv[hi_mask] > FEE).mean()
    wr_lo = (yv[lo_mask] > FEE).mean()
    delta_wr = wr_hi - wr_lo

    results.append({
        "feature": feat_name,
        "col": feat_col,
        "corr": corr,
        "ic": ic,
        "wr_hi": wr_hi,
        "wr_lo": wr_lo,
        "delta_wr": delta_wr,
        "n": valid.sum(),
    })

    print(f"  {feat_name:<33s} {corr:+.4f} {ic:+.4f} {wr_hi:.3f} {wr_lo:.3f} {delta_wr:+.3f} {int(valid.sum()):>7d}")

# Rank by |IC|
results.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"\n{'='*80}")
print("── Ranked by |IC| (Spearman) ──")
for i, r in enumerate(results):
    marker = "***" if abs(r["delta_wr"]) > 0.02 else ""
    print(f"  {i+1:2d}. {r['feature']:<33s} IC={r['ic']:+.4f} Δ_WR={r['delta_wr']:+.3f} {marker}")

# ── 7. Combination analysis ──
print(f"\n{'='*80}")
print("── Combination Analysis (within opportunity zone) ──\n")

combos = [
    ("cvd_slope_5 > 0 AND ma_20_slope > 0", lambda d: (d["cvd_slope_5"] > 0) & (d["ma_20_slope"] > 0)),
    ("cvd_slope_5 < 0 AND ma_20_slope < 0", lambda d: (d["cvd_slope_5"] < 0) & (d["ma_20_slope"] < 0)),
    ("mom_3 > 0 AND buy_ratio_trend > 0", lambda d: (d["mom_3"] > 0) & (d["buy_ratio_trend"] > 0)),
    ("mom_3 < 0 AND buy_ratio_trend < 0", lambda d: (d["mom_3"] < 0) & (d["buy_ratio_trend"] < 0)),
    ("ma_20_50_cross > 0 AND cvd_slope_5 > 0", lambda d: (d["ma_20_50_cross"] > 0) & (d["cvd_slope_5"] > 0)),
    ("ma_20_50_cross < 0 AND cvd_slope_5 < 0", lambda d: (d["ma_20_50_cross"] < 0) & (d["cvd_slope_5"] < 0)),
    ("taker_buy_pct_ma5 > 0.52 AND mom_3 > 0", lambda d: (d["taker_buy_pct_ma5"] > 0.52) & (d["mom_3"] > 0)),
    ("taker_buy_pct_ma5 < 0.48 AND mom_3 < 0", lambda d: (d["taker_buy_pct_ma5"] < 0.48) & (d["mom_3"] < 0)),
    ("obv_slope_norm > 0 AND ma_20_slope > 0", lambda d: (d["obv_slope_norm"] > 0) & (d["ma_20_slope"] > 0)),
    ("obv_slope_norm < 0 AND ma_20_slope < 0", lambda d: (d["obv_slope_norm"] < 0) & (d["ma_20_slope"] < 0)),
    ("cvd_slope_5 > 0 AND mom_3 > 0 AND ma_20_slope > 0", lambda d: (d["cvd_slope_5"] > 0) & (d["mom_3"] > 0) & (d["ma_20_slope"] > 0)),
    ("cvd_slope_5 < 0 AND mom_3 < 0 AND ma_20_slope < 0", lambda d: (d["cvd_slope_5"] < 0) & (d["mom_3"] < 0) & (d["ma_20_slope"] < 0)),
    # Stronger filter combos
    ("range_pos_20 < 0.3 AND cvd_slope_5 > 0 AND ma_20_slope > 0", lambda d: (d["range_pos_20"] < 0.3) & (d["cvd_slope_5"] > 0) & (d["ma_20_slope"] > 0)),
    ("range_pos_20 > 0.7 AND cvd_slope_5 < 0 AND ma_20_slope < 0", lambda d: (d["range_pos_20"] > 0.7) & (d["cvd_slope_5"] < 0) & (d["ma_20_slope"] < 0)),
    ("vwap_dist < -0.001 AND cvd_slope_5 > 0", lambda d: (d["vwap_dist"] < -0.001) & (d["cvd_slope_5"] > 0)),
    ("vwap_dist > 0.001 AND cvd_slope_5 < 0", lambda d: (d["vwap_dist"] > 0.001) & (d["cvd_slope_5"] < 0)),
]

for label, fn in combos:
    try:
        mask = fn(opp)
        sub = opp[mask]
    except Exception:
        continue
    if len(sub) < 100:
        print(f"  {label:<60s} SKIP (n={len(sub)})")
        continue

    # Direction depends on combo
    is_long = ">" in label.split("AND")[0] and "< 0" not in label.split("AND")[0]

    if is_long:
        wr = (sub["fwd_6"] > FEE).mean()
        avg_ret = sub["fwd_6"].mean() * 100
        direction = "LONG"
    else:
        wr = (sub["fwd_6"] < -FEE).mean()
        avg_ret = -sub["fwd_6"].mean() * 100
        direction = "SHORT"

    selectivity = len(sub) / len(opp)
    marker = " <<<" if wr > 0.55 else ""

    print(f"  {label}")
    print(f"    → {direction} WR={wr:.3f} avg_ret={avg_ret:+.3f}% n={len(sub):,} select={selectivity:.3f}{marker}")

# ── 8. Temporal stability check ──
print(f"\n{'='*80}")
print("── Temporal Stability: Best combo by year ──\n")

# Test the triple combo (cvd + mom + ma)
opp_c = opp.copy()
opp_c["year"] = opp_c.index.year
long_mask = (opp_c["cvd_slope_5"] > 0) & (opp_c["mom_3"] > 0) & (opp_c["ma_20_slope"] > 0)
short_mask = (opp_c["cvd_slope_5"] < 0) & (opp_c["mom_3"] < 0) & (opp_c["ma_20_slope"] < 0)

print("  Triple combo (cvd_slope + mom + ma_slope):")
for yr in sorted(opp_c["year"].unique()):
    yr_data = opp_c[opp_c["year"] == yr]
    # Long
    l_sub = yr_data[long_mask.reindex(yr_data.index, fill_value=False)]
    s_sub = yr_data[short_mask.reindex(yr_data.index, fill_value=False)]

    l_wr = (l_sub["fwd_6"] > FEE).mean() if len(l_sub) > 10 else float("nan")
    s_wr = (s_sub["fwd_6"] < -FEE).mean() if len(s_sub) > 10 else float("nan")
    print(f"    {yr}: LONG WR={l_wr:.3f} (n={len(l_sub):,}) | SHORT WR={s_wr:.3f} (n={len(s_sub):,})")

# ── 9. Summary ──
print(f"\n{'='*80}")
print("=== SUMMARY ===")
print("Key question: Any single feature > 55% directional WR in opportunity zones?")
above_55 = [r for r in results if max(r["wr_hi"], 1-r["wr_lo"]) > 0.55]
if above_55:
    print(f"  YES: {len(above_55)} features exceed 55%")
    for r in above_55:
        print(f"    - {r['feature']}: hi={r['wr_hi']:.3f}, lo_complement={1-r['wr_lo']:.3f}")
else:
    print("  NO single feature exceeds 55% directional WR alone.")
    best = results[0] if results else None
    if best:
        print(f"  Best: {best['feature']} IC={best['ic']:+.4f} Δ_WR={best['delta_wr']:+.3f}")
print("\nCheck combination results above for multi-feature signals.")
