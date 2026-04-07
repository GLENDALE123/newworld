"""BTC 5m 스캘핑 이벤트 탐색 — 데이터 분석 전용."""

import pandas as pd
import numpy as np

# ── 1. 데이터 로딩 ──
print("=== BTC 5m Scalping Event Exploration ===\n")
kline = pd.read_parquet("data/merged/BTCUSDT/kline_5m.parquet")
kline = kline.set_index("timestamp").sort_index()
tick = pd.read_parquet("data/merged/BTCUSDT/tick_bar.parquet")
tick = tick.set_index("timestamp").sort_index()
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)
elif tick.index.tz is not None and kline.index.tz is None:
    kline.index = kline.index.tz_localize(tick.index.tz)

print(f"5m bars: {len(kline):,} ({kline.index.min().date()} ~ {kline.index.max().date()})")

# ── 2. Forward returns (3~6 bars = 15~30min) ──
for n in [3, 4, 5, 6]:
    kline[f"fwd_{n}"] = kline["close"].shift(-n) / kline["close"] - 1

# Best forward return within 3-6 bars (MFE-like)
kline["fwd_max"] = kline[["fwd_3", "fwd_4", "fwd_5", "fwd_6"]].max(axis=1)
kline["fwd_min"] = kline[["fwd_3", "fwd_4", "fwd_5", "fwd_6"]].min(axis=1)
# Best absolute opportunity (long or short)
kline["fwd_best_long"] = kline["fwd_max"]
kline["fwd_best_short"] = -kline["fwd_min"]

FEE = 0.0008  # round-trip

print("\n── Forward Return Distribution (close-to-close, 3-6 bars) ──")
for n in [3, 6]:
    col = f"fwd_{n}"
    s = kline[col].dropna()
    print(f"\n  fwd_{n} ({n*5}min):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:02d}: {s.quantile(p/100)*100:+.4f}%")
    print(f"    mean: {s.mean()*100:+.4f}%, std: {s.std()*100:.4f}%")
    print(f"    >+0.08% (profitable long after fee):  {(s > FEE).mean()*100:.1f}%")
    print(f"    <-0.08% (profitable short after fee): {(s < -FEE).mean()*100:.1f}%")

# ── 3. 피처 엔지니어링 (분석용) ──
c = kline["close"]
h = kline["high"]
l = kline["low"]
v = kline["volume"]
o = kline["open"]

# Volatility
kline["ret"] = c.pct_change()
kline["atr_12"] = (h - l).rolling(12).mean()  # 1h ATR
kline["atr_ratio"] = (h - l) / kline["atr_12"]  # current bar vs avg
kline["vol_20"] = kline["ret"].rolling(20).std()  # ~100min vol
kline["vol_60"] = kline["ret"].rolling(60).std()  # ~5h vol
kline["vol_squeeze"] = kline["vol_20"] / kline["vol_60"]  # <1 = squeeze

# Volume
kline["vol_ma_12"] = v.rolling(12).mean()
kline["vol_ratio"] = v / kline["vol_ma_12"]

# CVD from tick_bar: resample to 5m
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({
    "buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"
})
kline = kline.drop(columns=["trade_count"], errors="ignore")
kline = kline.join(tick_5m[["buy_volume", "sell_volume", "cvd_raw", "trade_count"]], how="left")
kline["buy_ratio"] = kline["buy_volume"] / (kline["buy_volume"] + kline["sell_volume"])
kline["cvd_5"] = kline["cvd_raw"].rolling(5).sum()  # 25min CVD

# Time features
kline["hour"] = kline.index.hour
# Sessions: Asia 0-8 UTC, Europe 8-14, US 14-22, Late 22-24
def session(h):
    if 0 <= h < 8: return "Asia"
    elif 8 <= h < 14: return "Europe"
    elif 14 <= h < 22: return "US"
    else: return "Late"
kline["session"] = kline["hour"].apply(session)

# Candle patterns
kline["body"] = (c - o) / (h - l + 1e-10)  # normalized body
kline["upper_wick"] = (h - np.maximum(o, c)) / (h - l + 1e-10)
kline["lower_wick"] = (np.minimum(o, c) - l) / (h - l + 1e-10)
kline["consec_bull"] = 0
kline["consec_bear"] = 0
bull = (c > o).astype(int)
bear = (c < o).astype(int)
# Consecutive count (vectorized)
for i in range(1, 6):
    kline["consec_bull"] += bull.shift(i).fillna(0) * (bull.shift(i-1).fillna(0) if i > 0 else 1)
# Simpler: count of bullish bars in last 5
kline["bull_count_5"] = bull.rolling(5).sum()
kline["bear_count_5"] = bear.rolling(5).sum()

# Range position (where is close in recent range)
kline["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)

# Drop warmup
df = kline.iloc[60:].copy()
df = df.dropna(subset=["fwd_3", "fwd_6"])
print(f"\nAnalysis rows: {len(df):,}")

# ── 4. Top/Bottom 5% 분석 ──
# Long opportunities
q95_long = df["fwd_best_long"].quantile(0.95)
q05_long = df["fwd_best_long"].quantile(0.05)
top5_long = df[df["fwd_best_long"] >= q95_long]
bot5_long = df[df["fwd_best_long"] <= q05_long]
middle = df[(df["fwd_best_long"].between(df["fwd_best_long"].quantile(0.40), df["fwd_best_long"].quantile(0.60)))]

print(f"\n{'='*70}")
print("── Top 5% Long Moments (best fwd return in 3-6 bars) ──")
print(f"Threshold: >{q95_long*100:.3f}%, count: {len(top5_long):,}")
print(f"  Mean return: {top5_long['fwd_best_long'].mean()*100:.3f}%")

def profile(subset, label):
    print(f"\n  [{label}] n={len(subset):,}")
    # Volatility
    print(f"  vol_squeeze (vol20/vol60):  mean={subset['vol_squeeze'].mean():.3f}, median={subset['vol_squeeze'].median():.3f}")
    print(f"  atr_ratio (cur/avg ATR):    mean={subset['atr_ratio'].mean():.3f}, median={subset['atr_ratio'].median():.3f}")
    print(f"  vol_20 (100min vol):        mean={subset['vol_20'].mean()*100:.4f}%")
    # Volume
    print(f"  vol_ratio (v/ma12):         mean={subset['vol_ratio'].mean():.3f}, median={subset['vol_ratio'].median():.3f}")
    # CVD
    print(f"  buy_ratio:                  mean={subset['buy_ratio'].mean():.3f}")
    print(f"  cvd_5 (25min):              mean={subset['cvd_5'].mean():.1f}")
    # Session
    sess = subset["session"].value_counts(normalize=True)
    print(f"  Session: {', '.join(f'{k}={v:.1%}' for k,v in sess.items())}")
    # Hour distribution (top 3)
    hr = subset["hour"].value_counts(normalize=True).head(5)
    print(f"  Top hours: {', '.join(f'{k}h={v:.1%}' for k,v in hr.items())}")
    # Candle patterns
    print(f"  bull_count_5:               mean={subset['bull_count_5'].mean():.2f}")
    print(f"  bear_count_5:               mean={subset['bear_count_5'].mean():.2f}")
    print(f"  body (normalized):          mean={subset['body'].mean():.3f}")
    # Range position
    print(f"  range_pos_20:               mean={subset['range_pos_20'].mean():.3f}, median={subset['range_pos_20'].median():.3f}")
    # Trade count
    if "trade_count" in subset.columns:
        print(f"  trade_count:                mean={subset['trade_count'].mean():.0f}, median={subset['trade_count'].median():.0f}")

profile(top5_long, "TOP 5% (Long)")
profile(bot5_long, "BOTTOM 5% (Long = worst moments)")
profile(middle, "MIDDLE 40-60% (Chop/Noise)")

# ── 5. Short side ──
q95_short = df["fwd_best_short"].quantile(0.95)
top5_short = df[df["fwd_best_short"] >= q95_short]
print(f"\n{'='*70}")
print("── Top 5% Short Moments ──")
print(f"Threshold: >{q95_short*100:.3f}% drop, count: {len(top5_short):,}")
profile(top5_short, "TOP 5% (Short)")

# ── 6. Profitable after fee 분석 ──
print(f"\n{'='*70}")
print("── Profitability After Fee (0.08% RT) ──")
# fwd_3 기준
for direction, col in [("Long", "fwd_3"), ("Short", "fwd_3")]:
    if direction == "Short":
        profitable = df[col] < -FEE
    else:
        profitable = df[col] > FEE
    pct = profitable.mean()
    print(f"\n  {direction} (fwd_3, 15min): {pct*100:.1f}% of bars are profitable after fee")

    if profitable.sum() > 100:
        prof_df = df[profitable]
        unpf_df = df[~profitable]
        print(f"    Profitable moments:")
        print(f"      vol_squeeze: {prof_df['vol_squeeze'].mean():.3f} vs {unpf_df['vol_squeeze'].mean():.3f} (unprofitable)")
        print(f"      vol_ratio:   {prof_df['vol_ratio'].mean():.3f} vs {unpf_df['vol_ratio'].mean():.3f}")
        print(f"      atr_ratio:   {prof_df['atr_ratio'].mean():.3f} vs {unpf_df['atr_ratio'].mean():.3f}")
        print(f"      buy_ratio:   {prof_df['buy_ratio'].mean():.3f} vs {unpf_df['buy_ratio'].mean():.3f}")
        print(f"      trade_count: {prof_df['trade_count'].mean():.0f} vs {unpf_df['trade_count'].mean():.0f}")

# ── 7. Conditional analysis: squeeze → breakout ──
print(f"\n{'='*70}")
print("── Squeeze → Breakout Analysis ──")
squeeze = df["vol_squeeze"] < 0.7
expansion = df["vol_squeeze"] > 1.3
normal = df["vol_squeeze"].between(0.8, 1.2)

for label, mask in [("Squeeze (<0.7)", squeeze), ("Normal (0.8-1.2)", normal), ("Expansion (>1.3)", expansion)]:
    sub = df[mask]
    if len(sub) < 100:
        continue
    long_prof = (sub["fwd_best_long"] > FEE).mean()
    short_prof = (sub["fwd_best_short"] > FEE).mean()
    avg_long = sub["fwd_best_long"].mean() * 100
    avg_short = sub["fwd_best_short"].mean() * 100
    print(f"\n  {label}: n={len(sub):,}")
    print(f"    Long profitable:  {long_prof*100:.1f}%, avg best: {avg_long:.3f}%")
    print(f"    Short profitable: {short_prof*100:.1f}%, avg best: {avg_short:.3f}%")

# ── 8. Volume surge analysis ──
print(f"\n{'='*70}")
print("── Volume Surge Analysis ──")
for vr_label, vr_lo, vr_hi in [("Low (<0.5x)", 0, 0.5), ("Normal (0.7-1.3x)", 0.7, 1.3),
                                 ("High (2-3x)", 2, 3), ("Extreme (>3x)", 3, 100)]:
    mask = df["vol_ratio"].between(vr_lo, vr_hi)
    sub = df[mask]
    if len(sub) < 100:
        continue
    long_prof = (sub["fwd_best_long"] > FEE).mean()
    avg_long = sub["fwd_best_long"].mean() * 100
    avg_short = sub["fwd_best_short"].mean() * 100
    print(f"\n  {vr_label}: n={len(sub):,}")
    print(f"    Long profitable: {long_prof*100:.1f}%, avg best long: {avg_long:.3f}%, avg best short: {avg_short:.3f}%")

# ── 9. Session analysis ──
print(f"\n{'='*70}")
print("── Session Analysis ──")
for sess in ["Asia", "Europe", "US", "Late"]:
    sub = df[df["session"] == sess]
    avg_long = sub["fwd_best_long"].mean() * 100
    avg_short = sub["fwd_best_short"].mean() * 100
    long_prof = (sub["fwd_best_long"] > FEE).mean()
    short_prof = (sub["fwd_best_short"] > FEE).mean()
    avg_spread = sub["atr_ratio"].mean()
    print(f"\n  {sess}: n={len(sub):,}")
    print(f"    Long profitable: {long_prof*100:.1f}%, Short profitable: {short_prof*100:.1f}%")
    print(f"    Avg best long: {avg_long:.3f}%, Avg best short: {avg_short:.3f}%")
    print(f"    Avg atr_ratio: {avg_spread:.3f}")

# ── 10. No-trade zone 특성 ──
print(f"\n{'='*70}")
print("── No-Trade Zone (both fwd_best < fee) ──")
no_trade = (df["fwd_best_long"] < FEE) & (df["fwd_best_short"] < FEE)
nt = df[no_trade]
print(f"No-trade bars: {len(nt):,} ({len(nt)/len(df)*100:.1f}%)")
if len(nt) > 100:
    profile(nt, "NO-TRADE ZONE")

print(f"\n{'='*70}")
print("=== DONE ===")
