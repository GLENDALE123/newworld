"""BTC 5m 스캘핑 이벤트 탐색 v2 — book_depth + metrics 포함."""

import pandas as pd
import numpy as np

print("=== BTC 5m Scalping Event Exploration v2 ===\n")

# ── 1. 데이터 로딩 ──
kline = pd.read_parquet("data/merged/BTCUSDT/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet("data/merged/BTCUSDT/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet("data/merged/BTCUSDT/book_depth.parquet")
metrics = pd.read_parquet("data/merged/BTCUSDT/metrics.parquet").set_index("timestamp").sort_index()
bt = pd.read_parquet("data/merged/BTCUSDT/book_ticker.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet("data/merged/BTCUSDT/funding_rate.parquet").set_index("timestamp").sort_index()

# TZ alignment
for d in [tick, bt]:
    if d.index.tz is None and kline.index.tz is not None:
        d.index = d.index.tz_localize(kline.index.tz)

print(f"5m: {len(kline):,} bars ({kline.index.min().date()} ~ {kline.index.max().date()})")
print(f"book_depth: {len(bd):,} rows ({bd['timestamp'].min()} ~ {bd['timestamp'].max()})")
print(f"metrics: {len(metrics):,} rows ({metrics.index.min()} ~ {metrics.index.max()})")
print(f"book_ticker: {len(bt):,} rows ({bt.index.min()} ~ {bt.index.max()})")

# ── 2. Forward returns ──
c = kline["close"]; o = kline["open"]; h = kline["high"]; l = kline["low"]; v = kline["volume"]
for n in [3, 4, 5, 6]:
    kline[f"fwd_{n}"] = c.shift(-n) / c - 1
kline["fwd_max"] = kline[["fwd_3", "fwd_4", "fwd_5", "fwd_6"]].max(axis=1)
kline["fwd_min"] = kline[["fwd_3", "fwd_4", "fwd_5", "fwd_6"]].min(axis=1)
kline["fwd_best_long"] = kline["fwd_max"]
kline["fwd_best_short"] = -kline["fwd_min"]
FEE = 0.0008

# ── 3. 기존 피처 (v1과 동일) ──
kline["ret"] = c.pct_change()
kline["atr_12"] = (h - l).rolling(12).mean()
kline["atr_ratio"] = (h - l) / kline["atr_12"]
kline["vol_20"] = kline["ret"].rolling(20).std()
kline["vol_60"] = kline["ret"].rolling(60).std()
kline["vol_squeeze"] = kline["vol_20"] / kline["vol_60"]
kline["vol_ma_12"] = v.rolling(12).mean()
kline["vol_ratio"] = v / kline["vol_ma_12"]

# Tick bar → 5m CVD
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"})
kline = kline.drop(columns=["trade_count"], errors="ignore")
kline = kline.join(tick_5m[["buy_volume", "sell_volume", "cvd_raw", "trade_count"]], how="left")
kline["buy_ratio"] = kline["buy_volume"] / (kline["buy_volume"] + kline["sell_volume"])

# Session
kline["hour"] = kline.index.hour
def session(h):
    if 0 <= h < 8: return "Asia"
    elif 8 <= h < 14: return "Europe"
    elif 14 <= h < 22: return "US"
    else: return "Late"
kline["session"] = kline["hour"].apply(session)

# Candle
kline["body_ratio"] = (c - o) / (h - l + 1e-10)
bull = (c > o).astype(int)
kline["bull_count_5"] = bull.rolling(5).sum()
kline["bear_count_5"] = (c < o).astype(int).rolling(5).sum()
kline["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)

# ── 4. NEW: Book Depth 피처 ──
print("\n[Processing book_depth...]")
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and kline.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(kline.index.tz)

# Pivot: percentage levels as columns, aggregate to 5min
# Key levels: ±0.5%, ±1%, ±2%, ±5%
key_pcts = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]
bd_key = bd[bd["percentage"].isin(key_pcts)].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")

# Pivot to get bid/ask depth at each level per 5min
bd_pivot = bd_key.pivot_table(
    index="ts_5m", columns="percentage", values="notional", aggfunc="last"
)
bd_pivot.columns = [f"depth_{c}" for c in bd_pivot.columns]

# Depth imbalance: bid vs ask at various levels
# Negative pct = bid side, positive = ask side
if "depth_-0.5" in bd_pivot.columns and "depth_0.5" in bd_pivot.columns:
    bd_pivot["depth_imb_05"] = (bd_pivot["depth_-0.5"] - bd_pivot["depth_0.5"]) / (bd_pivot["depth_-0.5"] + bd_pivot["depth_0.5"] + 1e-10)
if "depth_-1.0" in bd_pivot.columns and "depth_1.0" in bd_pivot.columns:
    bd_pivot["depth_imb_10"] = (bd_pivot["depth_-1.0"] - bd_pivot["depth_1.0"]) / (bd_pivot["depth_-1.0"] + bd_pivot["depth_1.0"] + 1e-10)
if "depth_-2.0" in bd_pivot.columns and "depth_2.0" in bd_pivot.columns:
    bd_pivot["depth_imb_20"] = (bd_pivot["depth_-2.0"] - bd_pivot["depth_2.0"]) / (bd_pivot["depth_-2.0"] + bd_pivot["depth_2.0"] + 1e-10)
if "depth_-5.0" in bd_pivot.columns and "depth_5.0" in bd_pivot.columns:
    bd_pivot["depth_imb_50"] = (bd_pivot["depth_-5.0"] - bd_pivot["depth_5.0"]) / (bd_pivot["depth_-5.0"] + bd_pivot["depth_5.0"] + 1e-10)

# Total near depth (within 1%)
near_bid_cols = [c for c in bd_pivot.columns if c.startswith("depth_-") and float(c.split("_")[1]) >= -1.0]
near_ask_cols = [c for c in bd_pivot.columns if c.startswith("depth_") and not c.startswith("depth_-") and "imb" not in c and float(c.split("_")[1].replace("depth","")) <= 1.0]

kline = kline.join(bd_pivot, how="left")
print(f"  book_depth features: {[c for c in bd_pivot.columns if 'imb' in c]}")

# ── 5. NEW: Metrics 피처 ──
print("[Processing metrics...]")
# OI, long/short ratios
metrics_cols = ["sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio"]
for col in metrics_cols:
    if col in metrics.columns:
        kline[col] = metrics[col].reindex(kline.index, method="ffill")

# OI change
if "sum_open_interest_value" in kline.columns:
    kline["oi_change_5"] = kline["sum_open_interest_value"].pct_change(5)
    kline["oi_change_12"] = kline["sum_open_interest_value"].pct_change(12)

# Taker LS ratio change
if "sum_taker_long_short_vol_ratio" in kline.columns:
    kline["taker_ls_ma5"] = kline["sum_taker_long_short_vol_ratio"].rolling(5).mean()

print(f"  metrics features added")

# ── 6. NEW: Book Ticker 피처 ──
print("[Processing book_ticker...]")
if bt.index.tz is None and kline.index.tz is not None:
    bt.index = bt.index.tz_localize(kline.index.tz)
bt_5m = bt.resample("5min").agg({
    "spread_bps": "mean",
    "obi": "mean",
})
bt_5m.columns = ["spread_bps_mean", "obi_mean"]
kline = kline.join(bt_5m, how="left")

# OBI trend
kline["obi_ma5"] = kline["obi_mean"].rolling(5).mean()
kline["obi_ma12"] = kline["obi_mean"].rolling(12).mean()
kline["obi_trend"] = kline["obi_ma5"] - kline["obi_ma12"]

# Funding
kline["funding_rate"] = funding["funding_rate"].reindex(kline.index, method="ffill") if "funding_rate" in funding.columns else np.nan

print(f"\n  Total columns: {kline.shape[1]}")

# ── 7. Analysis ──
df = kline.iloc[60:].dropna(subset=["fwd_3", "fwd_6"]).copy()
print(f"\nAnalysis rows: {len(df):,}")

# Check data coverage for new features
for feat in ["depth_imb_05", "depth_imb_10", "obi_mean", "sum_open_interest_value",
             "sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio", "spread_bps_mean"]:
    if feat in df.columns:
        cov = df[feat].notna().mean()
        print(f"  {feat}: {cov:.1%} coverage")

# ── 8. Forward return distribution (전체 + book_depth 기간) ──
print(f"\n{'='*70}")
print("── Forward Return Distribution ──")
# book_depth 기간만 (2023+)
df_bd = df[df.index >= "2023-01-01"]
print(f"\nFull period: {len(df):,} bars")
print(f"Book depth period (2023+): {len(df_bd):,} bars")

for period_label, sub in [("Full", df), ("2023+ (w/ book_depth)", df_bd)]:
    s = sub["fwd_best_long"].dropna()
    prof_long = (s > FEE).mean()
    prof_short = (sub["fwd_best_short"] > FEE).mean()
    print(f"\n  [{period_label}]")
    print(f"    Long profitable (best 3-6bar > fee): {prof_long:.1%}")
    print(f"    Short profitable: {prof_short:.1%}")
    print(f"    P95 long: {s.quantile(0.95)*100:.3f}%, P99: {s.quantile(0.99)*100:.3f}%")

# ── 9. Profile function ──
def profile(subset, label, extra_cols=None):
    print(f"\n  [{label}] n={len(subset):,}")
    print(f"  vol_squeeze:    mean={subset['vol_squeeze'].mean():.3f}")
    print(f"  atr_ratio:      mean={subset['atr_ratio'].mean():.3f}")
    print(f"  vol_20:         mean={subset['vol_20'].mean()*100:.4f}%")
    print(f"  vol_ratio:      mean={subset['vol_ratio'].mean():.3f}")
    print(f"  buy_ratio:      mean={subset['buy_ratio'].mean():.3f}")
    print(f"  trade_count:    mean={subset['trade_count'].mean():.0f}")
    sess = subset["session"].value_counts(normalize=True)
    print(f"  Session: {', '.join(f'{k}={v:.1%}' for k,v in sess.items())}")

    # NEW features
    for col, name in [
        ("depth_imb_05", "depth_imb ±0.5%"),
        ("depth_imb_10", "depth_imb ±1.0%"),
        ("depth_imb_20", "depth_imb ±2.0%"),
        ("obi_mean", "OBI (book_ticker)"),
        ("obi_trend", "OBI trend (5-12)"),
        ("spread_bps_mean", "spread (bps)"),
        ("sum_open_interest_value", "OI value"),
        ("oi_change_5", "OI change 5bar"),
        ("oi_change_12", "OI change 12bar"),
        ("sum_taker_long_short_vol_ratio", "taker L/S ratio"),
        ("taker_ls_ma5", "taker L/S ma5"),
        ("count_toptrader_long_short_ratio", "toptrader L/S"),
        ("count_long_short_ratio", "retail L/S"),
        ("funding_rate", "funding rate"),
    ]:
        if col in subset.columns:
            vals = subset[col].dropna()
            if len(vals) > 10:
                print(f"  {name:25s} mean={vals.mean():.6f}, median={vals.median():.6f}")

# ── 10. Top/Bottom 5% with new features (2023+ period) ──
print(f"\n{'='*70}")
print("── Top/Bottom 5% Analysis (2023+ period, with book_depth) ──")

q95 = df_bd["fwd_best_long"].quantile(0.95)
q05 = df_bd["fwd_best_long"].quantile(0.05)
top5 = df_bd[df_bd["fwd_best_long"] >= q95]
bot5 = df_bd[df_bd["fwd_best_long"] <= q05]
mid = df_bd[df_bd["fwd_best_long"].between(df_bd["fwd_best_long"].quantile(0.40), df_bd["fwd_best_long"].quantile(0.60))]

print(f"\nTop5% threshold: >{q95*100:.3f}%")
profile(top5, "TOP 5% (Long)")
profile(bot5, "BOTTOM 5% (worst for long)")
profile(mid, "MIDDLE 40-60% (noise/chop)")

# Short side
q95s = df_bd["fwd_best_short"].quantile(0.95)
top5s = df_bd[df_bd["fwd_best_short"] >= q95s]
print(f"\n  Short top5% threshold: >{q95s*100:.3f}%")
profile(top5s, "TOP 5% (Short)")

# ── 11. Feature importance: difference between top5 and middle ──
print(f"\n{'='*70}")
print("── Feature Contrast: Top5% vs Middle (normalized difference) ──")
print(f"  Positive = higher in top5%, Negative = lower")

contrast_features = [
    "vol_squeeze", "atr_ratio", "vol_ratio", "buy_ratio", "trade_count",
    "depth_imb_05", "depth_imb_10", "depth_imb_20",
    "obi_mean", "obi_trend", "spread_bps_mean",
    "oi_change_5", "oi_change_12",
    "sum_taker_long_short_vol_ratio", "taker_ls_ma5",
    "count_toptrader_long_short_ratio", "count_long_short_ratio",
    "funding_rate",
]

diffs = []
for col in contrast_features:
    if col not in df_bd.columns:
        continue
    t = top5[col].dropna()
    m = mid[col].dropna()
    if len(t) < 10 or len(m) < 10:
        continue
    t_mean = t.mean()
    m_mean = m.mean()
    m_std = m.std()
    if m_std > 0:
        z = (t_mean - m_mean) / m_std
    else:
        z = 0
    diffs.append({"feature": col, "top5_mean": t_mean, "mid_mean": m_mean, "z_score": z})

diffs.sort(key=lambda x: abs(x["z_score"]), reverse=True)
for d in diffs:
    bar = "█" * min(int(abs(d["z_score"]) * 5), 30)
    sign = "+" if d["z_score"] > 0 else "-"
    print(f"  {d['feature']:40s} z={d['z_score']:+.3f} {sign}{bar}")

# ── 12. Conditional: book depth imbalance + opportunity ──
print(f"\n{'='*70}")
print("── Book Depth Imbalance → Profitability ──")
for imb_col, label in [("depth_imb_05", "±0.5%"), ("depth_imb_10", "±1.0%"), ("depth_imb_20", "±2.0%")]:
    if imb_col not in df_bd.columns:
        continue
    valid = df_bd[df_bd[imb_col].notna()]
    if len(valid) < 1000:
        print(f"  {label}: insufficient data")
        continue

    for q_label, q_lo, q_hi in [("Strong bid wall (top 20%)", 0.80, 1.0),
                                  ("Neutral (40-60%)", 0.40, 0.60),
                                  ("Strong ask wall (bottom 20%)", 0.0, 0.20)]:
        lo = valid[imb_col].quantile(q_lo)
        hi = valid[imb_col].quantile(q_hi)
        sub = valid[valid[imb_col].between(lo, hi)]
        if len(sub) < 50:
            continue
        long_prof = (sub["fwd_best_long"] > FEE).mean()
        short_prof = (sub["fwd_best_short"] > FEE).mean()
        avg_fwd6 = sub["fwd_6"].mean() * 100
        print(f"  {label} {q_label}: n={len(sub):,} | long_prof={long_prof:.1%} | short_prof={short_prof:.1%} | avg_fwd6={avg_fwd6:+.4f}%")

# ── 13. OI change → profitability ──
print(f"\n{'='*70}")
print("── OI Change → Profitability ──")
if "oi_change_5" in df_bd.columns:
    valid = df_bd[df_bd["oi_change_5"].notna()]
    for label, lo_q, hi_q in [("OI surging (top 10%)", 0.90, 1.0),
                                ("OI stable (40-60%)", 0.40, 0.60),
                                ("OI dropping (bottom 10%)", 0.0, 0.10)]:
        lo = valid["oi_change_5"].quantile(lo_q)
        hi = valid["oi_change_5"].quantile(hi_q)
        sub = valid[valid["oi_change_5"].between(lo, hi)]
        if len(sub) < 50:
            continue
        long_prof = (sub["fwd_best_long"] > FEE).mean()
        short_prof = (sub["fwd_best_short"] > FEE).mean()
        avg_vol = sub["vol_20"].mean() * 100
        print(f"  {label}: n={len(sub):,} | long={long_prof:.1%} | short={short_prof:.1%} | vol={avg_vol:.3f}%")

# ── 14. Taker L/S ratio → direction ──
print(f"\n{'='*70}")
print("── Taker L/S Ratio → Direction ──")
if "sum_taker_long_short_vol_ratio" in df_bd.columns:
    valid = df_bd[df_bd["sum_taker_long_short_vol_ratio"].notna()]
    for label, lo_q, hi_q in [("Heavy long (>1.5)", None, None),
                                ("Balanced (0.8-1.2)", None, None),
                                ("Heavy short (<0.7)", None, None)]:
        if "Heavy long" in label:
            sub = valid[valid["sum_taker_long_short_vol_ratio"] > 1.5]
        elif "Balanced" in label:
            sub = valid[valid["sum_taker_long_short_vol_ratio"].between(0.8, 1.2)]
        else:
            sub = valid[valid["sum_taker_long_short_vol_ratio"] < 0.7]
        if len(sub) < 50:
            continue
        long_prof = (sub["fwd_best_long"] > FEE).mean()
        short_prof = (sub["fwd_best_short"] > FEE).mean()
        avg_fwd = sub["fwd_6"].mean() * 100
        print(f"  {label}: n={len(sub):,} | long={long_prof:.1%} | short={short_prof:.1%} | fwd6={avg_fwd:+.4f}%")

# ── 15. Spread → profitability ──
print(f"\n{'='*70}")
print("── Spread → Profitability ──")
if "spread_bps_mean" in df_bd.columns:
    valid = df_bd[df_bd["spread_bps_mean"].notna()]
    for label, lo_q, hi_q in [("Tight spread (bottom 20%)", 0.0, 0.20),
                                ("Normal spread (40-60%)", 0.40, 0.60),
                                ("Wide spread (top 20%)", 0.80, 1.0)]:
        lo = valid["spread_bps_mean"].quantile(lo_q)
        hi = valid["spread_bps_mean"].quantile(hi_q)
        sub = valid[valid["spread_bps_mean"].between(lo, hi)]
        if len(sub) < 50:
            continue
        long_prof = (sub["fwd_best_long"] > FEE).mean()
        short_prof = (sub["fwd_best_short"] > FEE).mean()
        print(f"  {label} ({lo:.3f}-{hi:.3f}bps): n={len(sub):,} | long={long_prof:.1%} | short={short_prof:.1%}")

# ── 16. No-trade zone revisited ──
print(f"\n{'='*70}")
print("── No-Trade Zone (2023+, both fwd_best < fee) ──")
no_trade = df_bd[(df_bd["fwd_best_long"] < FEE) & (df_bd["fwd_best_short"] < FEE)]
print(f"No-trade bars: {len(no_trade):,} ({len(no_trade)/len(df_bd)*100:.1f}%)")
profile(no_trade, "NO-TRADE ZONE")

print(f"\n{'='*70}")
print("=== DONE ===")
