"""파생 피처 방향 예측력 테스트 — 로우 피처의 interaction/divergence/composite."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

print("=== Derived Feature Predictability Test ===\n")

# ── 데이터 로딩 (ETH — 비효율 더 크므로) ──
SYMBOL = "ETHUSDT"
kline = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()

for d in [tick]:
    if d.index.tz is None and kline.index.tz is not None:
        d.index = d.index.tz_localize(kline.index.tz)

c = kline["close"]; o = kline["open"]; h = kline["high"]; l = kline["low"]; v = kline["volume"]

# Forward return
kline["fwd_6"] = c.shift(-6) / c - 1
FEE = 0.0008

# ── Raw ingredients ──
kline["ret"] = c.pct_change()
kline["ret_3"] = c / c.shift(3) - 1
kline["ret_6"] = c / c.shift(6) - 1
kline["vol_20"] = kline["ret"].rolling(20).std()
kline["vol_60"] = kline["ret"].rolling(60).std()
kline["vol_squeeze"] = kline["vol_20"] / kline["vol_60"]
kline["atr_12"] = (h - l).rolling(12).mean()
kline["atr_ratio"] = (h - l) / kline["atr_12"]
kline["vol_ma_12"] = v.rolling(12).mean()
kline["vol_ratio"] = v / kline["vol_ma_12"]
kline["ma_20"] = c.rolling(20).mean()
kline["ma_50"] = c.rolling(50).mean()
kline["body_ratio"] = (c - o) / (h - l + 1e-10)
kline["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)
kline["vwap_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()
kline["vwap_dist"] = c / kline["vwap_20"] - 1

# Tick bar
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"})
kline = kline.drop(columns=["trade_count"], errors="ignore")
kline = kline.join(tick_5m[["buy_volume", "sell_volume", "cvd_raw", "trade_count"]], how="left")
kline["buy_ratio"] = kline["buy_volume"] / (kline["buy_volume"] + kline["sell_volume"])
kline["cvd_cumsum"] = kline["cvd_raw"].cumsum()
kline["cvd_slope_5"] = kline["cvd_cumsum"] - kline["cvd_cumsum"].shift(5)
kline["taker_buy_pct"] = kline["taker_buy_volume"] / kline["volume"]

# Book depth
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and kline.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(kline.index.tz)
key_pcts = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]
bd_key = bd[bd["percentage"].isin(key_pcts)].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_pivot = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
bd_pivot.columns = [f"depth_{c}" for c in bd_pivot.columns]
for neg, pos, tag in [(-1.0, 1.0, "10"), (-2.0, 2.0, "20"), (-0.5, 0.5, "05")]:
    cn, cp = f"depth_{neg}", f"depth_{pos}"
    if cn in bd_pivot.columns and cp in bd_pivot.columns:
        bd_pivot[f"depth_imb_{tag}"] = (bd_pivot[cn] - bd_pivot[cp]) / (bd_pivot[cn] + bd_pivot[cp] + 1e-10)
# Total near depth
if "depth_-1.0" in bd_pivot.columns and "depth_1.0" in bd_pivot.columns:
    bd_pivot["total_depth_10"] = bd_pivot["depth_-1.0"] + bd_pivot["depth_1.0"]
kline = kline.join(bd_pivot, how="left")

# Metrics
for col in ["sum_open_interest_value", "sum_taker_long_short_vol_ratio",
            "count_toptrader_long_short_ratio", "count_long_short_ratio"]:
    if col in metrics.columns:
        kline[col] = metrics[col].reindex(kline.index, method="ffill")
if "sum_open_interest_value" in kline.columns:
    kline["oi_change_5"] = kline["sum_open_interest_value"].pct_change(5)
kline["funding_rate"] = funding["funding_rate"].reindex(kline.index, method="ffill") if "funding_rate" in funding.columns else np.nan

# ══════════════════════════════════════════════════════════════
# ██  DERIVED FEATURES  ██
# ══════════════════════════════════════════════════════════════
print("[Building derived features...]")

# ── 1. DIVERGENCE 계열 ──
# Price↑ but CVD↓ = 매수세 없는 상승 → 반전 short 시그널
kline["div_price_cvd"] = kline["ret_3"] - (kline["cvd_slope_5"] / (kline["cvd_slope_5"].rolling(20).std() + 1e-10)) * kline["vol_20"]
# 단순 부호 divergence
kline["div_sign_price_cvd"] = np.sign(kline["ret_3"]) * np.sign(-kline["cvd_slope_5"])  # +1 = divergence

# Price↑ but OI↓ = 숏 청산 랠리 (약한 상승) → 반전
if "oi_change_5" in kline.columns:
    kline["div_price_oi"] = kline["ret_3"] * (-kline["oi_change_5"])  # positive = price up & OI down

# Price↑ but depth wall↓ (지지 약화 중 상승)
if "depth_imb_10" in kline.columns:
    di_change = kline["depth_imb_10"] - kline["depth_imb_10"].shift(3)
    kline["div_price_depth"] = kline["ret_3"] * (-di_change)  # positive = price up & bid support weakening

# Volume↑ but Price flat = 흡수 (absorption)
kline["absorption"] = kline["vol_ratio"] / (abs(kline["ret"]) / kline["vol_20"] + 1e-10)

# ── 2. INTERACTION 계열 ──
# Depth imbalance × mean-reversion signal
if "depth_imb_10" in kline.columns:
    kline["depth_x_vwap"] = kline["depth_imb_10"] * (-kline["vwap_dist"])  # bid wall + below VWAP = strong long
    kline["depth_x_mom"] = kline["depth_imb_10"] * (-kline["ret_3"])  # bid wall + price dropped = bounce
    kline["depth_x_range"] = kline["depth_imb_10"] * (0.5 - kline["range_pos_20"])  # bid wall + low in range

# CVD slope × vol ratio (강한 CVD + 높은 거래량 = 확신 있는 흐름)
kline["cvd_x_vol"] = kline["cvd_slope_5"] * kline["vol_ratio"]

# Taker ratio × momentum (군중이 한쪽으로 몰림 × 가격 방향)
kline["taker_x_mom"] = kline["taker_buy_pct"] * kline["ret_3"]

# OI change × momentum (OI 증가 + 가격 상승 = 신규 롱 진입)
if "oi_change_5" in kline.columns:
    kline["oi_x_mom"] = kline["oi_change_5"] * kline["ret_3"]

# ── 3. COMPOSITE 계열 ──
# Mean-reversion score (여러 반전 시그널 합산, z-score 정규화)
def zscore_rolling(s, w=60):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-10)

kline["z_vwap"] = zscore_rolling(-kline["vwap_dist"])  # VWAP 아래면 +
kline["z_range"] = zscore_rolling(0.5 - kline["range_pos_20"])  # range 하단이면 +
kline["z_mom"] = zscore_rolling(-kline["ret_3"])  # 하락했으면 +
kline["z_cvd"] = zscore_rolling(-kline["cvd_slope_5"])  # CVD 하락이면 +

kline["mr_score"] = (kline["z_vwap"] + kline["z_range"] + kline["z_mom"] + kline["z_cvd"]) / 4

# Depth-enhanced MR score (depth 방향과 결합)
if "depth_imb_10" in kline.columns:
    kline["z_depth"] = zscore_rolling(kline["depth_imb_10"])
    kline["mr_depth_score"] = (kline["z_vwap"] + kline["z_range"] + kline["z_mom"] + kline["z_depth"]) / 4

# Liquidation pressure proxy: OI dropping + high vol + directional move
if "oi_change_5" in kline.columns:
    kline["liq_pressure"] = (-kline["oi_change_5"]) * kline["vol_ratio"] * abs(kline["ret_3"])

# Smart money divergence: toptrader vs retail
if "count_toptrader_long_short_ratio" in kline.columns and "count_long_short_ratio" in kline.columns:
    kline["smart_retail_div"] = kline["count_toptrader_long_short_ratio"] - kline["count_long_short_ratio"]

# Volatility regime change speed
kline["vol_accel"] = kline["vol_squeeze"] - kline["vol_squeeze"].shift(3)

# Microstructure: trade intensity (trades per volume)
kline["trade_intensity"] = kline["trade_count"] / (kline["volume"] + 1e-10)
kline["trade_intensity_z"] = zscore_rolling(kline["trade_intensity"])

# Order flow toxicity proxy: |price_change| / volume
kline["toxicity"] = abs(kline["ret"]) / (kline["volume"] / kline["vol_ma_12"] + 1e-10)
kline["toxicity_z"] = zscore_rolling(kline["toxicity"])

# Depth wall velocity: how fast is the depth imbalance changing
if "depth_imb_10" in kline.columns:
    kline["depth_velocity"] = kline["depth_imb_10"].diff(1)
    kline["depth_accel"] = kline["depth_velocity"].diff(1)

# Funding rate × position crowding
if "funding_rate" in kline.columns and "sum_taker_long_short_vol_ratio" in kline.columns:
    kline["funding_x_crowd"] = kline["funding_rate"] * kline["sum_taker_long_short_vol_ratio"]

# ── 4. PATTERN 계열 ──
# Squeeze breakout detector: vol was low, now expanding
kline["squeeze_break"] = (kline["vol_squeeze"].shift(1) < 0.7).astype(float) * kline["atr_ratio"]

# Exhaustion candle: big body + big volume but at range extreme
kline["exhaustion_long"] = (kline["body_ratio"] > 0.5) * kline["vol_ratio"] * kline["range_pos_20"]
kline["exhaustion_short"] = (kline["body_ratio"] < -0.5) * kline["vol_ratio"] * (1 - kline["range_pos_20"])

# Hammer/shooting star proxy
kline["lower_wick_ratio"] = (np.minimum(o, c) - l) / (h - l + 1e-10)
kline["upper_wick_ratio"] = (h - np.maximum(o, c)) / (h - l + 1e-10)
kline["hammer_score"] = kline["lower_wick_ratio"].rolling(3).max() * (1 - kline["range_pos_20"])  # big lower wick at bottom
kline["shooting_star_score"] = kline["upper_wick_ratio"].rolling(3).max() * kline["range_pos_20"]  # big upper wick at top

# ══════════════════════════════════════════════════════════════
# ██  TEST  ██
# ══════════════════════════════════════════════════════════════
df = kline.loc["2023-01-01":].iloc[60:].dropna(subset=["fwd_6"]).copy()
opp = df[(df["vol_squeeze"] > 1.0) & (df["vol_ratio"] > 1.5)].copy()
print(f"\n2023+ bars: {len(df):,}, Opportunity zone: {len(opp):,} ({len(opp)/len(df)*100:.1f}%)")

derived_features = {
    # Divergence
    "div_price_cvd": "Price-CVD divergence",
    "div_sign_price_cvd": "Price-CVD sign divergence",
    "div_price_oi": "Price-OI divergence",
    "div_price_depth": "Price-Depth divergence",
    "absorption": "Absorption (vol/move)",
    # Interaction
    "depth_x_vwap": "Depth × VWAP distance",
    "depth_x_mom": "Depth × Momentum",
    "depth_x_range": "Depth × Range position",
    "cvd_x_vol": "CVD × Volume ratio",
    "taker_x_mom": "Taker × Momentum",
    "oi_x_mom": "OI × Momentum",
    # Composite
    "mr_score": "MR composite (4-factor)",
    "mr_depth_score": "MR+Depth composite",
    "liq_pressure": "Liquidation pressure",
    "smart_retail_div": "Smart-Retail divergence",
    "vol_accel": "Vol regime acceleration",
    "trade_intensity_z": "Trade intensity (z)",
    "toxicity_z": "Toxicity (z)",
    "depth_velocity": "Depth wall velocity",
    "depth_accel": "Depth wall acceleration",
    "funding_x_crowd": "Funding × Crowding",
    # Pattern
    "squeeze_break": "Squeeze breakout",
    "exhaustion_long": "Exhaustion (bullish)",
    "exhaustion_short": "Exhaustion (bearish)",
    "hammer_score": "Hammer score",
    "shooting_star_score": "Shooting star score",
}

# Also include best raw features for comparison
raw_features = {
    "vwap_dist": "[RAW] VWAP distance",
    "ret_3": "[RAW] Momentum 3bar",
    "depth_imb_10": "[RAW] Depth imb ±1%",
    "cvd_slope_5": "[RAW] CVD slope 5bar",
    "range_pos_20": "[RAW] Range pos 20bar",
}

all_features = {**raw_features, **derived_features}
target = opp["fwd_6"]
results = []

print(f"\n{'='*95}")
print(f"{'Feature':<35s} {'Corr':>7s} {'IC':>7s} {'WR_hi':>7s} {'WR_lo':>7s} {'Δ_WR':>7s} {'n':>8s}")
print(f"{'-'*95}")

for feat_col, feat_name in all_features.items():
    if feat_col not in opp.columns:
        continue
    x = opp[feat_col]
    y = target
    valid = ~(x.isna() | y.isna() | np.isinf(x))
    if valid.sum() < 300:
        print(f"  {feat_name:<33s} SKIP (n={valid.sum()})")
        continue

    xv = x[valid].values.astype(np.float64)
    yv = y[valid].values.astype(np.float64)

    corr = np.corrcoef(xv, yv)[0, 1]
    ic, _ = spearmanr(xv, yv)
    med = np.median(xv)
    hi = xv >= med; lo = xv < med
    wr_hi = (yv[hi] > FEE).mean()
    wr_lo = (yv[lo] > FEE).mean()
    delta = wr_hi - wr_lo

    is_derived = feat_col not in raw_features
    results.append({"feature": feat_name, "col": feat_col, "corr": corr, "ic": ic,
                     "wr_hi": wr_hi, "wr_lo": wr_lo, "delta_wr": delta, "n": int(valid.sum()),
                     "derived": is_derived})
    print(f"  {feat_name:<33s} {corr:+.4f} {ic:+.4f} {wr_hi:.3f} {wr_lo:.3f} {delta:+.4f} {int(valid.sum()):>8d}")

results.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"\n{'='*95}")
print("── Ranked by |IC| ──")
for i, r in enumerate(results):
    tag = "DERIVED" if r["derived"] else "RAW"
    beaten = " ★" if r["derived"] and abs(r["ic"]) > 0.088 else ""  # beats best raw
    print(f"  {i+1:2d}. [{tag:7s}] {r['feature']:<33s} IC={r['ic']:+.4f} Δ_WR={r['delta_wr']:+.4f}{beaten}")

# ── Best derived combos ──
print(f"\n{'='*95}")
print("── Best Derived Feature Combinations ──\n")

def eval_combo(label, mask, direction, data=opp):
    sub = data[mask]
    n = len(sub)
    if n < 50:
        print(f"  {label:<65s} SKIP (n={n})")
        return
    if direction == "LONG":
        wr = (sub["fwd_6"] > FEE).mean()
        avg = sub["fwd_6"].mean() * 100
    else:
        wr = (sub["fwd_6"] < -FEE).mean()
        avg = -sub["fwd_6"].mean() * 100
    sel = n / len(data)
    marker = " <<<" if wr > 0.55 else " **" if wr > 0.52 else ""
    print(f"  {label:<65s} {direction} WR={wr:.3f} avg={avg:+.4f}% n={n:,} sel={sel:.3f}{marker}")

# MR score based
if "mr_score" in opp.columns:
    q80 = opp["mr_score"].quantile(0.80)
    q20 = opp["mr_score"].quantile(0.20)
    eval_combo("MR score > Q80 (strong mean-reversion long)", opp["mr_score"] > q80, "LONG")
    eval_combo("MR score < Q20 (strong mean-reversion short)", opp["mr_score"] < q20, "SHORT")
    q90 = opp["mr_score"].quantile(0.90)
    q10 = opp["mr_score"].quantile(0.10)
    eval_combo("MR score > Q90 (extreme mean-reversion long)", opp["mr_score"] > q90, "LONG")
    eval_combo("MR score < Q10 (extreme mean-reversion short)", opp["mr_score"] < q10, "SHORT")

if "mr_depth_score" in opp.columns:
    q80 = opp["mr_depth_score"].quantile(0.80)
    q20 = opp["mr_depth_score"].quantile(0.20)
    q90 = opp["mr_depth_score"].quantile(0.90)
    q10 = opp["mr_depth_score"].quantile(0.10)
    eval_combo("MR+Depth score > Q80", opp["mr_depth_score"] > q80, "LONG")
    eval_combo("MR+Depth score < Q20", opp["mr_depth_score"] < q20, "SHORT")
    eval_combo("MR+Depth score > Q90", opp["mr_depth_score"] > q90, "LONG")
    eval_combo("MR+Depth score < Q10", opp["mr_depth_score"] < q10, "SHORT")

# Depth interaction combos
if "depth_x_mom" in opp.columns:
    q80 = opp["depth_x_mom"].quantile(0.80)
    q20 = opp["depth_x_mom"].quantile(0.20)
    q90 = opp["depth_x_mom"].quantile(0.90)
    q10 = opp["depth_x_mom"].quantile(0.10)
    eval_combo("Depth×Mom > Q80 (bid wall + dropped)", opp["depth_x_mom"] > q80, "LONG")
    eval_combo("Depth×Mom < Q20 (ask wall + rallied)", opp["depth_x_mom"] < q20, "SHORT")
    eval_combo("Depth×Mom > Q90 (extreme)", opp["depth_x_mom"] > q90, "LONG")
    eval_combo("Depth×Mom < Q10 (extreme)", opp["depth_x_mom"] < q10, "SHORT")

# Divergence combos
if "div_price_oi" in opp.columns:
    q80 = opp["div_price_oi"].quantile(0.80)
    q20 = opp["div_price_oi"].quantile(0.20)
    eval_combo("Price-OI div > Q80 (price↑ OI↓ = weak rally)", opp["div_price_oi"] > q80, "SHORT")
    eval_combo("Price-OI div < Q20 (price↓ OI↓ = weak dump)", opp["div_price_oi"] < q20, "LONG")

# Exhaustion
if "exhaustion_long" in opp.columns:
    q90 = opp["exhaustion_long"].quantile(0.90)
    eval_combo("Exhaustion long > Q90 (big bull candle at top)", opp["exhaustion_long"] > q90, "SHORT")
if "exhaustion_short" in opp.columns:
    q90 = opp["exhaustion_short"].quantile(0.90)
    eval_combo("Exhaustion short > Q90 (big bear candle at bottom)", opp["exhaustion_short"] > q90, "LONG")

# Hammer/shooting star
if "hammer_score" in opp.columns:
    q90 = opp["hammer_score"].quantile(0.90)
    eval_combo("Hammer score > Q90 (wick rejection at bottom)", opp["hammer_score"] > q90, "LONG")
if "shooting_star_score" in opp.columns:
    q90 = opp["shooting_star_score"].quantile(0.90)
    eval_combo("Shooting star > Q90 (wick rejection at top)", opp["shooting_star_score"] > q90, "SHORT")

# Liq pressure + depth
if "liq_pressure" in opp.columns and "depth_imb_10" in opp.columns:
    lp_q80 = opp["liq_pressure"].quantile(0.80)
    di_q70 = opp["depth_imb_10"].quantile(0.70)
    di_q30 = opp["depth_imb_10"].quantile(0.30)
    eval_combo("High liq pressure + bid wall → LONG bounce",
               (opp["liq_pressure"] > lp_q80) & (opp["depth_imb_10"] > di_q70), "LONG")
    eval_combo("High liq pressure + ask wall → SHORT dump",
               (opp["liq_pressure"] > lp_q80) & (opp["depth_imb_10"] < di_q30), "SHORT")

# Best combo: MR+Depth + liq pressure
if "mr_depth_score" in opp.columns and "liq_pressure" in opp.columns:
    eval_combo("MR+Depth Q80 + high liq pressure → LONG",
               (opp["mr_depth_score"] > opp["mr_depth_score"].quantile(0.80)) &
               (opp["liq_pressure"] > opp["liq_pressure"].quantile(0.70)), "LONG")
    eval_combo("MR+Depth Q20 + high liq pressure → SHORT",
               (opp["mr_depth_score"] < opp["mr_depth_score"].quantile(0.20)) &
               (opp["liq_pressure"] > opp["liq_pressure"].quantile(0.70)), "SHORT")

# ── Temporal stability for best derived ──
print(f"\n{'='*95}")
print("── Temporal Stability: MR+Depth Q90 ──\n")
if "mr_depth_score" in opp.columns:
    opp_ts = opp.copy()
    opp_ts["period"] = opp_ts.index.to_period("Q").astype(str)
    q90_l = opp["mr_depth_score"].quantile(0.90)
    q10_s = opp["mr_depth_score"].quantile(0.10)

    print("  MR+Depth > Q90 → LONG:")
    for period in sorted(opp_ts["period"].unique()):
        p_data = opp_ts[opp_ts["period"] == period]
        sub = p_data[p_data["mr_depth_score"] > q90_l]
        if len(sub) < 10:
            print(f"    {period}: n={len(sub)} (skip)")
            continue
        wr = (sub["fwd_6"] > FEE).mean()
        print(f"    {period}: WR={wr:.3f} n={len(sub)}")

    print("\n  MR+Depth < Q10 → SHORT:")
    for period in sorted(opp_ts["period"].unique()):
        p_data = opp_ts[opp_ts["period"] == period]
        sub = p_data[p_data["mr_depth_score"] < q10_s]
        if len(sub) < 10:
            print(f"    {period}: n={len(sub)} (skip)")
            continue
        wr = (sub["fwd_6"] < -FEE).mean()
        print(f"    {period}: WR={wr:.3f} n={len(sub)}")

print(f"\n{'='*95}")
print("=== DONE ===")
