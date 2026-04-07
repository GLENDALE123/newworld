"""방향 예측력 v3 — 1m 미시구조 + 멀티타임프레임 + volume_bar 추가.

이전 분석에서 빠졌던 데이터:
1. kline_1m: 5m 바 내부 1분봉 패턴 (마지막 1분 방향, 내부 변동성 등)
2. kline_15m / 1h: 상위 TF 추세/구조
3. volume_bar: 거래량 기반 바의 미시구조
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

SYMBOL = "ETHUSDT"
FEE = 0.0008
print(f"=== Direction Predictability v3 ({SYMBOL}) ===\n")

# ── Load all data ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
k1 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1m.parquet").set_index("timestamp").sort_index()
k15 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_15m.parquet").set_index("timestamp").sort_index()
k1h = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1h.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
vbar = pd.read_parquet(f"data/merged/{SYMBOL}/volume_bar.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()

# TZ align
for d in [tick, k1, k15, k1h, vbar]:
    if d.index.tz is None and k5.index.tz is not None:
        d.index = d.index.tz_localize(k5.index.tz)

print(f"5m: {len(k5):,}, 1m: {len(k1):,}, 15m: {len(k15):,}, 1h: {len(k1h):,}")
print(f"tick_bar: {len(tick):,}, volume_bar: {len(vbar):,}")

# ── Forward return (target) ──
c5 = k5["close"]
k5["fwd_6"] = c5.shift(-6) / c5 - 1

# ── 1. 기존 피처 (빠르게) ──
feat = pd.DataFrame(index=k5.index)
ret = c5.pct_change()
feat["ret_3"] = c5 / c5.shift(3) - 1
feat["vwap_dist"] = c5 / ((c5 * k5["volume"]).rolling(20).sum() / (k5["volume"].rolling(20).sum() + 1e-10)) - 1
feat["range_pos_20"] = (c5 - k5["low"].rolling(20).min()) / (k5["high"].rolling(20).max() - k5["low"].rolling(20).min() + 1e-10)
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"] / feat["vol_60"]
feat["vol_ratio"] = k5["volume"] / k5["volume"].rolling(12).mean()

# CVD
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
if tick.index.tz is None and k5.index.tz is not None:
    tick.index = tick.index.tz_localize(k5.index.tz)
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum"})
feat["cvd_slope_5"] = tick_5m["cvd_raw"].cumsum().diff(5).reindex(k5.index)

# Book depth
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)
bd_key = bd[bd["percentage"].isin([-1.0, 1.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_piv = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
if -1.0 in bd_piv.columns and 1.0 in bd_piv.columns:
    feat["depth_imb_10"] = ((bd_piv[-1.0] - bd_piv[1.0]) / (bd_piv[-1.0] + bd_piv[1.0] + 1e-10)).reindex(k5.index)

# ── 2. NEW: 1m 미시구조 피처 ──
print("\n[1m microstructure features...]")

# 5m 바 내부의 1분봉 통계
k1_ret = k1["close"].pct_change()
k1_body = (k1["close"] - k1["open"]) / (k1["high"] - k1["low"] + 1e-10)

# 5분 단위로 리샘플: 5개 1분봉의 통계
k1_5m = pd.DataFrame(index=k5.index)

# 마지막 1분 방향 (5m 바의 마지막 1분이 양봉인지)
k1_5m["last_1m_ret"] = k1_ret.resample("5min").last()

# 5m 내부 1분봉 방향 일관성 (5개 중 몇 개가 양봉)
k1_5m["bull_1m_count"] = (k1_ret > 0).astype(int).resample("5min").sum()

# 5m 내부 최대 1분 수익 vs 최소 1분 수익 (내부 변동성)
k1_5m["max_1m_ret"] = k1_ret.resample("5min").max()
k1_5m["min_1m_ret"] = k1_ret.resample("5min").min()
k1_5m["intra_5m_range"] = k1_5m["max_1m_ret"] - k1_5m["min_1m_ret"]

# 첫 1분 vs 마지막 1분 방향 (반전 패턴)
k1_5m["first_1m_ret"] = k1_ret.resample("5min").first()
k1_5m["first_last_div"] = np.sign(k1_5m["first_1m_ret"]) != np.sign(k1_5m["last_1m_ret"])
k1_5m["first_last_div"] = k1_5m["first_last_div"].astype(float)

# 마지막 2분의 방향 (close 직전 모멘텀)
k1_close = k1["close"].resample("5min")
# 5m 바의 4번째와 5번째 1분봉
k1_5m["tail_2m_ret"] = k1["close"].resample("5min").apply(
    lambda x: (x.iloc[-1] / x.iloc[-3] - 1) if len(x) >= 3 else np.nan
)

# 1분봉 거래량 집중도: 마지막 1분 거래량 / 전체 5분 거래량
k1_5m["last_1m_vol_pct"] = k1["volume"].resample("5min").apply(
    lambda x: x.iloc[-1] / (x.sum() + 1e-10) if len(x) > 0 else np.nan
)

# 1분봉 body 합산 vs 5분봉 body (내부 추세 일관성)
k1_body_sum = k1_body.resample("5min").sum()
k5_body = (k5["close"] - k5["open"]) / (k5["high"] - k5["low"] + 1e-10)
k1_5m["body_consistency"] = k1_body_sum / (abs(k5_body) * 5 + 1e-10)

feat = feat.join(k1_5m, how="left")
print(f"  Added {len(k1_5m.columns)} 1m features")

# ── 3. NEW: 멀티타임프레임 피처 ──
print("[Multi-timeframe features...]")

# 15m — shift(1)로 완성된 봉만 사용 (look-ahead 방지)
k15_ret = k15["close"].pct_change()
k15_ma8 = k15["close"].rolling(8).mean()
k15_ma20 = k15["close"].rolling(20).mean()
feat["tf15_ret_1"] = k15_ret.shift(1).reindex(k5.index, method="ffill")  # 직전 완성 15m 봉
feat["tf15_ret_4"] = (k15["close"].shift(1) / k15["close"].shift(5) - 1).reindex(k5.index, method="ffill")
feat["tf15_ma_slope"] = ((k15_ma8.shift(1) - k15_ma8.shift(3)) / (k15_ma8.shift(3) + 1e-10)).reindex(k5.index, method="ffill")
feat["tf15_trend"] = ((k15_ma8.shift(1) / k15_ma20.shift(1) - 1)).reindex(k5.index, method="ffill")
feat["tf15_body"] = ((k15["close"].shift(1) - k15["open"].shift(1)) / (k15["high"].shift(1) - k15["low"].shift(1) + 1e-10)).reindex(k5.index, method="ffill")
feat["tf15_range_pos"] = ((k15["close"].shift(1) - k15["low"].shift(1).rolling(20).min()) / (k15["high"].shift(1).rolling(20).max() - k15["low"].shift(1).rolling(20).min() + 1e-10)).reindex(k5.index, method="ffill")

# 1h — shift(1)로 완성된 봉만 사용
k1h_ret = k1h["close"].pct_change()
k1h_ma8 = k1h["close"].rolling(8).mean()
k1h_ma24 = k1h["close"].rolling(24).mean()
feat["tf1h_ret_1"] = k1h_ret.shift(1).reindex(k5.index, method="ffill")  # 직전 완성 1h 봉
feat["tf1h_ret_4"] = (k1h["close"].shift(1) / k1h["close"].shift(5) - 1).reindex(k5.index, method="ffill")
feat["tf1h_ma_slope"] = ((k1h_ma8.shift(1) - k1h_ma8.shift(3)) / (k1h_ma8.shift(3) + 1e-10)).reindex(k5.index, method="ffill")
feat["tf1h_trend"] = ((k1h_ma8.shift(1) / k1h_ma24.shift(1) - 1)).reindex(k5.index, method="ffill")
feat["tf1h_body"] = ((k1h["close"].shift(1) - k1h["open"].shift(1)) / (k1h["high"].shift(1) - k1h["low"].shift(1) + 1e-10)).reindex(k5.index, method="ffill")

# Cross-TF: 5m vs 15m vs 1h 방향 일치도
feat["xtf_align_5_15"] = np.sign(feat["ret_3"]) * np.sign(feat["tf15_ret_4"])
feat["xtf_align_5_1h"] = np.sign(feat["ret_3"]) * np.sign(feat["tf1h_ret_4"])
feat["xtf_align_15_1h"] = np.sign(feat["tf15_ret_4"]) * np.sign(feat["tf1h_ret_4"])
feat["xtf_all_align"] = (feat["xtf_align_5_15"] + feat["xtf_align_5_1h"] + feat["xtf_align_15_1h"]) / 3

# 5m이 상위 TF 추세 반대 = mean-reversion 기회
feat["mr_vs_15m"] = -feat["ret_3"] * feat["tf15_trend"]  # 5m 하락 + 15m 상승추세 = long 기회
feat["mr_vs_1h"] = -feat["ret_3"] * feat["tf1h_trend"]

print(f"  Added MTF features")

# ── 4. NEW: Volume bar 피처 ──
print("[Volume bar features...]")
vbar_5m = vbar.resample("5min").agg({
    "close": "last", "high": "max", "low": "min",
    "volume": "sum", "trade_count": "sum"
})
# 5분 내 volume bar 개수 = 거래 활발도
vbar_count = vbar.resample("5min").size()
feat["vbar_count"] = vbar_count.reindex(k5.index)
feat["vbar_count_ratio"] = feat["vbar_count"] / feat["vbar_count"].rolling(12).mean()

if "buy_volume" in vbar.columns:
    vbar_buy = vbar["buy_volume"].resample("5min").sum()
    vbar_sell = vbar["sell_volume"].resample("5min").sum() if "sell_volume" in vbar.columns else vbar["volume"].resample("5min").sum() - vbar_buy
    feat["vbar_buy_ratio"] = (vbar_buy / (vbar_buy + vbar_sell + 1e-10)).reindex(k5.index)

print(f"  Added volume bar features")

# ── 5. Metrics ──
for col in ["sum_open_interest_value", "sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(k5.index, method="ffill")
if "sum_open_interest_value" in feat.columns:
    feat["oi_change_5"] = feat["sum_open_interest_value"].pct_change(5)
    feat.drop(columns=["sum_open_interest_value"], inplace=True)
feat["funding_rate"] = funding["funding_rate"].reindex(k5.index, method="ffill") if "funding_rate" in funding.columns else np.nan

feat = feat.replace([np.inf, -np.inf], np.nan)
print(f"\nTotal features: {feat.shape[1]}")

# ── Analysis: 2023+ opportunity zone ──
df = k5.loc["2023-01-01":].iloc[60:].copy()
df = df.dropna(subset=["fwd_6"])
feat_df = feat.reindex(df.index)

opp = df[(feat_df["vol_squeeze"] > 1.0) & (feat_df["vol_ratio"] > 1.5)].index
feat_opp = feat_df.loc[opp]
target = df.loc[opp, "fwd_6"]

print(f"\n2023+ bars: {len(df):,}, Opportunity zone: {len(opp):,} ({len(opp)/len(df)*100:.1f}%)")

# ── Feature-by-feature IC ──
results = []
print(f"\n{'='*95}")
print(f"{'Feature':<35s} {'IC':>7s} {'Δ_WR':>7s} {'n':>8s} {'Type':>6s}")
print(f"{'-'*95}")

new_features = [c for c in feat.columns if c.startswith(("last_1m", "bull_1m", "max_1m", "min_1m", "intra_5m",
    "first_", "tail_2m", "last_1m_vol", "body_consist", "tf15", "tf1h", "xtf_", "mr_vs_",
    "vbar_"))]

for feat_col in feat.columns:
    if feat_col not in feat_opp.columns: continue
    x = feat_opp[feat_col]
    y = target
    valid = ~(x.isna() | y.isna() | np.isinf(x))
    if valid.sum() < 300: continue

    xv = x[valid].values.astype(np.float64)
    yv = y[valid].values.astype(np.float64)
    ic, _ = spearmanr(xv, yv)
    med = np.median(xv)
    hi = xv >= med; lo = xv < med
    wr_hi = (yv[hi] > FEE).mean()
    wr_lo = (yv[lo] > FEE).mean()
    delta = wr_hi - wr_lo

    is_new = feat_col in new_features
    results.append({"feature": feat_col, "ic": ic, "delta_wr": delta, "n": int(valid.sum()), "new": is_new})
    tag = "NEW" if is_new else "old"
    print(f"  {feat_col:<33s} {ic:+.4f} {delta:+.4f} {int(valid.sum()):>8d} {tag:>6s}")

results.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"\n{'='*95}")
print("── Ranked by |IC| (NEW features highlighted) ──\n")
for i, r in enumerate(results[:30]):
    tag = " ★" if r["new"] and abs(r["ic"]) > 0.05 else " ●" if r["new"] else ""
    print(f"  {i+1:2d}. {r['feature']:<33s} IC={r['ic']:+.4f} Δ_WR={r['delta_wr']:+.4f}{tag}")

# ── Cross-TF conditional analysis ──
print(f"\n{'='*95}")
print("── Cross-TF Conditional: 상위TF 추세 반대로 mean-revert ──\n")

for tf_col, tf_name in [("tf15_trend", "15m trend"), ("tf1h_trend", "1h trend")]:
    if tf_col not in feat_opp.columns: continue
    v = feat_opp[tf_col]
    fwd = target

    # 상위TF 상승 + 5m 하락 → long (mean-revert to higher TF trend)
    up_trend = v > v.quantile(0.7)
    down_5m = feat_opp["ret_3"] < feat_opp["ret_3"].quantile(0.3)
    mask_long = up_trend & down_5m
    sub = fwd[mask_long].dropna()
    if len(sub) > 50:
        wr = (sub > FEE).mean()
        avg = sub.mean() * 100
        print(f"  {tf_name} UP + 5m dipped → LONG: WR={wr:.3f} avg={avg:+.4f}% n={len(sub)}")

    # 상위TF 하락 + 5m 상승 → short
    dn_trend = v < v.quantile(0.3)
    up_5m = feat_opp["ret_3"] > feat_opp["ret_3"].quantile(0.7)
    mask_short = dn_trend & up_5m
    sub = fwd[mask_short].dropna()
    if len(sub) > 50:
        wr = (sub < -FEE).mean()
        avg = -sub.mean() * 100
        print(f"  {tf_name} DOWN + 5m rallied → SHORT: WR={wr:.3f} avg={avg:+.4f}% n={len(sub)}")

# ── 1m microstructure conditional ──
print(f"\n{'='*95}")
print("── 1m Microstructure Conditional ──\n")

if "tail_2m_ret" in feat_opp.columns:
    tail = feat_opp["tail_2m_ret"]
    # 마지막 2분 급등 후 → 반전 short?
    q90 = tail.quantile(0.90)
    q10 = tail.quantile(0.10)
    sub_up = target[tail > q90].dropna()
    sub_dn = target[tail < q10].dropna()
    if len(sub_up) > 50:
        print(f"  Tail 2m surge (>Q90) → next: WR_short={(sub_up < -FEE).mean():.3f} avg={-sub_up.mean()*100:+.4f}% n={len(sub_up)}")
    if len(sub_dn) > 50:
        print(f"  Tail 2m dump (<Q10) → next: WR_long={(sub_dn > FEE).mean():.3f} avg={sub_dn.mean()*100:+.4f}% n={len(sub_dn)}")

if "last_1m_vol_pct" in feat_opp.columns:
    lv = feat_opp["last_1m_vol_pct"]
    q80 = lv.quantile(0.80)
    # 마지막 1분에 거래량 집중 + 방향 = 의미 있는 마감
    high_lv = lv > q80
    up_close = feat_opp["last_1m_ret"] > 0
    dn_close = feat_opp["last_1m_ret"] < 0
    sub = target[high_lv & up_close].dropna()
    if len(sub) > 50:
        print(f"  High last-1m vol + up close → next long: WR={(sub > FEE).mean():.3f} n={len(sub)}")
    sub = target[high_lv & dn_close].dropna()
    if len(sub) > 50:
        print(f"  High last-1m vol + down close → next short: WR={(sub < -FEE).mean():.3f} n={len(sub)}")

# ── Best combo with new features ──
print(f"\n{'='*95}")
print("── Best Combos with New Features ──\n")

def eval_c(label, mask, direction):
    sub = target[mask].dropna()
    if len(sub) < 50:
        print(f"  {label:<65s} SKIP (n={len(sub)})")
        return
    if direction == "LONG":
        wr = (sub > FEE).mean(); avg = sub.mean() * 100
    else:
        wr = (sub < -FEE).mean(); avg = -sub.mean() * 100
    m = " <<<" if wr > 0.55 else " **" if wr > 0.52 else ""
    print(f"  {label:<65s} {direction} WR={wr:.3f} avg={avg:+.4f}% n={len(sub)}{m}")

if "tf1h_trend" in feat_opp.columns and "depth_imb_10" in feat_opp.columns:
    di_q70 = feat_opp["depth_imb_10"].quantile(0.70)
    di_q30 = feat_opp["depth_imb_10"].quantile(0.30)

    # 1h uptrend + 5m dipped + bid wall
    eval_c("1h UP + 5m dip + bid wall → LONG",
           (feat_opp["tf1h_trend"] > feat_opp["tf1h_trend"].quantile(0.7)) &
           (feat_opp["ret_3"] < feat_opp["ret_3"].quantile(0.3)) &
           (feat_opp["depth_imb_10"] > di_q70), "LONG")

    eval_c("1h DOWN + 5m rally + ask wall → SHORT",
           (feat_opp["tf1h_trend"] < feat_opp["tf1h_trend"].quantile(0.3)) &
           (feat_opp["ret_3"] > feat_opp["ret_3"].quantile(0.7)) &
           (feat_opp["depth_imb_10"] < di_q30), "SHORT")

if "tail_2m_ret" in feat_opp.columns and "tf15_trend" in feat_opp.columns:
    eval_c("15m UP + tail dump + bid wall → LONG",
           (feat_opp["tf15_trend"] > feat_opp["tf15_trend"].quantile(0.7)) &
           (feat_opp["tail_2m_ret"] < feat_opp["tail_2m_ret"].quantile(0.15)) &
           (feat_opp["depth_imb_10"] > di_q70), "LONG")

    eval_c("15m DOWN + tail surge + ask wall → SHORT",
           (feat_opp["tf15_trend"] < feat_opp["tf15_trend"].quantile(0.3)) &
           (feat_opp["tail_2m_ret"] > feat_opp["tail_2m_ret"].quantile(0.85)) &
           (feat_opp["depth_imb_10"] < di_q30), "SHORT")

if "mr_vs_1h" in feat_opp.columns:
    q90 = feat_opp["mr_vs_1h"].quantile(0.90)
    q10 = feat_opp["mr_vs_1h"].quantile(0.10)
    eval_c("MR vs 1h > Q90 (strong mean-revert long setup)", feat_opp["mr_vs_1h"] > q90, "LONG")
    eval_c("MR vs 1h < Q10 (strong mean-revert short setup)", feat_opp["mr_vs_1h"] < q10, "SHORT")

print(f"\n{'='*95}")
print("=== DONE ===")
