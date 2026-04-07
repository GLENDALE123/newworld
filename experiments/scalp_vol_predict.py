"""스캘핑 접근 전환: 방향 예측 → 변동성 예측.

핵심 가설: "방향은 못 맞추지만, 큰 움직임이 올 순간은 맞출 수 있다"
→ 큰 움직임 예측 + 단순 방향 룰(depth/mean-reversion) = 수익

Target: forward 6bar absolute return > threshold (binary)
방향: depth_imbalance sign으로 결정 (유일한 순방향 시그널)

Fee 시나리오:
A. Taker: 0.08% RT (현재)
B. Maker: 0.02% RT (지정가 진입)
C. Maker+rebate: 0.01% RT (상위 VIP)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SYMBOL = "ETHUSDT"
print(f"=== Volatility Prediction Approach ({SYMBOL}) ===\n")

# ── Data ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
k1 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1m.parquet").set_index("timestamp").sort_index()
k15 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_15m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()

for d in [tick, k1, k15]:
    if d.index.tz is None and k5.index.tz is not None:
        d.index = d.index.tz_localize(k5.index.tz)

c5 = k5["close"]; o5 = k5["open"]; h5 = k5["high"]; l5 = k5["low"]; v5 = k5["volume"]

# ── Targets: forward absolute move ──
# MFE in 6 bars (max of high - entry, entry - low)
fwd_abs = np.full(len(k5), np.nan)
fwd_dir_at_mfe = np.full(len(k5), np.nan)  # +1 if MFE was upward, -1 if downward
closes = c5.values; highs = h5.values; lows = l5.values

for i in range(len(closes) - 6):
    entry = closes[i]
    max_h = max(highs[i+1:i+7])
    min_l = min(lows[i+1:i+7])
    up_move = (max_h - entry) / entry
    dn_move = (entry - min_l) / entry
    fwd_abs[i] = max(up_move, dn_move)
    fwd_dir_at_mfe[i] = 1.0 if up_move >= dn_move else -1.0

k5["fwd_abs"] = fwd_abs
k5["fwd_dir"] = fwd_dir_at_mfe
# Also fixed 6-bar return for realistic evaluation
k5["fwd_6_ret"] = c5.shift(-6) / c5 - 1

print(f"fwd_abs distribution:")
fa = pd.Series(fwd_abs).dropna()
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {fa.quantile(p/100)*100:.3f}%")

# ── Features (모든 가용 데이터) ──
feat = pd.DataFrame(index=k5.index)
ret = c5.pct_change()

# Volatility features (이게 핵심)
feat["vol_5"] = ret.rolling(5).std()
feat["vol_12"] = ret.rolling(12).std()
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"] / feat["vol_60"]
feat["vol_accel"] = feat["vol_5"] / feat["vol_20"]  # 단기 vol 가속
feat["atr_12"] = (h5 - l5).rolling(12).mean()
feat["atr_ratio"] = (h5 - l5) / feat["atr_12"]
feat["bar_range_pct"] = (h5 - l5) / c5  # current bar range

# Volume (vol 터질 때 거래량도 터짐)
feat["vol_ratio"] = v5 / v5.rolling(12).mean()
feat["vol_ratio_3"] = v5.rolling(3).mean() / v5.rolling(12).mean()  # 최근 3bar 거래량 급증

# Tick bar
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"})
feat["trade_count"] = tick_5m["trade_count"].reindex(k5.index)
feat["tc_ratio"] = feat["trade_count"] / feat["trade_count"].rolling(12).mean()
feat["buy_ratio"] = (tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"])).reindex(k5.index)
feat["cvd_abs_5"] = tick_5m["cvd_raw"].abs().rolling(5).mean().reindex(k5.index)

# 1m microstructure
k1_ret = k1["close"].pct_change()
feat["intra_5m_range"] = (k1_ret.resample("5min").max() - k1_ret.resample("5min").min()).reindex(k5.index)
feat["bull_1m_count"] = (k1_ret > 0).astype(int).resample("5min").sum().reindex(k5.index)
feat["last_1m_abs"] = k1_ret.abs().resample("5min").last().reindex(k5.index)

# 15m (shift 1 — completed candle only)
k15_range = ((k15["high"] - k15["low"]) / k15["close"]).shift(1)
k15_vol = k15["close"].pct_change().rolling(4).std().shift(1)
feat["tf15_range"] = k15_range.reindex(k5.index, method="ffill")
feat["tf15_vol"] = k15_vol.reindex(k5.index, method="ffill")

# Book depth
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)
bd_key = bd[bd["percentage"].isin([-1.0, 1.0, -2.0, 2.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_piv = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
if -1.0 in bd_piv.columns and 1.0 in bd_piv.columns:
    feat["depth_imb_10"] = ((bd_piv[-1.0] - bd_piv[1.0]) / (bd_piv[-1.0] + bd_piv[1.0] + 1e-10)).reindex(k5.index)
    feat["depth_total_10"] = (bd_piv[-1.0] + bd_piv[1.0]).reindex(k5.index)
    # Depth thinning = vol 터질 징조
    feat["depth_total_10_ratio"] = feat["depth_total_10"] / feat["depth_total_10"].rolling(12).mean()

# Metrics
for col in ["sum_open_interest_value", "sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(k5.index, method="ffill")
if "sum_open_interest_value" in feat.columns:
    feat["oi_change_5"] = feat["sum_open_interest_value"].pct_change(5)
    feat["oi_abs_change"] = feat["sum_open_interest_value"].pct_change(5).abs()
    feat.drop(columns=["sum_open_interest_value"], inplace=True)
feat["funding_rate"] = funding["funding_rate"].reindex(k5.index, method="ffill") if "funding_rate" in funding.columns else np.nan
feat["funding_abs"] = feat["funding_rate"].abs()

# Price structure (mean-reversion 방향 결정용)
feat["ret_3"] = c5 / c5.shift(3) - 1
feat["vwap_dist"] = c5 / ((c5 * v5).rolling(20).sum() / (v5.rolling(20).sum() + 1e-10)) - 1
feat["range_pos_20"] = (c5 - l5.rolling(20).min()) / (h5.rolling(20).max() - l5.rolling(20).min() + 1e-10)

# Time
feat["hour"] = k5.index.hour
feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)

feat = feat.replace([np.inf, -np.inf], np.nan)
feat.drop(columns=["depth_total_10"], inplace=True, errors="ignore")
print(f"\nFeatures: {feat.shape[1]}")

# ── Analysis: 2023+ ──
df = k5.loc["2023-01-01":].iloc[60:].dropna(subset=["fwd_abs"]).copy()
feat_df = feat.reindex(df.index)

print(f"Analysis bars: {len(df):,}")

# ── 1. Volatility prediction: IC with fwd_abs ──
print(f"\n{'='*80}")
print("── Feature IC with Forward Absolute Move (fwd_abs) ──\n")

target_abs = df["fwd_abs"]
results = []

for col in feat.columns:
    x = feat_df[col]
    valid = ~(x.isna() | target_abs.isna() | np.isinf(x))
    if valid.sum() < 300: continue
    xv = x[valid].values.astype(np.float64)
    yv = target_abs[valid].values.astype(np.float64)
    ic, _ = spearmanr(xv, yv)
    results.append({"feature": col, "ic": ic, "n": int(valid.sum())})

results.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"  {'Feature':<30s} {'IC_abs':>8s}")
for r in results[:25]:
    bar = "█" * min(int(abs(r["ic"]) * 50), 30)
    print(f"  {r['feature']:<30s} {r['ic']:+.4f} {bar}")

# ── 2. 큰 움직임 예측 → 방향은 depth로 ──
print(f"\n{'='*80}")
print("── Strategy: Predict Big Move + Direction from Depth ──\n")

# Thresholds for "big move"
for fee_label, fee in [("Taker 0.08%", 0.0008), ("Maker 0.02%", 0.0002), ("VIP 0.01%", 0.0001)]:
    print(f"\n  === Fee scenario: {fee_label} ===")

    # "Big move" = fwd_abs > 2x fee (to have room for profit)
    big_threshold = fee * 2.5
    is_big = target_abs > big_threshold
    big_rate = is_big.mean()
    print(f"  Big move threshold: {big_threshold*100:.3f}%, rate: {big_rate:.1%}")

    # Base rate: random entry with depth direction
    di = feat_df["depth_imb_10"]
    fwd_ret = df["fwd_6_ret"]
    has_depth = di.notna()

    # If depth > 0 → long, else → short
    direction = np.where(di > 0, 1, -1)
    pnl_with_depth = np.where(direction == 1, fwd_ret, -fwd_ret) - fee
    valid_depth = has_depth & fwd_ret.notna()

    pnl_all = pnl_with_depth[valid_depth]
    wr_all = (pnl_all > 0).mean()
    avg_all = np.nanmean(pnl_all)
    print(f"  All bars + depth direction: WR={wr_all:.3f} avg_pnl={avg_all*100:+.4f}% n={valid_depth.sum():,}")

    # Only big moves + depth direction
    big_and_depth = is_big & valid_depth
    pnl_big = pnl_with_depth[big_and_depth]
    if len(pnl_big) > 100:
        wr_big = (pnl_big > 0).mean()
        avg_big = np.nanmean(pnl_big)
        print(f"  Big moves only + depth dir: WR={wr_big:.3f} avg_pnl={avg_big*100:+.4f}% n={big_and_depth.sum():,}")

    # Top vol features as filter
    for vol_feat, vol_name in [("vol_accel", "Vol acceleration"), ("tc_ratio", "Trade count ratio"),
                                ("atr_ratio", "ATR ratio"), ("vol_ratio", "Volume ratio"),
                                ("depth_total_10_ratio", "Depth thinning")]:
        if vol_feat not in feat_df.columns: continue
        vf = feat_df[vol_feat]
        valid_vf = valid_depth & vf.notna()
        if valid_vf.sum() < 500: continue

        for q_label, q in [("Q70", 0.70), ("Q80", 0.80), ("Q90", 0.90)]:
            threshold = vf.quantile(q)
            if vol_feat == "depth_total_10_ratio":
                # Depth thinning: LOW ratio = less depth = more likely to move
                mask = valid_vf & (vf < vf.quantile(1 - q))
            else:
                mask = valid_vf & (vf > threshold)

            sub_pnl = pnl_with_depth[mask]
            if len(sub_pnl) < 50: continue
            wr = (sub_pnl > 0).mean()
            avg = np.nanmean(sub_pnl)
            selectivity = mask.sum() / valid_vf.sum()
            marker = " <<<" if wr > 0.52 else ""
            if q == 0.80:  # Only print Q80 to keep output manageable
                print(f"    {vol_name} {q_label}: WR={wr:.3f} avg={avg*100:+.4f}% sel={selectivity:.2f} n={mask.sum():,}{marker}")

    # Combination: vol_accel + tc_ratio + depth direction
    if "vol_accel" in feat_df.columns and "tc_ratio" in feat_df.columns:
        va = feat_df["vol_accel"]; tc = feat_df["tc_ratio"]
        for q in [0.70, 0.80, 0.90]:
            combo_mask = valid_depth & (va > va.quantile(q)) & (tc > tc.quantile(q))
            sub = pnl_with_depth[combo_mask]
            if len(sub) < 50: continue
            wr = (sub > 0).mean()
            avg = np.nanmean(sub)
            sel = combo_mask.sum() / valid_depth.sum()
            marker = " <<<" if wr > 0.52 else ""
            print(f"    Vol_accel+TC_ratio Q{int(q*100)} + depth: WR={wr:.3f} avg={avg*100:+.4f}% sel={sel:.2f} n={combo_mask.sum():,}{marker}")

    # Triple: vol_accel + tc_ratio + atr_ratio + depth
    if "vol_accel" in feat_df.columns and "atr_ratio" in feat_df.columns:
        ar = feat_df["atr_ratio"]
        for q in [0.70, 0.80]:
            combo3 = valid_depth & (va > va.quantile(q)) & (tc > tc.quantile(q)) & (ar > ar.quantile(q))
            sub = pnl_with_depth[combo3]
            if len(sub) < 50: continue
            wr = (sub > 0).mean()
            avg = np.nanmean(sub)
            sel = combo3.sum() / valid_depth.sum()
            marker = " <<<" if wr > 0.52 else ""
            print(f"    Vol+TC+ATR Q{int(q*100)} + depth: WR={wr:.3f} avg={avg*100:+.4f}% sel={sel:.2f} n={combo3.sum():,}{marker}")

# ── 3. Mean-reversion 방향 (depth 대신) ──
print(f"\n{'='*80}")
print("── Alternative Direction: Mean-Reversion Rule ──\n")

# 방향 = -sign(ret_3): 올랐으면 short, 내렸으면 long
mr_dir = -np.sign(feat_df["ret_3"].values)
fwd_ret = df["fwd_6_ret"].values
valid_mr = ~np.isnan(fwd_ret) & ~np.isnan(mr_dir)

for fee_label, fee in [("Taker 0.08%", 0.0008), ("Maker 0.02%", 0.0002)]:
    pnl_mr = np.where(mr_dir == 1, fwd_ret, -fwd_ret) - fee

    # All bars
    sub = pnl_mr[valid_mr]
    print(f"\n  [{fee_label}] All bars MR direction: WR={(sub>0).mean():.3f} avg={np.nanmean(sub)*100:+.4f}% n={valid_mr.sum():,}")

    # With vol filter
    if "vol_accel" in feat_df.columns and "tc_ratio" in feat_df.columns:
        va = feat_df["vol_accel"]; tc = feat_df["tc_ratio"]
        for q in [0.70, 0.80, 0.90]:
            mask = valid_mr & (va > va.quantile(q)).values & (tc > tc.quantile(q)).values
            sub = pnl_mr[mask]
            if len(sub) < 50: continue
            wr = (sub > 0).mean()
            avg = np.nanmean(sub)
            sel = mask.sum() / valid_mr.sum()
            marker = " <<<" if wr > 0.52 else ""
            print(f"    Vol_accel+TC Q{int(q*100)} + MR dir: WR={wr:.3f} avg={avg*100:+.4f}% sel={sel:.2f} n={mask.sum():,}{marker}")

# ── 4. Combined: depth + MR 앙상블 ──
print(f"\n{'='*80}")
print("── Combined Direction: Depth + MR Agreement ──\n")

di_dir = np.where(feat_df["depth_imb_10"] > 0, 1, -1)
mr_dir_arr = -np.sign(feat_df["ret_3"].values)
agree = (di_dir == mr_dir_arr)  # depth와 MR이 같은 방향

for fee_label, fee in [("Taker 0.08%", 0.0008), ("Maker 0.02%", 0.0002)]:
    # When both agree, use that direction
    combined_dir = np.where(agree, di_dir, 0)  # disagree = HOLD
    pnl_comb = np.where(combined_dir == 1, fwd_ret, np.where(combined_dir == -1, -fwd_ret, 0)) - fee * (combined_dir != 0)

    trade_mask = (combined_dir != 0) & valid_mr & feat_df["depth_imb_10"].notna().values
    sub = pnl_comb[trade_mask]
    if len(sub) > 100:
        wr = (sub > 0).mean()
        avg = np.nanmean(sub)
        sel = trade_mask.sum() / valid_mr.sum()
        print(f"  [{fee_label}] Depth+MR agree: WR={wr:.3f} avg={avg*100:+.4f}% sel={sel:.2f} n={trade_mask.sum():,}")

    # + vol filter
    if "vol_accel" in feat_df.columns:
        va = feat_df["vol_accel"]
        for q in [0.70, 0.80]:
            mask = trade_mask & (va > va.quantile(q)).values
            sub = pnl_comb[mask]
            if len(sub) < 50: continue
            wr = (sub > 0).mean()
            avg = np.nanmean(sub)
            marker = " <<<" if wr > 0.52 else ""
            print(f"    + Vol_accel Q{int(q*100)}: WR={wr:.3f} avg={avg*100:+.4f}% sel={mask.sum()/valid_mr.sum():.2f} n={mask.sum():,}{marker}")

print(f"\n{'='*80}")
print("=== DONE ===")
