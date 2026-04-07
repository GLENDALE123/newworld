"""BTC 5m 스캘핑 방향 예측력 v2 — book_depth + metrics + 전체 피처."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

print("=== Direction Predictability v2 (all data) ===\n")

# ── 1. 데이터 로딩 + 피처 (v2 explore와 동일) ──
kline = pd.read_parquet("data/merged/ETHUSDT/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet("data/merged/ETHUSDT/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet("data/merged/ETHUSDT/book_depth.parquet")
metrics = pd.read_parquet("data/merged/ETHUSDT/metrics.parquet").set_index("timestamp").sort_index()
bt = pd.read_parquet("data/merged/ETHUSDT/book_ticker.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet("data/merged/ETHUSDT/funding_rate.parquet").set_index("timestamp").sort_index()

# TZ
for d in [tick, bt]:
    if d.index.tz is None and kline.index.tz is not None:
        d.index = d.index.tz_localize(kline.index.tz)

c = kline["close"]; o = kline["open"]; h = kline["high"]; l = kline["low"]; v = kline["volume"]

# Forward returns
kline["fwd_6"] = c.shift(-6) / c - 1
for n in [3, 4, 5]:
    kline[f"fwd_{n}"] = c.shift(-n) / c - 1
kline["fwd_best_long"] = kline[["fwd_3", "fwd_4", "fwd_5", "fwd_6"]].max(axis=1)
kline["fwd_best_short"] = -kline[["fwd_3", "fwd_4", "fwd_5", "fwd_6"]].min(axis=1)
FEE = 0.0008

# ── Basic features ──
kline["ret"] = c.pct_change()
kline["vol_20"] = kline["ret"].rolling(20).std()
kline["vol_60"] = kline["ret"].rolling(60).std()
kline["vol_squeeze"] = kline["vol_20"] / kline["vol_60"]
kline["atr_12"] = (h - l).rolling(12).mean()
kline["atr_ratio"] = (h - l) / kline["atr_12"]
kline["vol_ma_12"] = v.rolling(12).mean()
kline["vol_ratio"] = v / kline["vol_ma_12"]

# Tick bar
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"})
kline = kline.drop(columns=["trade_count"], errors="ignore")
kline = kline.join(tick_5m[["buy_volume", "sell_volume", "cvd_raw", "trade_count"]], how="left")
kline["buy_ratio"] = kline["buy_volume"] / (kline["buy_volume"] + kline["sell_volume"])

# ── Direction features (from v1) ──
kline["cvd_cumsum"] = kline["cvd_raw"].cumsum()
kline["cvd_slope_3"] = kline["cvd_cumsum"] - kline["cvd_cumsum"].shift(3)
kline["cvd_slope_5"] = kline["cvd_cumsum"] - kline["cvd_cumsum"].shift(5)
kline["buy_ratio_ma5"] = kline["buy_ratio"].rolling(5).mean()
kline["buy_ratio_ma12"] = kline["buy_ratio"].rolling(12).mean()
kline["buy_ratio_trend"] = kline["buy_ratio_ma5"] - kline["buy_ratio_ma12"]
kline["mom_3"] = c / c.shift(3) - 1
kline["ma_20"] = c.rolling(20).mean()
kline["ma_50"] = c.rolling(50).mean()
kline["ma_20_slope"] = (kline["ma_20"] - kline["ma_20"].shift(3)) / kline["ma_20"].shift(3)
kline["ma_50_slope"] = (kline["ma_50"] - kline["ma_50"].shift(5)) / kline["ma_50"].shift(5)
kline["ma_20_50_cross"] = kline["ma_20"] / kline["ma_50"] - 1
kline["taker_buy_pct"] = kline["taker_buy_volume"] / kline["volume"]
kline["taker_buy_pct_ma5"] = kline["taker_buy_pct"].rolling(5).mean()
kline["body_ratio"] = (c - o) / (h - l + 1e-10)
kline["body_ratio_3avg"] = kline["body_ratio"].rolling(3).mean()
kline["obv"] = (np.sign(kline["ret"]) * v).cumsum()
kline["obv_slope_5"] = kline["obv"] - kline["obv"].shift(5)
kline["obv_slope_norm"] = kline["obv_slope_5"] / (v.rolling(5).sum() + 1e-10)
kline["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)
kline["vwap_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()
kline["vwap_dist"] = c / kline["vwap_20"] - 1

# ── NEW: Book Depth ──
print("[book_depth...]")
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and kline.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(kline.index.tz)
key_pcts = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]
bd_key = bd[bd["percentage"].isin(key_pcts)].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_pivot = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
bd_pivot.columns = [f"depth_{c}" for c in bd_pivot.columns]

for neg, pos, tag in [(-1.0, 1.0, "10"), (-2.0, 2.0, "20"), (-5.0, 5.0, "50"), (-0.5, 0.5, "05")]:
    cn, cp = f"depth_{neg}", f"depth_{pos}"
    if cn in bd_pivot.columns and cp in bd_pivot.columns:
        bd_pivot[f"depth_imb_{tag}"] = (bd_pivot[cn] - bd_pivot[cp]) / (bd_pivot[cn] + bd_pivot[cp] + 1e-10)
kline = kline.join(bd_pivot, how="left")

# depth imbalance trends
for tag in ["05", "10", "20"]:
    col = f"depth_imb_{tag}"
    if col in kline.columns:
        kline[f"{col}_ma5"] = kline[col].rolling(5).mean()
        kline[f"{col}_change"] = kline[col] - kline[col].shift(3)

# ── NEW: Metrics ──
print("[metrics...]")
for col in ["sum_open_interest_value", "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio", "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        kline[col] = metrics[col].reindex(kline.index, method="ffill")

if "sum_open_interest_value" in kline.columns:
    kline["oi_change_5"] = kline["sum_open_interest_value"].pct_change(5)
    kline["oi_change_12"] = kline["sum_open_interest_value"].pct_change(12)
if "sum_taker_long_short_vol_ratio" in kline.columns:
    kline["taker_ls_ma5"] = kline["sum_taker_long_short_vol_ratio"].rolling(5).mean()
    kline["taker_ls_change"] = kline["sum_taker_long_short_vol_ratio"] - kline["sum_taker_long_short_vol_ratio"].shift(3)
if "count_toptrader_long_short_ratio" in kline.columns:
    kline["toptrader_ls_change"] = kline["count_toptrader_long_short_ratio"] - kline["count_toptrader_long_short_ratio"].shift(3)

# ── NEW: Book Ticker ──
print("[book_ticker...]")
if bt.index.tz is None and kline.index.tz is not None:
    bt.index = bt.index.tz_localize(kline.index.tz)
bt_5m = bt.resample("5min").agg({"spread_bps": "mean", "obi": "mean"})
bt_5m.columns = ["spread_bps_mean", "obi_mean"]
kline = kline.join(bt_5m, how="left")
kline["obi_ma5"] = kline["obi_mean"].rolling(5).mean()
kline["obi_trend"] = kline["obi_ma5"] - kline["obi_mean"].rolling(12).mean()

# Funding
kline["funding_rate"] = funding["funding_rate"].reindex(kline.index, method="ffill") if "funding_rate" in funding.columns else np.nan

# Session
kline["hour"] = kline.index.hour

# ── 2. Opportunity zone: 2023+ with book_depth coverage ──
df = kline.loc["2023-01-01":].iloc[60:].dropna(subset=["fwd_6"]).copy()

# Opportunity zone: vol expansion + volume surge
opp = df[(df["vol_squeeze"] > 1.0) & (df["vol_ratio"] > 1.5)].copy()
print(f"\n2023+ bars: {len(df):,}")
print(f"Opportunity zone: {len(opp):,} ({len(opp)/len(df)*100:.1f}%)")

# ── 3. Feature-by-feature predictive power ──
features_to_test = {
    # v1 features
    "cvd_slope_3": "CVD slope 3bar",
    "cvd_slope_5": "CVD slope 5bar",
    "buy_ratio_trend": "Buy ratio trend",
    "mom_3": "Momentum 3bar",
    "ma_20_slope": "MA20 slope",
    "ma_50_slope": "MA50 slope",
    "ma_20_50_cross": "MA20/50 cross",
    "taker_buy_pct_ma5": "Taker buy % ma5",
    "body_ratio_3avg": "Body ratio 3avg",
    "obv_slope_norm": "OBV slope norm",
    "range_pos_20": "Range pos 20bar",
    "vwap_dist": "VWAP distance",
    "funding_rate": "Funding rate",
    # NEW: book depth
    "depth_imb_05": "Depth imb ±0.5%",
    "depth_imb_10": "Depth imb ±1.0%",
    "depth_imb_20": "Depth imb ±2.0%",
    "depth_imb_50": "Depth imb ±5.0%",
    "depth_imb_10_ma5": "Depth imb 1% ma5",
    "depth_imb_10_change": "Depth imb 1% Δ3bar",
    "depth_imb_20_change": "Depth imb 2% Δ3bar",
    # NEW: metrics
    "oi_change_5": "OI change 5bar",
    "oi_change_12": "OI change 12bar",
    "sum_taker_long_short_vol_ratio": "Taker L/S ratio",
    "taker_ls_ma5": "Taker L/S ma5",
    "taker_ls_change": "Taker L/S Δ3bar",
    "count_toptrader_long_short_ratio": "TopTrader L/S",
    "toptrader_ls_change": "TopTrader L/S Δ3bar",
    "count_long_short_ratio": "Retail L/S",
    # NEW: book ticker
    "obi_mean": "OBI (book_ticker)",
    "obi_trend": "OBI trend",
    "spread_bps_mean": "Spread bps",
}

target = opp["fwd_6"]
results = []

print(f"\n{'='*90}")
print(f"{'Feature':<30s} {'Corr':>7s} {'IC':>7s} {'WR_hi':>7s} {'WR_lo':>7s} {'Δ_WR':>7s} {'n':>8s}")
print(f"{'-'*90}")

for feat_col, feat_name in features_to_test.items():
    if feat_col not in opp.columns:
        continue
    x = opp[feat_col]
    y = target
    valid = ~(x.isna() | y.isna() | np.isinf(x))
    if valid.sum() < 300:
        print(f"  {feat_name:<28s} SKIP (n={valid.sum()})")
        continue

    xv = x[valid].values.astype(np.float64)
    yv = y[valid].values.astype(np.float64)

    corr = np.corrcoef(xv, yv)[0, 1]
    ic, _ = spearmanr(xv, yv)

    med = np.median(xv)
    hi_mask = xv >= med; lo_mask = xv < med
    wr_hi = (yv[hi_mask] > FEE).mean()
    wr_lo = (yv[lo_mask] > FEE).mean()
    delta_wr = wr_hi - wr_lo

    results.append({"feature": feat_name, "col": feat_col, "corr": corr, "ic": ic,
                     "wr_hi": wr_hi, "wr_lo": wr_lo, "delta_wr": delta_wr, "n": int(valid.sum())})
    print(f"  {feat_name:<28s} {corr:+.4f} {ic:+.4f} {wr_hi:.3f} {wr_lo:.3f} {delta_wr:+.3f} {int(valid.sum()):>8d}")

results.sort(key=lambda x: abs(x["ic"]), reverse=True)
print(f"\n{'='*90}")
print("── Ranked by |IC| ──")
for i, r in enumerate(results):
    src = "NEW" if r["col"] in ["depth_imb_05","depth_imb_10","depth_imb_20","depth_imb_50",
                                  "depth_imb_10_ma5","depth_imb_10_change","depth_imb_20_change",
                                  "oi_change_5","oi_change_12","sum_taker_long_short_vol_ratio",
                                  "taker_ls_ma5","taker_ls_change","count_toptrader_long_short_ratio",
                                  "toptrader_ls_change","count_long_short_ratio",
                                  "obi_mean","obi_trend","spread_bps_mean"] else "v1"
    marker = "***" if abs(r["delta_wr"]) > 0.03 else ""
    print(f"  {i+1:2d}. [{src:3s}] {r['feature']:<28s} IC={r['ic']:+.4f} Δ_WR={r['delta_wr']:+.4f} {marker}")

# ── 4. Combination analysis ──
print(f"\n{'='*90}")
print("── Combination Analysis (opportunity zone, 2023+) ──\n")

def eval_combo(label, mask, direction, data=opp):
    sub = data[mask]
    n = len(sub)
    if n < 50:
        print(f"  {label:<65s} SKIP (n={n})")
        return None
    if direction == "LONG":
        wr = (sub["fwd_6"] > FEE).mean()
        avg = sub["fwd_6"].mean() * 100
    else:
        wr = (sub["fwd_6"] < -FEE).mean()
        avg = -sub["fwd_6"].mean() * 100
    sel = n / len(data)
    marker = " <<<" if wr > 0.55 else " **" if wr > 0.52 else ""
    print(f"  {label:<65s} {direction} WR={wr:.3f} avg={avg:+.4f}% n={n:,} sel={sel:.3f}{marker}")
    return {"label": label, "dir": direction, "wr": wr, "avg": avg, "n": n, "sel": sel}

# Helper: check column exists and not all NaN in opp
def has(col):
    return col in opp.columns and opp[col].notna().sum() > 100

# --- Single new features as filters ---
print("── A. Depth Imbalance + direction ──")
if has("depth_imb_10"):
    q80 = opp["depth_imb_10"].quantile(0.80)
    q20 = opp["depth_imb_10"].quantile(0.20)
    eval_combo("depth_imb_10 > Q80 (strong bid wall)", opp["depth_imb_10"] > q80, "LONG")
    eval_combo("depth_imb_10 < Q20 (strong ask wall)", opp["depth_imb_10"] < q20, "SHORT")

if has("depth_imb_20"):
    q80 = opp["depth_imb_20"].quantile(0.80)
    q20 = opp["depth_imb_20"].quantile(0.20)
    eval_combo("depth_imb_20 > Q80 (strong bid wall)", opp["depth_imb_20"] > q80, "LONG")
    eval_combo("depth_imb_20 < Q20 (strong ask wall)", opp["depth_imb_20"] < q20, "SHORT")

print("\n── B. OI change combos ──")
if has("oi_change_5"):
    q10 = opp["oi_change_5"].quantile(0.10)
    q90 = opp["oi_change_5"].quantile(0.90)
    eval_combo("OI dropping (Q10) + mom_3>0 → long reversal?",
               (opp["oi_change_5"] < q10) & (opp["mom_3"] > 0), "LONG")
    eval_combo("OI dropping (Q10) + mom_3<0 → short cascade?",
               (opp["oi_change_5"] < q10) & (opp["mom_3"] < 0), "SHORT")
    eval_combo("OI surging (Q90) + mom_3>0 → long continuation?",
               (opp["oi_change_5"] > q90) & (opp["mom_3"] > 0), "LONG")

print("\n── C. Taker L/S ratio combos ──")
if has("sum_taker_long_short_vol_ratio"):
    eval_combo("Taker heavy long (>1.5) → mean revert short",
               opp["sum_taker_long_short_vol_ratio"] > 1.5, "SHORT")
    eval_combo("Taker heavy short (<0.7) → mean revert long",
               opp["sum_taker_long_short_vol_ratio"] < 0.7, "LONG")
    if has("depth_imb_10"):
        eval_combo("Taker heavy short + bid wall → strong long",
                   (opp["sum_taker_long_short_vol_ratio"] < 0.7) & (opp["depth_imb_10"] > opp["depth_imb_10"].quantile(0.7)),
                   "LONG")
        eval_combo("Taker heavy long + ask wall → strong short",
                   (opp["sum_taker_long_short_vol_ratio"] > 1.5) & (opp["depth_imb_10"] < opp["depth_imb_10"].quantile(0.3)),
                   "SHORT")

print("\n── D. Depth + mean-reversion combos ──")
if has("depth_imb_10"):
    di10_q70 = opp["depth_imb_10"].quantile(0.70)
    di10_q30 = opp["depth_imb_10"].quantile(0.30)
    # Bid wall + recently dropped → long bounce
    eval_combo("Bid wall + mom_3<0 (dip into support)",
               (opp["depth_imb_10"] > di10_q70) & (opp["mom_3"] < 0), "LONG")
    eval_combo("Ask wall + mom_3>0 (rally into resistance)",
               (opp["depth_imb_10"] < di10_q30) & (opp["mom_3"] > 0), "SHORT")
    # + VWAP
    eval_combo("Bid wall + below VWAP + mom_3<0",
               (opp["depth_imb_10"] > di10_q70) & (opp["vwap_dist"] < -0.0005) & (opp["mom_3"] < 0), "LONG")
    eval_combo("Ask wall + above VWAP + mom_3>0",
               (opp["depth_imb_10"] < di10_q30) & (opp["vwap_dist"] > 0.0005) & (opp["mom_3"] > 0), "SHORT")

print("\n── E. Multi-signal combos (depth + metrics + price) ──")
if has("depth_imb_10") and has("sum_taker_long_short_vol_ratio"):
    # The dream combo: depth support + crowd wrong + mean-reversion
    eval_combo("Bid wall + taker heavy short + mom_3<0 → LONG",
               (opp["depth_imb_10"] > di10_q70) &
               (opp["sum_taker_long_short_vol_ratio"] < 0.8) &
               (opp["mom_3"] < 0), "LONG")
    eval_combo("Ask wall + taker heavy long + mom_3>0 → SHORT",
               (opp["depth_imb_10"] < di10_q30) &
               (opp["sum_taker_long_short_vol_ratio"] > 1.3) &
               (opp["mom_3"] > 0), "SHORT")

if has("depth_imb_10") and has("oi_change_5"):
    eval_combo("Bid wall + OI dropping + mom_3<0 → LONG (squeeze bounce)",
               (opp["depth_imb_10"] > di10_q70) &
               (opp["oi_change_5"] < opp["oi_change_5"].quantile(0.20)) &
               (opp["mom_3"] < 0), "LONG")
    eval_combo("Ask wall + OI dropping + mom_3>0 → SHORT (squeeze dump)",
               (opp["depth_imb_10"] < di10_q30) &
               (opp["oi_change_5"] < opp["oi_change_5"].quantile(0.20)) &
               (opp["mom_3"] > 0), "SHORT")

print("\n── F. Funding rate combos ──")
if has("funding_rate") and has("depth_imb_10"):
    eval_combo("High funding + ask wall + mom_3>0 → SHORT (overheated)",
               (opp["funding_rate"] > 0.0001) &
               (opp["depth_imb_10"] < di10_q30) &
               (opp["mom_3"] > 0), "SHORT")
    eval_combo("Negative funding + bid wall + mom_3<0 → LONG (oversold)",
               (opp["funding_rate"] < 0) &
               (opp["depth_imb_10"] > di10_q70) &
               (opp["mom_3"] < 0), "LONG")

print("\n── G. OBI (book_ticker) combos ──")
if has("obi_mean") and has("depth_imb_10"):
    obi_hi = opp["obi_mean"].quantile(0.75)
    obi_lo = opp["obi_mean"].quantile(0.25)
    eval_combo("High OBI + bid wall → LONG",
               (opp["obi_mean"] > obi_hi) & (opp["depth_imb_10"] > di10_q70), "LONG")
    eval_combo("Low OBI + ask wall → SHORT",
               (opp["obi_mean"] < obi_lo) & (opp["depth_imb_10"] < di10_q30), "SHORT")

# ── 5. Temporal stability for best combos ──
print(f"\n{'='*90}")
print("── Temporal Stability (by half-year) ──\n")

if has("depth_imb_10"):
    opp_ts = opp.copy()
    opp_ts["period"] = opp_ts.index.to_period("Q").astype(str)

    # Test: bid wall + mom<0 → LONG
    combo_mask = (opp_ts["depth_imb_10"] > di10_q70) & (opp_ts["mom_3"] < 0)
    print("  Combo: Bid wall (Q70) + mom_3<0 → LONG")
    for period in sorted(opp_ts["period"].unique()):
        p_data = opp_ts[opp_ts["period"] == period]
        p_mask = combo_mask.reindex(p_data.index, fill_value=False)
        sub = p_data[p_mask]
        if len(sub) < 20:
            print(f"    {period}: n={len(sub)} (skip)")
            continue
        wr = (sub["fwd_6"] > FEE).mean()
        avg = sub["fwd_6"].mean() * 100
        print(f"    {period}: WR={wr:.3f} avg={avg:+.4f}% n={len(sub)}")

    # Test: ask wall + mom>0 → SHORT
    combo_mask2 = (opp_ts["depth_imb_10"] < di10_q30) & (opp_ts["mom_3"] > 0)
    print("\n  Combo: Ask wall (Q30) + mom_3>0 → SHORT")
    for period in sorted(opp_ts["period"].unique()):
        p_data = opp_ts[opp_ts["period"] == period]
        p_mask = combo_mask2.reindex(p_data.index, fill_value=False)
        sub = p_data[p_mask]
        if len(sub) < 20:
            print(f"    {period}: n={len(sub)} (skip)")
            continue
        wr = (sub["fwd_6"] < -FEE).mean()
        avg = -sub["fwd_6"].mean() * 100
        print(f"    {period}: WR={wr:.3f} avg={avg:+.4f}% n={len(sub)}")

# ── 6. Summary ──
print(f"\n{'='*90}")
print("=== SUMMARY ===\n")
print("Top features by |IC| (new data highlighted):")
for i, r in enumerate(results[:10]):
    src = "NEW" if r["col"] in ["depth_imb_05","depth_imb_10","depth_imb_20","depth_imb_50",
                                  "depth_imb_10_ma5","depth_imb_10_change","depth_imb_20_change",
                                  "oi_change_5","oi_change_12","sum_taker_long_short_vol_ratio",
                                  "taker_ls_ma5","taker_ls_change","count_toptrader_long_short_ratio",
                                  "toptrader_ls_change","count_long_short_ratio",
                                  "obi_mean","obi_trend","spread_bps_mean"] else "v1"
    print(f"  {i+1:2d}. [{src}] {r['feature']:<28s} IC={r['ic']:+.4f}")
print("\nKey question: any combo > 55% WR? Check results above.")
