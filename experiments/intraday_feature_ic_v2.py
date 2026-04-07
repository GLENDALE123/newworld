"""단타 Phase 2 v2: 전체 데이터 피처 IC — 상위TF + depth 다단계 + cross-asset."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os

FEE = 0.0008

# BTC 데이터 미리 로드 (cross-asset용)
btc5 = pd.read_parquet("data/merged/BTCUSDT/kline_5m.parquet").set_index("timestamp").sort_index()
btc_metrics = pd.read_parquet("data/merged/BTCUSDT/metrics.parquet").set_index("timestamp").sort_index()
btc_ret_1h = (btc5["close"] / btc5["close"].shift(12) - 1)
btc_ret_4h = (btc5["close"] / btc5["close"].shift(48) - 1)
btc_oi = btc_metrics["sum_open_interest_value"].reindex(btc5.index, method="ffill") if "sum_open_interest_value" in btc_metrics.columns else None
btc_oi_chg = btc_oi.pct_change(12) if btc_oi is not None else None

for SYMBOL in ['ETHUSDT', 'BTCUSDT']:
    print(f"\n{'#'*70}")
    print(f"### {SYMBOL} — Full Feature IC (4h forward)")
    print(f"{'#'*70}\n")

    k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
    metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
    funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()
    depth = pd.read_parquet(f"data/cache/{SYMBOL}/depth_5m.parquet") if os.path.exists(f"data/cache/{SYMBOL}/depth_5m.parquet") else None
    tick_5m = pd.read_parquet(f"data/cache/{SYMBOL}/tick_5m.parquet") if os.path.exists(f"data/cache/{SYMBOL}/tick_5m.parquet") else None

    # 상위 TF (shift(1) — 완성된 봉만!)
    k15 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_15m.parquet").set_index("timestamp").sort_index()
    k1h = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1h.parquet").set_index("timestamp").sort_index()

    cc = k5["close"].values; n = len(cc); si = k5.index; vv = k5["volume"].values
    ret = pd.Series(cc, index=si).pct_change()

    # Target
    fwd_4h = np.full(n, np.nan)
    for i in range(n - 48): fwd_4h[i] = (cc[i + 48] - cc[i]) / cc[i]
    target = pd.Series(fwd_4h, index=si)

    f = pd.DataFrame(index=si)

    # === A. 포지셔닝 (v1과 동일) ===
    oi = metrics["sum_open_interest_value"].reindex(si, method="ffill") if "sum_open_interest_value" in metrics.columns else None
    if oi is not None:
        f["oi_chg_1h"] = oi.pct_change(12)
        f["oi_chg_4h"] = oi.pct_change(48)
        f["oi_chg_12h"] = oi.pct_change(144)
        f["oi_level_z"] = (oi - oi.rolling(288).mean()) / (oi.rolling(288).std() + 1e-10)
        f["oi_accel"] = f["oi_chg_1h"] - f["oi_chg_1h"].shift(12)
    fr = funding["funding_rate"].reindex(si, method="ffill") if "funding_rate" in funding.columns else None
    if fr is not None:
        f["funding"] = fr
        f["funding_abs"] = fr.abs()
        f["funding_extreme"] = (fr - fr.rolling(288).mean()) / (fr.rolling(288).std() + 1e-10)
    taker_ls = metrics["sum_taker_long_short_vol_ratio"].reindex(si, method="ffill") if "sum_taker_long_short_vol_ratio" in metrics.columns else None
    if taker_ls is not None:
        f["taker_ls"] = taker_ls
        f["taker_ls_z"] = (taker_ls - taker_ls.rolling(288).mean()) / (taker_ls.rolling(288).std() + 1e-10)
    top_ls = metrics["count_toptrader_long_short_ratio"].reindex(si, method="ffill") if "count_toptrader_long_short_ratio" in metrics.columns else None
    if top_ls is not None:
        f["toptrader_ls_z"] = (top_ls - top_ls.rolling(288).mean()) / (top_ls.rolling(288).std() + 1e-10)
    retail_ls = metrics["count_long_short_ratio"].reindex(si, method="ffill") if "count_long_short_ratio" in metrics.columns else None
    if retail_ls is not None and top_ls is not None:
        f["smart_retail_div"] = top_ls - retail_ls

    # === B. 가격 구조 (v1과 동일) ===
    f["ret_1h"] = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(12) - 1
    f["ret_4h"] = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(48) - 1
    f["ret_24h"] = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(288) - 1
    f["range_pos_288"] = (k5["close"] - k5["low"].rolling(288).min()) / (k5["high"].rolling(288).max() - k5["low"].rolling(288).min() + 1e-10)
    ma50 = pd.Series(cc, index=si).rolling(50).mean()
    ma200 = pd.Series(cc, index=si).rolling(200).mean()
    f["ma50_slope"] = (ma50 - ma50.shift(12)) / (ma50.shift(12) + 1e-10)
    f["ma_cross"] = ma50 / (ma200 + 1e-10) - 1
    f["vwap_dist_60"] = k5["close"] / ((k5["close"] * k5["volume"]).rolling(60).sum() / (k5["volume"].rolling(60).sum() + 1e-10)) - 1

    # === C. 변동성 ===
    f["vol_12"] = ret.rolling(12).std()
    f["vol_60"] = ret.rolling(60).std()
    f["vol_288"] = ret.rolling(288).std()
    f["vol_squeeze"] = f["vol_12"] / (f["vol_60"] + 1e-10)
    f["vol_regime"] = f["vol_60"] / (f["vol_288"] + 1e-10)

    # === D. 틱 ===
    if tick_5m is not None:
        f["buy_ratio"] = (tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"] + 1e-10)).reindex(si)
        f["buy_ratio_ma12"] = f["buy_ratio"].rolling(12).mean()
        cvd_cum = tick_5m["cvd_raw"].cumsum().reindex(si)
        f["cvd_slope_12"] = cvd_cum.diff(12)
        f["cvd_slope_48"] = cvd_cum.diff(48)

    # === NEW E. 상위 TF (shift(1) 필수!) ===
    # 15m — 완성된 직전 봉
    k15_ret = k15["close"].pct_change()
    k15_body = (k15["close"] - k15["open"]) / (k15["high"] - k15["low"] + 1e-10)
    f["tf15_ret_4"] = (k15["close"].shift(1) / k15["close"].shift(5) - 1).reindex(si, method="ffill")
    f["tf15_body"] = k15_body.shift(1).reindex(si, method="ffill")
    k15_ma8 = k15["close"].rolling(8).mean()
    k15_ma20 = k15["close"].rolling(20).mean()
    f["tf15_trend"] = (k15_ma8.shift(1) / k15_ma20.shift(1) - 1).reindex(si, method="ffill")
    f["tf15_range_pos"] = ((k15["close"].shift(1) - k15["low"].shift(1).rolling(20).min()) / (k15["high"].shift(1).rolling(20).max() - k15["low"].shift(1).rolling(20).min() + 1e-10)).reindex(si, method="ffill")

    # 1h — 완성된 직전 봉
    k1h_ret = k1h["close"].pct_change()
    k1h_body = (k1h["close"] - k1h["open"]) / (k1h["high"] - k1h["low"] + 1e-10)
    f["tf1h_ret_4"] = (k1h["close"].shift(1) / k1h["close"].shift(5) - 1).reindex(si, method="ffill")
    f["tf1h_body"] = k1h_body.shift(1).reindex(si, method="ffill")
    k1h_ma8 = k1h["close"].rolling(8).mean()
    k1h_ma24 = k1h["close"].rolling(24).mean()
    f["tf1h_trend"] = (k1h_ma8.shift(1) / k1h_ma24.shift(1) - 1).reindex(si, method="ffill")
    # 1h vol
    k1h_vol = k1h["close"].pct_change().rolling(8).std()
    f["tf1h_vol"] = k1h_vol.shift(1).reindex(si, method="ffill")

    # === NEW F. Depth 다단계 ===
    if depth is not None:
        for tag in ["05", "10", "20", "50"]:
            col = f"depth_imb_{tag}"
            if col in depth.columns:
                di = depth[col].reindex(si)
                f[col] = di
                f[f"{col}_ma12"] = di.rolling(12).mean()
                f[f"{col}_chg3"] = di - di.shift(3)
        # Depth wall 얇아지는 속도
        if "depth_total_10" in depth.columns:
            dt10 = depth["depth_total_10"].reindex(si)
            f["depth_thin_ratio"] = dt10 / (dt10.rolling(12).mean() + 1e-10)
            f["depth_thin_chg"] = dt10.pct_change(6)

    # === NEW G. Cross-asset (BTC → ETH) ===
    if SYMBOL != 'BTCUSDT':
        f["btc_ret_1h"] = btc_ret_1h.reindex(si)
        f["btc_ret_4h"] = btc_ret_4h.reindex(si)
        if btc_oi_chg is not None:
            f["btc_oi_chg_1h"] = btc_oi_chg.reindex(si)
        # ETH/BTC ratio z-score
        ratio = k5["close"] / btc5["close"].reindex(si)
        f["eth_btc_z"] = (ratio - ratio.rolling(288).mean()) / (ratio.rolling(288).std() + 1e-10)
    else:
        # BTC: use ETH as cross
        eth5 = pd.read_parquet("data/merged/ETHUSDT/kline_5m.parquet").set_index("timestamp").sort_index()
        eth_ret_1h = (eth5["close"] / eth5["close"].shift(12) - 1)
        f["eth_ret_1h"] = eth_ret_1h.reindex(si)

    # === NEW H. 복합 피처 ===
    if oi is not None and fr is not None:
        f["oi_x_funding"] = f["oi_chg_1h"] * f["funding"]
    if oi is not None:
        f["oi_x_ret"] = f["oi_chg_1h"] * f["ret_1h"]
    # OI drop + high vol = liquidation cascade
    if oi is not None:
        f["liq_pressure"] = (-f["oi_chg_1h"]) * f["vol_12"] * f["ret_1h"].abs()
    # Funding divergence from price (high funding but price dropping)
    if fr is not None:
        f["funding_price_div"] = f["funding"] * (-f["ret_4h"])

    f["hour"] = si.hour
    f = f.replace([np.inf, -np.inf], np.nan)
    print(f"Features: {f.shape[1]}")

    # ── IC (2024+) ──
    df = f.loc["2024-01-01":].copy()
    tgt = target.loc[df.index]
    valid_base = tgt.notna()

    results = []
    for col in f.columns:
        x = df[col]
        valid = valid_base & x.notna() & ~np.isinf(x)
        if valid.sum() < 1000: continue
        xv = x[valid].values.astype(np.float64)
        yv = tgt[valid].values.astype(np.float64)
        ic, _ = spearmanr(xv, yv)
        med = np.median(xv)
        hi = yv[xv >= med]; lo = yv[xv < med]
        delta_wr = (hi > FEE).mean() - (lo > FEE).mean()
        results.append({"feature": col, "ic": ic, "delta_wr": delta_wr, "n": int(valid.sum())})

    results.sort(key=lambda x: abs(x["ic"]), reverse=True)

    # Categorize
    def cat(name):
        if name.startswith(("oi_","funding","taker_","toptrader","retail","smart_","liq_")): return "POSITION"
        if name.startswith(("ret_","range_","ma_","ma5","vwap")): return "PRICE"
        if name.startswith(("vol_",)): return "VOL"
        if name.startswith("tf"): return "HTF"
        if name.startswith(("depth_",)): return "DEPTH"
        if name.startswith(("btc_","eth_")): return "CROSS"
        if name.startswith(("cvd_","buy_ratio")): return "FLOW"
        return "OTHER"

    print(f"\n{'Feature':<28s} {'IC':>8s} {'Δ_WR':>8s} {'Type':>10s}")
    print("-" * 60)
    for r in results[:35]:
        c = cat(r["feature"])
        mk = " ★" if abs(r["ic"]) > 0.04 else ""
        print(f"  {r['feature']:<26s} {r['ic']:+.4f} {r['delta_wr']:+.4f} {c:>10s}{mk}")

    # Category averages
    cats = {}
    for r in results:
        c = cat(r["feature"])
        cats.setdefault(c, []).append(abs(r["ic"]))
    print(f"\n  Category avg |IC|:")
    for c, ics in sorted(cats.items(), key=lambda x: -np.mean(x[1])):
        print(f"    {c:10s}: {np.mean(ics):.4f} ({len(ics)} features)")

print(f"\n=== DONE ===")
