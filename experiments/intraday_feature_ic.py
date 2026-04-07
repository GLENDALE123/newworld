"""단타 Phase 2: 피처별 IC 측정 — 4h forward return 방향 예측력.

스캘핑과 다른 피처셋:
- 스캘핑: buy_ratio, tc_ratio, vol_accel (미시구조)
- 단타: OI change, funding, taker L/S, toptrader L/S (포지셔닝)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEE = 0.0008

for SYMBOL in ['ETHUSDT', 'BTCUSDT']:
    print(f"\n{'#'*70}")
    print(f"### {SYMBOL} — 4h Direction Feature IC")
    print(f"{'#'*70}\n")

    k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
    metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
    funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()
    depth = pd.read_parquet(f"data/cache/{SYMBOL}/depth_5m.parquet") if __import__('os').path.exists(f"data/cache/{SYMBOL}/depth_5m.parquet") else None
    tick_5m = pd.read_parquet(f"data/cache/{SYMBOL}/tick_5m.parquet") if __import__('os').path.exists(f"data/cache/{SYMBOL}/tick_5m.parquet") else None

    cc = k5["close"].values; n = len(cc); si = k5.index; vv = k5["volume"].values
    ret = pd.Series(cc, index=si).pct_change()

    # Target: 4h (48-bar) forward return
    fwd_4h = np.full(n, np.nan)
    for i in range(n - 48):
        fwd_4h[i] = (cc[i + 48] - cc[i]) / cc[i]
    target = pd.Series(fwd_4h, index=si)

    # ── Features ──
    f = pd.DataFrame(index=si)

    # === A. 포지셔닝 피처 (단타 핵심) ===
    oi = metrics["sum_open_interest_value"].reindex(si, method="ffill") if "sum_open_interest_value" in metrics.columns else None
    if oi is not None:
        f["oi_chg_1h"] = oi.pct_change(12)
        f["oi_chg_4h"] = oi.pct_change(48)
        f["oi_chg_12h"] = oi.pct_change(144)
        f["oi_level_z"] = (oi - oi.rolling(288).mean()) / (oi.rolling(288).std() + 1e-10)  # 1-day z-score
        f["oi_accel"] = f["oi_chg_1h"] - f["oi_chg_1h"].shift(12)  # OI 변화 가속도

    fr = funding["funding_rate"].reindex(si, method="ffill") if "funding_rate" in funding.columns else None
    if fr is not None:
        f["funding"] = fr
        f["funding_abs"] = fr.abs()
        f["funding_ma3"] = fr.rolling(3 * 96).mean()  # 3-period funding avg (8h*3=24h)
        f["funding_extreme"] = (fr - fr.rolling(288).mean()) / (fr.rolling(288).std() + 1e-10)

    taker_ls = metrics["sum_taker_long_short_vol_ratio"].reindex(si, method="ffill") if "sum_taker_long_short_vol_ratio" in metrics.columns else None
    if taker_ls is not None:
        f["taker_ls"] = taker_ls
        f["taker_ls_z"] = (taker_ls - taker_ls.rolling(288).mean()) / (taker_ls.rolling(288).std() + 1e-10)
        f["taker_ls_chg"] = taker_ls - taker_ls.shift(12)

    top_ls = metrics["count_toptrader_long_short_ratio"].reindex(si, method="ffill") if "count_toptrader_long_short_ratio" in metrics.columns else None
    if top_ls is not None:
        f["toptrader_ls"] = top_ls
        f["toptrader_ls_z"] = (top_ls - top_ls.rolling(288).mean()) / (top_ls.rolling(288).std() + 1e-10)

    retail_ls = metrics["count_long_short_ratio"].reindex(si, method="ffill") if "count_long_short_ratio" in metrics.columns else None
    if retail_ls is not None:
        f["retail_ls"] = retail_ls
        f["smart_retail_div"] = top_ls - retail_ls if top_ls is not None else np.nan

    # === B. 가격 구조 피처 ===
    f["ret_1h"] = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(12) - 1
    f["ret_4h"] = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(48) - 1
    f["ret_24h"] = pd.Series(cc, index=si) / pd.Series(cc, index=si).shift(288) - 1
    f["range_pos_96"] = (k5["close"] - k5["low"].rolling(96).min()) / (k5["high"].rolling(96).max() - k5["low"].rolling(96).min() + 1e-10)
    f["range_pos_288"] = (k5["close"] - k5["low"].rolling(288).min()) / (k5["high"].rolling(288).max() - k5["low"].rolling(288).min() + 1e-10)
    ma50 = pd.Series(cc, index=si).rolling(50).mean()
    ma200 = pd.Series(cc, index=si).rolling(200).mean()
    f["ma50_slope"] = (ma50 - ma50.shift(12)) / (ma50.shift(12) + 1e-10)
    f["ma200_slope"] = (ma200 - ma200.shift(24)) / (ma200.shift(24) + 1e-10)
    f["ma_cross"] = ma50 / (ma200 + 1e-10) - 1
    f["vwap_dist_60"] = k5["close"] / ((k5["close"] * k5["volume"]).rolling(60).sum() / (k5["volume"].rolling(60).sum() + 1e-10)) - 1

    # === C. 변동성 피처 ===
    f["vol_12"] = ret.rolling(12).std()
    f["vol_60"] = ret.rolling(60).std()
    f["vol_288"] = ret.rolling(288).std()
    f["vol_squeeze"] = f["vol_12"] / (f["vol_60"] + 1e-10)
    f["vol_regime"] = f["vol_60"] / (f["vol_288"] + 1e-10)

    # === D. 거래량/틱 피처 ===
    f["vol_ratio_12"] = pd.Series(vv, index=si) / (pd.Series(vv, index=si).rolling(12).mean() + 1e-10)
    f["vol_ratio_60"] = pd.Series(vv, index=si) / (pd.Series(vv, index=si).rolling(60).mean() + 1e-10)
    if tick_5m is not None:
        f["buy_ratio"] = (tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"] + 1e-10)).reindex(si)
        f["buy_ratio_ma12"] = f["buy_ratio"].rolling(12).mean()
        cvd_cum = tick_5m["cvd_raw"].cumsum().reindex(si)
        f["cvd_slope_12"] = cvd_cum.diff(12)
        f["cvd_slope_48"] = cvd_cum.diff(48)

    # === E. Depth 피처 ===
    if depth is not None:
        di10 = depth["depth_imb_10"].reindex(si) if "depth_imb_10" in depth.columns else None
        if di10 is not None:
            f["depth_imb_10"] = di10
            f["depth_imb_ma12"] = di10.rolling(12).mean()

    # === F. 시간 피처 ===
    f["hour"] = si.hour
    f["hour_sin"] = np.sin(2 * np.pi * si.hour / 24)
    f["day_of_week"] = si.dayofweek

    # === G. 복합 피처 ===
    if oi is not None and fr is not None:
        f["oi_x_funding"] = f["oi_chg_1h"] * f["funding"]  # OI 감소 + 높은 funding = 청산 압력
    if oi is not None:
        f["oi_x_ret"] = f["oi_chg_1h"] * f["ret_1h"]  # OI↓ + price↓ = 롱 청산

    f = f.replace([np.inf, -np.inf], np.nan)
    print(f"Features: {f.shape[1]}")

    # ── IC 측정 (2024+ 데이터) ──
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
        # Median split WR
        med = np.median(xv)
        hi = yv[xv >= med]; lo = yv[xv < med]
        wr_hi = (hi > FEE).mean(); wr_lo = (lo > FEE).mean()
        delta_wr = wr_hi - wr_lo
        results.append({"feature": col, "ic": ic, "delta_wr": delta_wr, "n": int(valid.sum())})

    results.sort(key=lambda x: abs(x["ic"]), reverse=True)

    print(f"\n{'Feature':<25s} {'IC':>8s} {'Δ_WR':>8s} {'n':>8s} {'Type':>10s}")
    print("-" * 65)
    for r in results[:30]:
        # Categorize
        if r["feature"] in ["oi_chg_1h","oi_chg_4h","oi_chg_12h","oi_level_z","oi_accel",
                              "funding","funding_abs","funding_ma3","funding_extreme",
                              "taker_ls","taker_ls_z","taker_ls_chg",
                              "toptrader_ls","toptrader_ls_z","retail_ls","smart_retail_div",
                              "oi_x_funding","oi_x_ret"]:
            cat = "POSITION"
        elif r["feature"] in ["ret_1h","ret_4h","ret_24h","range_pos_96","range_pos_288",
                                "ma50_slope","ma200_slope","ma_cross","vwap_dist_60"]:
            cat = "PRICE"
        elif r["feature"].startswith("vol_") or r["feature"].startswith("vol_"):
            cat = "VOL"
        else:
            cat = "OTHER"

        mk = " ★" if abs(r["ic"]) > 0.05 else ""
        print(f"  {r['feature']:<23s} {r['ic']:+.4f} {r['delta_wr']:+.4f} {r['n']:8,} {cat:>10s}{mk}")

    # Top positioning features vs top price features
    pos_feats = [r for r in results if r["feature"] in ["oi_chg_1h","oi_chg_4h","oi_chg_12h","oi_accel",
                 "funding","funding_extreme","taker_ls_z","toptrader_ls_z","oi_x_funding","oi_x_ret"]]
    price_feats = [r for r in results if r["feature"] in ["ret_1h","ret_4h","ret_24h","ma50_slope","ma_cross","range_pos_288"]]

    print(f"\n  Positioning features avg |IC|: {np.mean([abs(r['ic']) for r in pos_feats]):.4f}" if pos_feats else "")
    print(f"  Price features avg |IC|: {np.mean([abs(r['ic']) for r in price_feats]):.4f}" if price_feats else "")

print(f"\n{'='*60}")
print("=== DONE ===")
