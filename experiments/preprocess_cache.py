"""전처리 캐시 생성 — 무거운 데이터를 5m 피처로 미리 집계."""

import pandas as pd
import numpy as np
import os

SYMBOL = "ETHUSDT"
CACHE_DIR = f"data/cache/{SYMBOL}"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"=== Preprocessing Cache: {SYMBOL} ===\n")

k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()

# 1. Book depth → 5m depth_imb
print("[1] book_depth → depth_imb_5m...")
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)

key_pcts = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]
bd_key = bd[bd["percentage"].isin(key_pcts)].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_pivot = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")

result = pd.DataFrame(index=bd_pivot.index)
for neg, pos, tag in [(-0.5, 0.5, "05"), (-1.0, 1.0, "10"), (-2.0, 2.0, "20"), (-5.0, 5.0, "50")]:
    cn, cp = neg, pos
    if cn in bd_pivot.columns and cp in bd_pivot.columns:
        result[f"depth_imb_{tag}"] = (bd_pivot[cn] - bd_pivot[cp]) / (bd_pivot[cn] + bd_pivot[cp] + 1e-10)
        result[f"depth_total_{tag}"] = bd_pivot[cn] + bd_pivot[cp]

result.to_parquet(f"{CACHE_DIR}/depth_5m.parquet")
print(f"  Saved: {result.shape} → {CACHE_DIR}/depth_5m.parquet")
del bd, bd_key, bd_pivot

# 2. Tick bar → 5m aggregation
print("[2] tick_bar → tick_5m...")
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
if tick.index.tz is None and k5.index.tz is not None:
    tick.index = tick.index.tz_localize(k5.index.tz)

tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({
    "buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"
})
tick_5m.to_parquet(f"{CACHE_DIR}/tick_5m.parquet")
print(f"  Saved: {tick_5m.shape} → {CACHE_DIR}/tick_5m.parquet")
del tick

# 3. 1m → 5m micro features
print("[3] kline_1m → micro_5m...")
k1 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1m.parquet").set_index("timestamp").sort_index()
if k1.index.tz is None and k5.index.tz is not None:
    k1.index = k1.index.tz_localize(k5.index.tz)

k1_ret = k1["close"].pct_change()
micro = pd.DataFrame(index=k5.index)
micro["intra_range"] = (k1_ret.resample("5min").max() - k1_ret.resample("5min").min()).reindex(k5.index)
micro["bull_1m"] = (k1_ret > 0).astype(int).resample("5min").sum().reindex(k5.index)
micro["last_1m_ret"] = k1_ret.resample("5min").last().reindex(k5.index)
micro["last_1m_abs"] = k1_ret.abs().resample("5min").last().reindex(k5.index)
micro["first_1m_ret"] = k1_ret.resample("5min").first().reindex(k5.index)
micro["tail_2m_ret"] = k1["close"].resample("5min").apply(
    lambda x: (x.iloc[-1] / x.iloc[-3] - 1) if len(x) >= 3 else np.nan
).reindex(k5.index)
micro["last_1m_vol_pct"] = k1["volume"].resample("5min").apply(
    lambda x: x.iloc[-1] / (x.sum() + 1e-10) if len(x) > 0 else np.nan
).reindex(k5.index)

micro.to_parquet(f"{CACHE_DIR}/micro_5m.parquet")
print(f"  Saved: {micro.shape} → {CACHE_DIR}/micro_5m.parquet")
del k1

# Also for BTC
for sym in ["BTCUSDT"]:
    cache = f"data/cache/{sym}"
    os.makedirs(cache, exist_ok=True)
    print(f"\n[{sym}]")

    k5b = pd.read_parquet(f"data/merged/{sym}/kline_5m.parquet").set_index("timestamp").sort_index()

    # Depth
    print(f"  book_depth...")
    bdb = pd.read_parquet(f"data/merged/{sym}/book_depth.parquet")
    bdb["timestamp"] = pd.to_datetime(bdb["timestamp"])
    if bdb["timestamp"].dt.tz is None and k5b.index.tz is not None:
        bdb["timestamp"] = bdb["timestamp"].dt.tz_localize(k5b.index.tz)
    bd_keyb = bdb[bdb["percentage"].isin(key_pcts)].copy()
    bd_keyb["ts_5m"] = bd_keyb["timestamp"].dt.floor("5min")
    bd_pivb = bd_keyb.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
    resb = pd.DataFrame(index=bd_pivb.index)
    for neg, pos, tag in [(-0.5, 0.5, "05"), (-1.0, 1.0, "10"), (-2.0, 2.0, "20"), (-5.0, 5.0, "50")]:
        if neg in bd_pivb.columns and pos in bd_pivb.columns:
            resb[f"depth_imb_{tag}"] = (bd_pivb[neg] - bd_pivb[pos]) / (bd_pivb[neg] + bd_pivb[pos] + 1e-10)
            resb[f"depth_total_{tag}"] = bd_pivb[neg] + bd_pivb[pos]
    resb.to_parquet(f"{cache}/depth_5m.parquet")
    del bdb, bd_keyb, bd_pivb

    # Tick
    print(f"  tick_bar...")
    tickb = pd.read_parquet(f"data/merged/{sym}/tick_bar.parquet").set_index("timestamp").sort_index()
    if tickb.index.tz is None and k5b.index.tz is not None:
        tickb.index = tickb.index.tz_localize(k5b.index.tz)
    tickb["cvd_raw"] = tickb["buy_volume"] - tickb["sell_volume"]
    tick_5mb = tickb.resample("5min").agg({"buy_volume":"sum","sell_volume":"sum","cvd_raw":"sum","trade_count":"sum"})
    tick_5mb.to_parquet(f"{cache}/tick_5m.parquet")
    del tickb

    # 1m micro
    print(f"  kline_1m...")
    k1b = pd.read_parquet(f"data/merged/{sym}/kline_1m.parquet").set_index("timestamp").sort_index()
    if k1b.index.tz is None and k5b.index.tz is not None:
        k1b.index = k1b.index.tz_localize(k5b.index.tz)
    k1b_ret = k1b["close"].pct_change()
    microb = pd.DataFrame(index=k5b.index)
    microb["intra_range"] = (k1b_ret.resample("5min").max() - k1b_ret.resample("5min").min()).reindex(k5b.index)
    microb["bull_1m"] = (k1b_ret > 0).astype(int).resample("5min").sum().reindex(k5b.index)
    microb["last_1m_ret"] = k1b_ret.resample("5min").last().reindex(k5b.index)
    microb["last_1m_abs"] = k1b_ret.abs().resample("5min").last().reindex(k5b.index)
    microb.to_parquet(f"{cache}/micro_5m.parquet")
    del k1b
    print(f"  Done.")

print(f"\n=== Cache complete ===")
