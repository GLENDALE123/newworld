"""TP exit 전략: 진입 후 6bar 내에 TP에 도달하면 즉시 청산.

핵심: close-to-close가 아닌 "TP hit 확률"이 진짜 스캘핑 메트릭.
depth 방향 진입 후, 짧은 TP(0.05~0.15%)에 먼저 도달하는 비율은?
"""

import pandas as pd
import numpy as np

SYMBOL = "ETHUSDT"
print(f"=== TP Exit Strategy Analysis ({SYMBOL}) ===\n")

k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()

if tick.index.tz is None and k5.index.tz is not None:
    tick.index = tick.index.tz_localize(k5.index.tz)

c = k5["close"].values; h = k5["high"].values; l = k5["low"].values; v = k5["volume"].values
n = len(c)

# Book depth
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)
bd_key = bd[bd["percentage"].isin([-1.0, 1.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_piv = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
depth_imb = pd.Series(np.nan, index=k5.index)
if -1.0 in bd_piv.columns and 1.0 in bd_piv.columns:
    depth_imb = ((bd_piv[-1.0] - bd_piv[1.0]) / (bd_piv[-1.0] + bd_piv[1.0] + 1e-10)).reindex(k5.index)

# Volatility features
ret = pd.Series(c).pct_change().values
feat_vol_accel = pd.Series(c).pct_change().rolling(5).std().values / (pd.Series(c).pct_change().rolling(20).std().values + 1e-10)
feat_vol_ratio = pd.Series(v) / pd.Series(v).rolling(12).mean()
feat_tc = tick.resample("5min").agg({"trade_count": "sum"})["trade_count"].reindex(k5.index)
feat_tc_ratio = (feat_tc / feat_tc.rolling(12).mean()).values

# MR direction
feat_ret3 = pd.Series(c) / pd.Series(c).shift(3) - 1

di_arr = depth_imb.values

# ── Compute TP/SL hit rates for various scenarios ──
print("Computing TP/SL exit outcomes...\n")

MAX_HOLD = 6
tp_levels = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]  # in %
sl_levels = [0.05, 0.10, 0.15, 0.20, 0.30]  # in %

# For each direction rule and TP/SL combo, compute hit rates
def compute_tpsl(direction_arr, tp_pct, sl_pct, max_hold):
    """direction_arr: +1/-1/0 per bar. Returns arrays of outcome per bar."""
    outcomes = np.full(n, np.nan)  # +1=TP, -1=SL, 0=timeout
    pnl = np.full(n, np.nan)

    for i in range(n - max_hold):
        d = direction_arr[i]
        if d == 0 or np.isnan(d):
            continue
        entry = c[i]
        tp = entry * (1 + d * tp_pct / 100)
        sl = entry * (1 - d * sl_pct / 100)

        hit = 0
        for j in range(i+1, i+max_hold+1):
            if d == 1:
                if h[j] >= tp:
                    hit = 1; pnl[i] = tp_pct / 100; break
                if l[j] <= sl:
                    hit = -1; pnl[i] = -sl_pct / 100; break
            else:
                if l[j] <= tp:  # tp is below for short
                    hit = 1; pnl[i] = tp_pct / 100; break
                if h[j] >= sl:  # sl is above for short
                    hit = -1; pnl[i] = -sl_pct / 100; break

        if hit == 0:
            # timeout: close at last bar
            if d == 1:
                pnl[i] = (c[min(i+max_hold, n-1)] - entry) / entry
            else:
                pnl[i] = (entry - c[min(i+max_hold, n-1)]) / entry
        outcomes[i] = hit

    return outcomes, pnl

# Direction methods
directions = {
    "depth_imb": np.where(di_arr > 0, 1, np.where(di_arr < 0, -1, 0)).astype(float),
    "mean_revert": -np.sign(feat_ret3.values),
    "depth+MR_agree": np.where(
        (np.where(di_arr > 0, 1, -1) == -np.sign(feat_ret3.values)),
        np.where(di_arr > 0, 1, -1), 0
    ).astype(float),
}

# Focus on 2023+
start_idx = k5.index.get_loc(k5.loc["2023-01-01":].index[0])

print(f"{'Direction':<20s} {'TP%':>5s} {'SL%':>5s} {'Fee':>5s} {'TP_hit':>7s} {'SL_hit':>7s} {'TO':>5s} "
      f"{'WR':>6s} {'avg_pnl':>9s} {'n':>8s}")
print("-" * 100)

for dir_name, dir_arr in directions.items():
    for tp_pct in [0.05, 0.08, 0.10, 0.15, 0.20]:
        for sl_pct in [0.05, 0.10, 0.15]:
            for fee_label, fee_pct in [("0.02", 0.02), ("0.08", 0.08)]:
                outcomes, raw_pnl = compute_tpsl(dir_arr, tp_pct, sl_pct, MAX_HOLD)

                # Filter to 2023+
                valid = ~np.isnan(outcomes)
                valid[:start_idx] = False
                oc = outcomes[valid]
                rp = raw_pnl[valid]

                if len(oc) < 500: continue

                tp_hit = (oc == 1).mean()
                sl_hit = (oc == -1).mean()
                timeout = (oc == 0).mean()

                # Net P&L after fee
                net_pnl = rp - fee_pct / 100
                wr = (net_pnl > 0).mean()
                avg = net_pnl.mean()

                marker = " <<<" if avg > 0 else ""
                # Only print promising ones or key benchmarks
                if avg > -0.0003 or (tp_pct == 0.10 and sl_pct == 0.10 and fee_label == "0.02"):
                    print(f"  {dir_name:<18s} {tp_pct:5.2f} {sl_pct:5.2f} {fee_label:>5s} "
                          f"{tp_hit:7.1%} {sl_hit:7.1%} {timeout:5.1%} "
                          f"{wr:6.1%} {avg*100:+8.4f}% {len(oc):>8,}{marker}")

# ── Vol filter + TP/SL ──
print(f"\n{'='*80}")
print("── With Vol Filter (vol_accel Q80 + tc_ratio Q80) ──\n")

va = pd.Series(feat_vol_accel)
tc = pd.Series(feat_tc_ratio)
vol_mask = (va > va.quantile(0.80)).values & (tc > tc.quantile(0.80)).values

for dir_name, dir_arr in directions.items():
    filtered_dir = np.where(vol_mask, dir_arr, 0)

    for tp_pct in [0.05, 0.08, 0.10, 0.15]:
        for sl_pct in [0.05, 0.10]:
            for fee_label, fee_pct in [("0.02", 0.02), ("0.08", 0.08)]:
                outcomes, raw_pnl = compute_tpsl(filtered_dir, tp_pct, sl_pct, MAX_HOLD)
                valid = ~np.isnan(outcomes)
                valid[:start_idx] = False
                oc = outcomes[valid]; rp = raw_pnl[valid]
                if len(oc) < 100: continue

                tp_hit = (oc == 1).mean()
                sl_hit = (oc == -1).mean()
                net_pnl = rp - fee_pct / 100
                wr = (net_pnl > 0).mean()
                avg = net_pnl.mean()

                marker = " <<<" if avg > 0 else ""
                if avg > -0.0003 or (tp_pct == 0.10 and sl_pct == 0.10):
                    print(f"  {dir_name:<18s} {tp_pct:5.2f} {sl_pct:5.2f} {fee_label:>5s} "
                          f"TP={tp_hit:5.1%} SL={sl_hit:5.1%} WR={wr:5.1%} "
                          f"avg={avg*100:+7.4f}% n={len(oc):,}{marker}")

print(f"\n{'='*80}")
print("=== DONE ===")
