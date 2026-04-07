"""스캘핑 최적화: 동적 TP/SL + Kelly sizing + wider barriers for taker.

목표: 1) Taker에서도 양수 달성 2) Maker에서 수익률 극대화
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

SYMBOL = "ETHUSDT"
print(f"=== Scalping Optimization ({SYMBOL}) ===\n")

k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")

if tick.index.tz is None and k5.index.tz is not None:
    tick.index = tick.index.tz_localize(k5.index.tz)

cc = k5["close"].values; hh = k5["high"].values; ll = k5["low"].values; vv = k5["volume"].values
n = len(cc)

# Direction: depth+MR agree
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)
bd_key = bd[bd["percentage"].isin([-1.0, 1.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_piv = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
depth_imb = pd.Series(np.nan, index=k5.index)
if -1.0 in bd_piv.columns and 1.0 in bd_piv.columns:
    depth_imb = ((bd_piv[-1.0] - bd_piv[1.0]) / (bd_piv[-1.0] + bd_piv[1.0] + 1e-10)).reindex(k5.index)

ret3 = pd.Series(cc, index=k5.index) / pd.Series(cc, index=k5.index).shift(3) - 1
depth_dir = np.where(depth_imb > 0, 1, np.where(depth_imb < 0, -1, 0))
mr_dir = -np.sign(ret3.values)
agree = (depth_dir == mr_dir) & (depth_dir != 0)
direction = np.where(agree, depth_dir, 0).astype(float)

# Vol filter
ret = pd.Series(cc, index=k5.index).pct_change()
vol_accel = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-10)
tick_5m = tick.resample("5min").agg({"trade_count": "sum"})
tc = tick_5m["trade_count"].reindex(k5.index)
tc_ratio = tc / tc.rolling(12).mean()

va_q80 = vol_accel.quantile(0.80)
tc_q80 = tc_ratio.quantile(0.80)
entry_mask = (direction != 0) & (vol_accel > va_q80).values & (tc_ratio > tc_q80).values

# ATR for dynamic sizing
atr = (k5["high"] - k5["low"]).rolling(12).mean().values

start_idx = k5.index.get_loc(k5.loc["2023-01-01":].index[0])

# ── 1. TP/SL Grid Search (2023+, no position limit for speed) ──
print("── 1. TP/SL Grid Search ──\n")
print(f"{'TP%':>5s} {'SL%':>5s} {'Hold':>4s} {'Fee':>5s} {'TP_hit':>7s} {'WR':>6s} {'avg_pnl':>9s} {'n':>7s} {'EV/trade':>9s}")
print("-" * 75)

best_taker = {"avg": -999}
best_maker = {"avg": -999}

for tp_pct in [0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    for sl_pct in [0.03, 0.05, 0.08, 0.10, 0.15]:
        for max_hold in [6, 12]:
            results = {}
            for fee_pct in [0.02, 0.08]:
                pnls = []
                tp_hits = 0; total = 0
                for i in range(start_idx, n - max_hold):
                    if not entry_mask[i]: continue
                    d = direction[i]
                    if d == 0: continue
                    entry = cc[i]
                    tp = entry * (1 + d * tp_pct / 100)
                    sl = entry * (1 - d * sl_pct / 100)

                    hit = 0
                    for j in range(i+1, min(i+max_hold+1, n)):
                        if d == 1:
                            if hh[j] >= tp: hit = 1; break
                            if ll[j] <= sl: hit = -1; break
                        else:
                            if ll[j] <= tp: hit = 1; break
                            if hh[j] >= sl: hit = -1; break

                    if hit == 1:
                        pnl = tp_pct/100 - fee_pct/100
                        tp_hits += 1
                    elif hit == -1:
                        pnl = -sl_pct/100 - fee_pct/100
                    else:
                        if d == 1: pnl = (cc[min(i+max_hold,n-1)] - entry)/entry - fee_pct/100
                        else: pnl = (entry - cc[min(i+max_hold,n-1)])/entry - fee_pct/100
                    pnls.append(pnl)
                    total += 1

                if total < 100: continue
                pnls = np.array(pnls)
                wr = (pnls > 0).mean()
                avg = pnls.mean()
                tp_rate = tp_hits / total

                results[fee_pct] = {"wr": wr, "avg": avg, "tp_rate": tp_rate, "n": total}

                if fee_pct == 0.08 and avg > best_taker["avg"]:
                    best_taker = {"tp": tp_pct, "sl": sl_pct, "hold": max_hold, "avg": avg, "wr": wr, "n": total, "tp_rate": tp_rate}
                if fee_pct == 0.02 and avg > best_maker["avg"]:
                    best_maker = {"tp": tp_pct, "sl": sl_pct, "hold": max_hold, "avg": avg, "wr": wr, "n": total, "tp_rate": tp_rate}

            # Print only promising
            for fee_pct in [0.02, 0.08]:
                if fee_pct not in results: continue
                r = results[fee_pct]
                if r["avg"] > -0.001 or (fee_pct == 0.02 and r["avg"] > 0):
                    tag = "MAKER" if fee_pct == 0.02 else "TAKER"
                    marker = " <<<" if r["avg"] > 0.0003 else " **" if r["avg"] > 0 else ""
                    print(f"  {tp_pct:5.2f} {sl_pct:5.2f} {max_hold:4d} {tag:>5s} "
                          f"{r['tp_rate']:7.1%} {r['wr']:6.1%} {r['avg']*100:+8.4f}% {r['n']:>7,}{marker}")

print(f"\n  BEST MAKER: TP={best_maker['tp']}% SL={best_maker['sl']}% Hold={best_maker['hold']} "
      f"WR={best_maker['wr']:.1%} avg={best_maker['avg']*100:+.4f}% n={best_maker['n']:,}")
print(f"  BEST TAKER: TP={best_taker['tp']}% SL={best_taker['sl']}% Hold={best_taker['hold']} "
      f"WR={best_taker['wr']:.1%} avg={best_taker['avg']*100:+.4f}% n={best_taker['n']:,}")

# ── 2. Dynamic TP/SL (ATR-proportional) ──
print(f"\n{'='*75}")
print("── 2. Dynamic TP/SL (ATR-proportional) ──\n")

for tp_mult in [1.5, 2.0, 2.5, 3.0]:
    for sl_mult in [0.5, 0.75, 1.0]:
        for max_hold in [6, 12]:
            for fee_pct in [0.02, 0.08]:
                pnls = []
                tp_hits = 0; total = 0
                for i in range(start_idx, n - max_hold):
                    if not entry_mask[i]: continue
                    d = direction[i]
                    if d == 0 or np.isnan(atr[i]) or atr[i] <= 0: continue
                    entry = cc[i]
                    tp_dist = tp_mult * atr[i]
                    sl_dist = sl_mult * atr[i]
                    tp = entry + d * tp_dist
                    sl = entry - d * sl_dist

                    hit = 0
                    for j in range(i+1, min(i+max_hold+1, n)):
                        if d == 1:
                            if hh[j] >= tp: hit = 1; break
                            if ll[j] <= sl: hit = -1; break
                        else:
                            if ll[j] <= tp: hit = 1; break
                            if hh[j] >= sl: hit = -1; break

                    if hit == 1:
                        pnl = tp_dist/entry - fee_pct/100
                        tp_hits += 1
                    elif hit == -1:
                        pnl = -sl_dist/entry - fee_pct/100
                    else:
                        if d == 1: pnl = (cc[min(i+max_hold,n-1)] - entry)/entry - fee_pct/100
                        else: pnl = (entry - cc[min(i+max_hold,n-1)])/entry - fee_pct/100
                    pnls.append(pnl)
                    total += 1

                if total < 100: continue
                pnls = np.array(pnls)
                wr = (pnls > 0).mean()
                avg = pnls.mean()
                tp_rate = tp_hits / total

                if avg > 0 or (fee_pct == 0.08 and avg > -0.0005):
                    tag = "MAKER" if fee_pct == 0.02 else "TAKER"
                    marker = " <<<" if avg > 0.0003 else " **" if avg > 0 else ""
                    print(f"  tp={tp_mult:.1f}x sl={sl_mult:.2f}x hold={max_hold} {tag:>5s} "
                          f"TP={tp_rate:.1%} WR={wr:.1%} avg={avg*100:+.4f}% n={total:,}{marker}")

# ── 3. Kelly Criterion ──
print(f"\n{'='*75}")
print("── 3. Kelly Position Sizing ──\n")

# Use best maker params
tp = best_maker["tp"]; sl = best_maker["sl"]; hold = best_maker["hold"]
fee = 0.02
print(f"Using TP={tp}% SL={sl}% Hold={hold} Maker fee={fee}%")

# Simulate with Kelly
pnls_seq = []
for i in range(start_idx, n - hold):
    if not entry_mask[i]: continue
    d = direction[i]
    if d == 0: continue
    entry = cc[i]
    tp_p = entry * (1 + d * tp / 100)
    sl_p = entry * (1 - d * sl / 100)
    hit = 0
    for j in range(i+1, min(i+hold+1, n)):
        if d == 1:
            if hh[j] >= tp_p: hit = 1; break
            if ll[j] <= sl_p: hit = -1; break
        else:
            if ll[j] <= tp_p: hit = 1; break
            if hh[j] >= sl_p: hit = -1; break
    if hit == 1: pnl = tp/100 - fee/100
    elif hit == -1: pnl = -sl/100 - fee/100
    else:
        if d == 1: pnl = (cc[min(i+hold,n-1)] - entry)/entry - fee/100
        else: pnl = (entry - cc[min(i+hold,n-1)])/entry - fee/100
    pnls_seq.append(pnl)

pnls_arr = np.array(pnls_seq)
win_pnl = pnls_arr[pnls_arr > 0].mean()
loss_pnl = abs(pnls_arr[pnls_arr <= 0].mean())
wr = (pnls_arr > 0).mean()

# Kelly fraction: f = (p*b - q) / b where b = win/loss ratio
b = win_pnl / loss_pnl
kelly = (wr * b - (1 - wr)) / b
half_kelly = kelly / 2
quarter_kelly = kelly / 4

print(f"  WR={wr:.3f} avg_win={win_pnl*100:.4f}% avg_loss={loss_pnl*100:.4f}%")
print(f"  Win/Loss ratio: {b:.2f}")
print(f"  Full Kelly: {kelly:.1%}")
print(f"  Half Kelly: {half_kelly:.1%}")
print(f"  Quarter Kelly: {quarter_kelly:.1%}")

# Simulate with different leverage
for label, frac in [("Fixed $1K", None), ("Quarter Kelly", quarter_kelly),
                     ("Half Kelly", half_kelly), ("Full Kelly", kelly)]:
    equity = 10000
    eq_curve = [equity]
    max_eq = equity; max_dd = 0

    for pnl in pnls_arr:
        if frac is not None:
            pos_size = equity * frac
        else:
            pos_size = 1000
        equity += pos_size * pnl
        eq_curve.append(equity)
        if equity > max_eq: max_eq = equity
        dd = (max_eq - equity) / max_eq
        if dd > max_dd: max_dd = dd

    final = equity
    ret_pct = (final - 10000) / 10000 * 100
    days = 730  # approx 2 years
    ann = ret_pct * 365 / days

    print(f"\n  [{label}]")
    print(f"    $10,000 → ${final:,.0f} ({ret_pct:+.1f}%) Ann: {ann:+.1f}%")
    print(f"    Max DD: {max_dd:.1%}")

    # With leverage caps
    if frac is not None:
        for max_lev in [3, 5, 10, 20]:
            equity = 10000; max_eq = equity; max_dd = 0
            for pnl in pnls_arr:
                pos_size = min(equity * frac, equity * max_lev)
                equity += pos_size * pnl
                if equity > max_eq: max_eq = equity
                dd = (max_eq - equity) / max_eq
                if dd > max_dd: max_dd = dd
                if equity <= 0: equity = 0; break
            ret_pct = (equity - 10000) / 10000 * 100
            ann = ret_pct * 365 / days
            print(f"    Capped {max_lev}x: ${equity:,.0f} ({ret_pct:+.1f}%) DD={max_dd:.1%} Ann={ann:+.1f}%")

print(f"\n{'='*75}")
print("=== DONE ===")
