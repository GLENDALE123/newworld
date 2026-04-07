"""하이브리드 스캘핑 — 동시 포지션 1개 제한 + 현실적 시뮬레이션.

이전 scalp_hybrid_wf.py와 동일한 전략이지만:
- 동시 포지션 1개만 허용
- 이전 포지션이 청산(TP/SL/timeout)될 때까지 새 진입 불가
- 실제 에쿼티 커브 계산
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SYMBOL = "ETHUSDT"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FEE_PCT = 0.02
TP_PCT = 0.10
SL_PCT = 0.05
MAX_HOLD = 6

print(f"=== Hybrid Scalping Realistic Sim: {SYMBOL} ===")
print(f"  TP={TP_PCT}% SL={SL_PCT}% Fee={FEE_PCT}% MAX_POS=1\n")

# ── Data (same pipeline) ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()
k1 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1m.parquet").set_index("timestamp").sort_index()

for d in [tick, k1]:
    if d.index.tz is None and k5.index.tz is not None:
        d.index = d.index.tz_localize(k5.index.tz)

cc = k5["close"].values; hh = k5["high"].values; ll = k5["low"].values; vv = k5["volume"].values
n = len(cc)

# ── Direction: depth+MR agree ──
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
ret3 = ret3.values
depth_dir = np.where(depth_imb > 0, 1, np.where(depth_imb < 0, -1, 0))
mr_dir = -np.sign(ret3)
agree = (depth_dir == mr_dir) & (depth_dir != 0)
direction = np.where(agree, depth_dir, 0).astype(float)

# ── Features ──
ret = pd.Series(cc, index=k5.index).pct_change()
feat = pd.DataFrame(index=k5.index)
feat["vol_5"] = ret.rolling(5).std()
feat["vol_12"] = ret.rolling(12).std()
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"] / feat["vol_60"]
feat["vol_accel"] = feat["vol_5"] / feat["vol_20"]
feat["atr_12"] = (k5["high"] - k5["low"]).rolling(12).mean()
feat["atr_ratio"] = (k5["high"] - k5["low"]) / feat["atr_12"]
feat["bar_range"] = (k5["high"] - k5["low"]) / k5["close"]
feat["vol_ratio"] = pd.Series(vv, index=k5.index) / pd.Series(vv, index=k5.index).rolling(12).mean()
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({"buy_volume":"sum","sell_volume":"sum","cvd_raw":"sum","trade_count":"sum"})
feat["tc"] = tick_5m["trade_count"].reindex(k5.index)
feat["tc_ratio"] = feat["tc"] / feat["tc"].rolling(12).mean()
feat["buy_ratio"] = (tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"])).reindex(k5.index)
feat["cvd_abs"] = tick_5m["cvd_raw"].abs().rolling(5).mean().reindex(k5.index)
k1_ret = k1["close"].pct_change()
feat["intra_range"] = (k1_ret.resample("5min").max() - k1_ret.resample("5min").min()).reindex(k5.index)
feat["bull_1m"] = (k1_ret > 0).astype(int).resample("5min").sum().reindex(k5.index)
feat["last_1m_abs"] = k1_ret.abs().resample("5min").last().reindex(k5.index)
feat["depth_imb"] = depth_imb
feat["depth_imb_abs"] = depth_imb.abs()
for col in ["sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(k5.index, method="ffill")
if "sum_open_interest_value" in metrics.columns:
    oi = metrics["sum_open_interest_value"].reindex(k5.index, method="ffill")
    feat["oi_change_5"] = oi.pct_change(5)
feat["funding_rate"] = funding["funding_rate"].reindex(k5.index, method="ffill") if "funding_rate" in funding.columns else np.nan
feat["ret_1"] = ret
feat["ret_3"] = ret3
feat["body_ratio"] = (k5["close"] - k5["open"]) / (k5["high"] - k5["low"] + 1e-10)
feat["range_pos_20"] = (k5["close"] - k5["low"].rolling(20).min()) / (k5["high"].rolling(20).max() - k5["low"].rolling(20).min() + 1e-10)
vwap = (k5["close"]*k5["volume"]).rolling(20).sum() / (k5["volume"].rolling(20).sum()+1e-10)
feat["vwap_dist"] = k5["close"] / vwap - 1
feat["direction"] = direction
feat["depth_imb_strength"] = depth_imb.abs()
feat["mr_strength"] = np.abs(ret3)
feat["hour"] = k5.index.hour
feat["is_us"] = ((feat["hour"]>=14)&(feat["hour"]<22)).astype(float)
feat = feat.replace([np.inf, -np.inf], np.nan)

model_features = [c for c in feat.columns if c not in ["direction"]]
feat_vals = feat[model_features].values
n_feat = len(model_features)

# Vol filter
va_q80 = feat["vol_accel"].quantile(0.80)
tc_q80 = feat["tc_ratio"].quantile(0.80)
entry_mask = (direction != 0) & (feat["vol_accel"] > va_q80).values & (feat["tc_ratio"] > tc_q80).values

# TP/SL labels for training
tp_hit = np.full(n, np.nan)
for i in range(n - MAX_HOLD):
    d = direction[i]
    if d == 0 or np.isnan(d): continue
    entry = cc[i]
    tp = entry * (1 + d * TP_PCT / 100)
    sl = entry * (1 - d * SL_PCT / 100)
    hit = 0
    for j in range(i+1, i+MAX_HOLD+1):
        if d == 1:
            if hh[j] >= tp: hit = 1; break
            if ll[j] <= sl: hit = -1; break
        else:
            if ll[j] <= tp: hit = 1; break
            if hh[j] >= sl: hit = -1; break
    tp_hit[i] = 1.0 if hit == 1 else 0.0

# ── Model ──
class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.nan_to_num(np.array(X, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
        self.y = torch.tensor(np.nan_to_num(np.array(y, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TPModel(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.LayerNorm(hidden//2), nn.Dropout(0.2),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ── Walk-forward with model training ──
print("[Walk-forward training...]")
splits = []
cursor = pd.Timestamp("2023-01-01", tz=k5.index.tz)
end = k5.index.max()
while cursor + pd.DateOffset(months=8) <= end:
    tr_e = cursor + pd.DateOffset(months=6)
    va_e = tr_e + pd.DateOffset(months=1)
    te_e = va_e + pd.DateOffset(months=1)
    splits.append({"train": (cursor, tr_e), "val": (tr_e, va_e), "test": (va_e, te_e)})
    cursor += pd.DateOffset(months=1)

# Store model predictions for all OOS bars
oos_probs = np.full(n, np.nan)

for i, sp in enumerate(splits):
    tr_s, tr_e = sp["train"]; va_s, va_e = sp["val"]; te_s, te_e = sp["test"]
    tr_idx = (k5.index >= tr_s) & (k5.index < tr_e)
    va_idx = (k5.index >= va_s) & (k5.index < va_e)
    te_idx = (k5.index >= te_s) & (k5.index < te_e)

    tr_entry = tr_idx & entry_mask
    va_entry = va_idx & entry_mask

    X_tr = feat_vals[tr_entry]; y_tr = tp_hit[tr_entry]
    X_va = feat_vals[va_entry]; y_va = tp_hit[va_entry]
    valid_tr = ~np.isnan(y_tr); valid_va = ~np.isnan(y_va)
    X_tr = X_tr[valid_tr]; y_tr = y_tr[valid_tr]
    X_va = X_va[valid_va]; y_va = y_va[valid_va]

    if len(X_tr) < 200 or len(X_va) < 50: continue

    train_ds = DS(X_tr, y_tr); val_ds = DS(X_va, y_va)
    pos_rate = y_tr.mean()
    pos_weight = torch.tensor([(1-pos_rate)/(pos_rate+1e-10)], dtype=torch.float32).to(DEVICE)

    model = TPModel(n_feat, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tl = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

    best_v = float("inf"); wait = 0; best_s = None
    for ep in range(40):
        model.train()
        for X, y in tl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(X), y); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); vloss = 0; nb = 0
        with torch.no_grad():
            for X, y in vl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                vloss += criterion(model(X), y).item(); nb += 1
        vloss /= max(nb,1)
        if vloss < best_v:
            best_v = vloss; wait = 0; best_s = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 7: break
    model.load_state_dict(best_s)

    # Predict on test entry bars
    model.eval()
    te_positions = np.where(te_idx & entry_mask)[0]
    if len(te_positions) == 0: continue
    X_te = feat_vals[te_positions]
    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(np.array(X_te, dtype=np.float64, copy=True), 0.0), dtype=torch.float32).to(DEVICE)
        probs = torch.sigmoid(model(inp)).cpu().numpy()
    for j, pos in enumerate(te_positions):
        oos_probs[pos] = probs[j]

    print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} entries={len(te_positions)}")

# ── Sequential simulation with position limit ──
print(f"\n{'='*80}")
print("=== SEQUENTIAL SIMULATION (max 1 position) ===\n")

start_idx = k5.index.get_loc(k5.loc["2023-01-01":].index[0])

for prob_thr in [0.0, 0.50]:
    for fee_pct in [0.02, 0.08]:
        INITIAL = 10000
        POS_SIZE = 1000

        equity = INITIAL
        equity_curve = [(k5.index[start_idx], equity)]
        trades = []
        in_position = False
        pos_entry_bar = 0
        pos_direction = 0
        pos_entry_price = 0
        pos_tp = 0
        pos_sl = 0

        for idx in range(start_idx, n):
            # Check if current position should be closed
            if in_position:
                bars_held = idx - pos_entry_bar
                closed = False

                if pos_direction == 1:
                    if hh[idx] >= pos_tp:
                        pnl = TP_PCT / 100 - fee_pct / 100
                        closed = True
                    elif ll[idx] <= pos_sl:
                        pnl = -SL_PCT / 100 - fee_pct / 100
                        closed = True
                    elif bars_held >= MAX_HOLD:
                        pnl = (cc[idx] - pos_entry_price) / pos_entry_price - fee_pct / 100
                        closed = True
                else:
                    if ll[idx] <= pos_tp:
                        pnl = TP_PCT / 100 - fee_pct / 100
                        closed = True
                    elif hh[idx] >= pos_sl:
                        pnl = -SL_PCT / 100 - fee_pct / 100
                        closed = True
                    elif bars_held >= MAX_HOLD:
                        pnl = (pos_entry_price - cc[idx]) / pos_entry_price - fee_pct / 100
                        closed = True

                if closed:
                    equity += POS_SIZE * pnl
                    trades.append({
                        "entry_ts": k5.index[pos_entry_bar],
                        "exit_ts": k5.index[idx],
                        "dir": "long" if pos_direction == 1 else "short",
                        "pnl": pnl,
                        "bars": bars_held,
                    })
                    in_position = False

            # Try to open new position
            if not in_position and entry_mask[idx] and direction[idx] != 0:
                # Check model probability
                prob = oos_probs[idx]
                if np.isnan(prob): continue
                if prob < prob_thr: continue

                pos_entry_bar = idx
                pos_direction = int(direction[idx])
                pos_entry_price = cc[idx]
                pos_tp = pos_entry_price * (1 + pos_direction * TP_PCT / 100)
                pos_sl = pos_entry_price * (1 - pos_direction * SL_PCT / 100)
                in_position = True

            equity_curve.append((k5.index[idx], equity))

        tdf = pd.DataFrame(trades)
        if len(tdf) == 0:
            print(f"  prob≥{prob_thr:.2f} fee={fee_pct}%: No trades")
            continue

        final = equity
        total_ret = (final - INITIAL) / INITIAL * 100
        wr = (tdf["pnl"] > 0).mean()
        avg_pnl = tdf["pnl"].mean()
        days = (tdf["exit_ts"].max() - tdf["entry_ts"].min()).days
        ann = total_ret * 365 / max(days, 1)
        tpd = len(tdf) / max(days, 1)

        # Drawdown
        eq_vals = [e[1] for e in equity_curve]
        pk = INITIAL; dd = 0
        for e in eq_vals:
            if e > pk: pk = e
            d = (pk - e) / pk
            if dd < d: dd = d

        # Avg bars held
        avg_bars = tdf["bars"].mean()

        tag = "MAKER" if fee_pct == 0.02 else "TAKER"
        print(f"  [{tag}] prob≥{prob_thr:.2f}: ${INITIAL:,}→${final:,.0f} ({total_ret:+.1f}%) "
              f"trades={len(tdf):,} ({tpd:.1f}/d) WR={wr:.1%} avg={avg_pnl*100:+.4f}% "
              f"DD={dd:.1%} bars={avg_bars:.1f}")

        # Quarterly for maker, prob≥0.0
        if fee_pct == 0.02:
            tdf["quarter"] = tdf["entry_ts"].dt.to_period("Q")
            qr = tdf.groupby("quarter").agg(
                n=("pnl","count"), wr=("pnl", lambda x:(x>0).mean()),
                avg=("pnl","mean"), total=("pnl","sum"))
            if prob_thr == 0.0:
                print(f"\n    {'Q':8s} {'N':>5s} {'WR':>6s} {'Avg':>9s} {'$PnL':>8s} {'Bars/d':>6s}")
                pq = 0
                for p, r in qr.iterrows():
                    qp = r["total"] * POS_SIZE
                    qd = 90  # approx
                    if qp > 0: pq += 1
                    m = "✓" if qp > 0 else "✗"
                    print(f"    {str(p):8s} {int(r['n']):5d} {r['wr']:.1%} {r['avg']*100:+.4f}% {qp:+8,.0f} {m}")
                print(f"    Positive Q: {pq}/{len(qr)}")

            # Direction breakdown
            for d in ["long", "short"]:
                sub = tdf[tdf["dir"] == d]
                if len(sub) > 0:
                    print(f"    {d.upper()}: n={len(sub):,} WR={(sub['pnl']>0).mean():.1%} "
                          f"avg={sub['pnl'].mean()*100:+.4f}%")
            print()

print(f"{'='*80}")
print("=== DONE ===")
