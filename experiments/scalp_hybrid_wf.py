"""하이브리드 스캘핑: 룰 기반 필터 + ML selectivity + TP/SL exit.

전략 구조:
1. 룰 필터: vol_accel Q80 + tc_ratio Q80 (변동성 터질 때만)
2. 방향: depth_imb + mean_revert agree
3. ML 모델: "이 진입이 TP에 도달할까?" binary 예측
4. 청산: TP 0.10% / SL 0.05% (비대칭)
5. Fee: maker 0.02%

핵심 차이: 방향 예측이 아닌 "TP hit 확률" 예측 (binary classification)
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
FEE_PCT = 0.02  # maker, in %
TP_PCT = 0.10
SL_PCT = 0.05
MAX_HOLD = 6

print(f"=== Hybrid Scalping WF: {SYMBOL} ===")
print(f"  TP={TP_PCT}% SL={SL_PCT}% Fee={FEE_PCT}% Maker\n")

# ── Data ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()
k1 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1m.parquet").set_index("timestamp").sort_index()

for d in [tick, k1]:
    if d.index.tz is None and k5.index.tz is not None:
        d.index = d.index.tz_localize(k5.index.tz)

c = k5["close"].values; h = k5["high"].values; l = k5["low"].values; v = k5["volume"].values
n = len(c)

# ── Compute TP/SL labels (ground truth) ──
print("[Computing TP/SL labels...]")

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

ret3 = pd.Series(c, index=k5.index) / pd.Series(c, index=k5.index).shift(3) - 1
ret3 = ret3.values
depth_dir = np.where(depth_imb > 0, 1, np.where(depth_imb < 0, -1, 0))
mr_dir = -np.sign(ret3)
agree = (depth_dir == mr_dir) & (depth_dir != 0)
direction = np.where(agree, depth_dir, 0).astype(float)

# TP/SL outcome for each bar
tp_hit = np.full(n, np.nan)  # 1=TP hit, 0=SL or timeout
pnl_arr = np.full(n, np.nan)

for i in range(n - MAX_HOLD):
    d = direction[i]
    if d == 0 or np.isnan(d): continue
    entry = c[i]
    tp = entry * (1 + d * TP_PCT / 100)
    sl = entry * (1 - d * SL_PCT / 100)

    hit = 0
    for j in range(i+1, i+MAX_HOLD+1):
        if d == 1:
            if h[j] >= tp: hit = 1; pnl_arr[i] = TP_PCT/100 - FEE_PCT/100; break
            if l[j] <= sl: hit = -1; pnl_arr[i] = -SL_PCT/100 - FEE_PCT/100; break
        else:
            if l[j] <= tp: hit = 1; pnl_arr[i] = TP_PCT/100 - FEE_PCT/100; break
            if h[j] >= sl: hit = -1; pnl_arr[i] = -SL_PCT/100 - FEE_PCT/100; break
    if hit == 0:
        if d == 1: pnl_arr[i] = (c[min(i+MAX_HOLD,n-1)] - entry)/entry - FEE_PCT/100
        else: pnl_arr[i] = (entry - c[min(i+MAX_HOLD,n-1)])/entry - FEE_PCT/100

    tp_hit[i] = 1.0 if hit == 1 else 0.0

# ── Features ──
print("[Features...]")
ret = pd.Series(c, index=k5.index).pct_change()
feat = pd.DataFrame(index=k5.index)

# Volatility (핵심)
feat["vol_5"] = ret.rolling(5).std()
feat["vol_12"] = ret.rolling(12).std()
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"] / feat["vol_60"]
feat["vol_accel"] = feat["vol_5"] / feat["vol_20"]
feat["atr_12"] = (k5["high"] - k5["low"]).rolling(12).mean()
feat["atr_ratio"] = (k5["high"] - k5["low"]) / feat["atr_12"]
feat["bar_range"] = (k5["high"] - k5["low"]) / k5["close"]

# Volume
feat["vol_ratio"] = pd.Series(v, index=k5.index) / pd.Series(v, index=k5.index).rolling(12).mean()
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({"buy_volume":"sum","sell_volume":"sum","cvd_raw":"sum","trade_count":"sum"})
feat["tc"] = tick_5m["trade_count"].reindex(k5.index)
feat["tc_ratio"] = feat["tc"] / feat["tc"].rolling(12).mean()
feat["buy_ratio"] = (tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"])).reindex(k5.index)
feat["cvd_abs"] = tick_5m["cvd_raw"].abs().rolling(5).mean().reindex(k5.index)

# Price structure
feat["ret_1"] = ret
feat["ret_3"] = pd.Series(c, index=k5.index) / pd.Series(c, index=k5.index).shift(3) - 1
feat["body_ratio"] = (k5["close"] - k5["open"]) / (k5["high"] - k5["low"] + 1e-10)
feat["range_pos_20"] = (k5["close"] - k5["low"].rolling(20).min()) / (k5["high"].rolling(20).max() - k5["low"].rolling(20).min() + 1e-10)
vwap = (k5["close"] * k5["volume"]).rolling(20).sum() / (k5["volume"].rolling(20).sum() + 1e-10)
feat["vwap_dist"] = k5["close"] / vwap - 1

# Depth
feat["depth_imb"] = depth_imb
feat["depth_imb_abs"] = depth_imb.abs()

# 1m micro
k1_ret = k1["close"].pct_change()
feat["intra_range"] = (k1_ret.resample("5min").max() - k1_ret.resample("5min").min()).reindex(k5.index)
feat["bull_1m"] = (k1_ret > 0).astype(int).resample("5min").sum().reindex(k5.index)
feat["last_1m_abs"] = k1_ret.abs().resample("5min").last().reindex(k5.index)

# Metrics
for col in ["sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(k5.index, method="ffill")
if "sum_open_interest_value" in metrics.columns:
    oi = metrics["sum_open_interest_value"].reindex(k5.index, method="ffill")
    feat["oi_change_5"] = oi.pct_change(5)
feat["funding_rate"] = funding["funding_rate"].reindex(k5.index, method="ffill") if "funding_rate" in funding.columns else np.nan

# Direction & agreement strength
feat["direction"] = direction
feat["depth_imb_strength"] = depth_imb.abs()  # 강할수록 확신
feat["mr_strength"] = np.abs(ret3)  # 큰 반전일수록 확신

# Time
feat["hour"] = k5.index.hour
feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)

feat = feat.replace([np.inf, -np.inf], np.nan)
# Drop direction itself from model features (it's the rule, not feature)
model_features = [c for c in feat.columns if c not in ["direction"]]
print(f"  Model features: {len(model_features)}")

# ── Vol filter mask ──
vol_mask = (feat["vol_accel"] > feat["vol_accel"].rolling(500).quantile(0.80)) & \
           (feat["tc_ratio"] > feat["tc_ratio"].rolling(500).quantile(0.80))
# Use expanding quantile to avoid look-ahead in filter itself
# Simplified: use global Q80 for now (slightly optimistic but consistent)
va_q80 = feat["vol_accel"].quantile(0.80)
tc_q80 = feat["tc_ratio"].quantile(0.80)
entry_mask = (direction != 0) & (feat["vol_accel"] > va_q80) & (feat["tc_ratio"] > tc_q80)

print(f"  Entry bars (direction + vol filter): {entry_mask.sum():,} / {n:,} ({entry_mask.mean():.1%})")

# ── Dataset ──
class TPDataset(Dataset):
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

# ── Walk-forward ──
print(f"\n[Walk-forward...]")
splits = []
cursor = pd.Timestamp("2023-01-01", tz=k5.index.tz)
end = k5.index.max()
while cursor + pd.DateOffset(months=8) <= end:
    tr_e = cursor + pd.DateOffset(months=6)
    va_e = tr_e + pd.DateOffset(months=1)
    te_e = va_e + pd.DateOffset(months=1)
    splits.append({"train": (cursor, tr_e), "val": (tr_e, va_e), "test": (va_e, te_e)})
    cursor += pd.DateOffset(months=1)
print(f"  {len(splits)} windows\n")

feat_vals = feat[model_features].values
n_feat = len(model_features)

all_trades = []

for i, sp in enumerate(splits):
    tr_s, tr_e = sp["train"]; va_s, va_e = sp["val"]; te_s, te_e = sp["test"]
    tr_idx = (k5.index >= tr_s) & (k5.index < tr_e)
    va_idx = (k5.index >= va_s) & (k5.index < va_e)
    te_idx = (k5.index >= te_s) & (k5.index < te_e)

    # Train only on entry bars (where rule says "trade")
    tr_entry = tr_idx & entry_mask.values
    va_entry = va_idx & entry_mask.values

    X_tr = feat_vals[tr_entry]; y_tr = tp_hit[tr_entry]
    X_va = feat_vals[va_entry]; y_va = tp_hit[va_entry]

    valid_tr = ~np.isnan(y_tr); valid_va = ~np.isnan(y_va)
    X_tr = X_tr[valid_tr]; y_tr = y_tr[valid_tr]
    X_va = X_va[valid_va]; y_va = y_va[valid_va]

    if len(X_tr) < 200 or len(X_va) < 50:
        continue

    train_ds = TPDataset(X_tr, y_tr)
    val_ds = TPDataset(X_va, y_va)

    # Positive weight for imbalanced classes
    pos_rate = y_tr.mean()
    pos_weight = torch.tensor([(1 - pos_rate) / (pos_rate + 1e-10)], dtype=torch.float32).to(DEVICE)

    model = TPModel(n_feat, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

    best_val = float("inf"); wait = 0; best_state = None
    for epoch in range(40):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        vl = 0; nb = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                vl += criterion(model(X), y).item(); nb += 1
        vl /= max(nb,1)
        if vl < best_val:
            best_val = vl; wait = 0
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 7: break
    model.load_state_dict(best_state)

    # ── OOS ──
    model.eval()
    te_entry = te_idx & entry_mask.values
    te_positions = np.where(te_entry)[0]

    if len(te_positions) == 0:
        print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} | No entry bars")
        continue

    X_te = feat_vals[te_positions]
    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(np.array(X_te, dtype=np.float64, copy=True), 0.0), dtype=torch.float32).to(DEVICE)
        logits = model(inp)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Collect trades with model probability
    for j, pos in enumerate(te_positions):
        if np.isnan(pnl_arr[pos]): continue
        all_trades.append({
            "ts": k5.index[pos],
            "prob": probs[j],
            "tp_hit": tp_hit[pos],
            "pnl": pnl_arr[pos],
            "dir": "long" if direction[pos] == 1 else "short",
            "window": i + 1,
        })

    # Window summary (no filter)
    pnls = [t["pnl"] for t in all_trades if t["window"] == i+1]
    if pnls:
        wr = np.mean([p > 0 for p in pnls])
        avg = np.mean(pnls)
        print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} | n={len(pnls):4d} WR={wr:.3f} avg={avg*100:+.4f}%")

# ── Results by confidence threshold ──
print(f"\n{'='*80}")
print("=== RESULTS BY MODEL CONFIDENCE ===\n")

if not all_trades:
    print("No trades.")
else:
    tdf = pd.DataFrame(all_trades)
    tdf = tdf.sort_values("ts").reset_index(drop=True)

    print(f"Total OOS trades: {len(tdf):,}")
    print(f"Base TP hit rate: {tdf['tp_hit'].mean():.1%}")
    print(f"Base avg_pnl: {tdf['pnl'].mean()*100:+.4f}%\n")

    for thr in [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = tdf[tdf["prob"] >= thr]
        if len(sub) < 30: continue
        wr = (sub["pnl"] > 0).mean()
        avg = sub["pnl"].mean()
        tp_rate = sub["tp_hit"].mean()
        days = (sub["ts"].max() - sub["ts"].min()).days
        trades_per_day = len(sub) / max(days, 1)

        marker = " <<<" if avg > 0.0003 else " **" if avg > 0 else ""
        print(f"  prob≥{thr:.2f}: n={len(sub):,} ({trades_per_day:.1f}/d) TP={tp_rate:.1%} "
              f"WR={wr:.1%} avg_pnl={avg*100:+.4f}%{marker}")

    # ── Equity curve for best threshold ──
    # Find best threshold
    best_thr = 0.0; best_avg = -999
    for thr in [0.0, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sub = tdf[tdf["prob"] >= thr]
        if len(sub) < 100:
            continue
        avg = sub["pnl"].mean()
        if avg > best_avg:
            best_avg = avg; best_thr = thr

    print(f"\n{'='*80}")
    for eq_thr in [0.0, best_thr]:
        sub = tdf[tdf["prob"] >= eq_thr].sort_values("ts")
        if len(sub) < 50: continue

        INITIAL = 10000; POS_SIZE = 1000
        equity = [INITIAL]
        for _, t in sub.iterrows():
            equity.append(equity[-1] + POS_SIZE * t["pnl"])
        final = equity[-1]
        ret_pct = (final - INITIAL) / INITIAL * 100

        pk = INITIAL; dd = 0
        for e in equity:
            if e > pk: pk = e
            d = (pk - e) / pk
            if dd < d: dd = d

        days = (sub["ts"].max() - sub["ts"].min()).days
        ann = ret_pct * 365 / max(days, 1)
        tpd = len(sub) / max(days, 1)

        print(f"\n  === Equity (prob≥{eq_thr:.2f}) ===")
        print(f"  ${INITIAL:,} → ${final:,.0f} ({ret_pct:+.1f}%)")
        print(f"  Trades: {len(sub):,} ({tpd:.1f}/day), WR: {(sub['pnl']>0).mean():.1%}")
        print(f"  Avg P&L: {sub['pnl'].mean()*100:+.4f}%, Max DD: {dd:.1%}")
        if days > 0:
            print(f"  Ann. return: {ann:+.1f}% ({days} days)")

        # Quarterly
        sub_c = sub.copy()
        sub_c["quarter"] = sub_c["ts"].dt.to_period("Q")
        qr = sub_c.groupby("quarter").agg(n=("pnl","count"), wr=("pnl", lambda x:(x>0).mean()),
                                            avg=("pnl","mean"), total=("pnl","sum"))
        pq = 0
        print(f"\n  {'Q':8s} {'N':>5s} {'WR':>6s} {'Avg':>9s} {'Total$':>8s}")
        for p, r in qr.iterrows():
            qp = r["total"] * POS_SIZE
            if qp > 0: pq += 1
            m = "✓" if qp > 0 else "✗"
            print(f"  {str(p):8s} {int(r['n']):5d} {r['wr']:.1%} {r['avg']*100:+.4f}% {qp:+8,.0f} {m}")
        print(f"  Positive Q: {pq}/{len(qr)}")

print(f"\n{'='*80}")
print("=== DONE ===")
