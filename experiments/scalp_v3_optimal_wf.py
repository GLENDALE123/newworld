"""v3 최적 설정 full walk-forward + 에쿼티 커브.

설정: MLP-256 + CE loss + conf≥0.5 + rr=1.5 + hold=6
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scalping.labeler_v3 import generate_scalp_labels_v3

SYMBOL = "ETHUSDT"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HOLD = 6
FEE = 0.0008
MIN_RR = 1.5
CONF_THRESHOLDS = [0.0, 0.35, 0.4, 0.45]

print(f"=== v3 Optimal Walk-Forward: {SYMBOL} ===")
print(f"  MLP-256, CE, conf thresholds={CONF_THRESHOLDS}, rr={MIN_RR}, hold={HOLD}\n")

# ── Data + Labels + Features (same as baseline) ──
kline = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)

print("[Labels...]")
labels = generate_scalp_labels_v3(kline, max_holds=[HOLD], fee=FEE, min_rr=MIN_RR)

print("[Features...]")
c = kline["close"]; o = kline["open"]; h = kline["high"]; l = kline["low"]; v = kline["volume"]
ret = c.pct_change()
feat = pd.DataFrame(index=kline.index)
feat["ret_1"] = ret
feat["ret_3"] = c / c.shift(3) - 1
feat["ret_6"] = c / c.shift(6) - 1
feat["body_ratio"] = (c - o) / (h - l + 1e-10)
feat["body_ratio_3avg"] = feat["body_ratio"].rolling(3).mean()
feat["upper_wick"] = (h - np.maximum(o, c)) / (h - l + 1e-10)
feat["lower_wick"] = (np.minimum(o, c) - l) / (h - l + 1e-10)
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"] / feat["vol_60"]
feat["atr_12"] = (h - l).rolling(12).mean()
feat["atr_ratio"] = (h - l) / feat["atr_12"]
feat["vol_ratio"] = v / v.rolling(12).mean()
ma20 = c.rolling(20).mean(); ma50 = c.rolling(50).mean()
feat["ma_20_slope"] = (ma20 - ma20.shift(3)) / (ma20.shift(3) + 1e-10)
feat["ma_50_slope"] = (ma50 - ma50.shift(5)) / (ma50.shift(5) + 1e-10)
feat["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)
vwap = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
feat["vwap_dist"] = c / vwap - 1
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"})
feat = feat.join(tick_5m[["cvd_raw", "trade_count"]], how="left")
feat["cvd_cumsum"] = feat["cvd_raw"].cumsum()
feat["cvd_slope_5"] = feat["cvd_cumsum"] - feat["cvd_cumsum"].shift(5)
feat["buy_ratio"] = tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"])
feat["taker_buy_pct"] = kline["taker_buy_volume"] / kline["volume"]
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and kline.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(kline.index.tz)
bd_key = bd[bd["percentage"].isin([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_pivot = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
bd_pivot.columns = [f"depth_{c}" for c in bd_pivot.columns]
for neg, pos, tag in [(-1.0, 1.0, "10"), (-2.0, 2.0, "20")]:
    cn, cp = f"depth_{neg}", f"depth_{pos}"
    if cn in bd_pivot.columns and cp in bd_pivot.columns:
        bd_pivot[f"depth_imb_{tag}"] = (bd_pivot[cn] - bd_pivot[cp]) / (bd_pivot[cn] + bd_pivot[cp] + 1e-10)
feat = feat.join(bd_pivot[[c for c in bd_pivot.columns if "imb" in c]], how="left")
for col in ["sum_open_interest_value", "sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(kline.index, method="ffill")
if "sum_open_interest_value" in feat.columns:
    feat["oi_change_5"] = feat["sum_open_interest_value"].pct_change(5)
    feat.drop(columns=["sum_open_interest_value"], inplace=True)
feat["funding_rate"] = funding["funding_rate"].reindex(kline.index, method="ffill") if "funding_rate" in funding.columns else np.nan
feat["hammer_score"] = feat["lower_wick"].rolling(3).max() * (1 - feat["range_pos_20"])
feat["shooting_star"] = feat["upper_wick"].rolling(3).max() * feat["range_pos_20"]
feat.drop(columns=["cvd_raw", "cvd_cumsum"], inplace=True, errors="ignore")
feat = feat.replace([np.inf, -np.inf], np.nan)

action = labels[f"action_{HOLD}"]
edge_long = labels[f"edge_long_{HOLD}"]
edge_short = labels[f"edge_short_{HOLD}"]
n_feat = feat.shape[1]
print(f"  {n_feat} features, {len(feat):,} bars\n")

# ── Dataset / Model ──
class DS(Dataset):
    def __init__(self, X, y, edge):
        self.X = torch.tensor(np.nan_to_num(np.array(X, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
        y_arr = np.nan_to_num(np.array(y, dtype=np.float64, copy=True), nan=1.0)
        self.y = torch.tensor(y_arr + 1, dtype=torch.long)
        self.edge = torch.tensor(np.nan_to_num(np.array(edge, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.edge[i]

class MLP256(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.2),
            nn.Linear(128, 3),
        )
    def forward(self, x): return self.net(x)

# ── Walk-forward ──
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 1

def make_splits(index, start_year=2023):
    start = pd.Timestamp(f"{start_year}-01-01", tz=index.tz)
    end = index.max()
    splits = []
    cursor = start
    while cursor + pd.DateOffset(months=TRAIN_MONTHS + VAL_MONTHS + TEST_MONTHS) <= end:
        tr_e = cursor + pd.DateOffset(months=TRAIN_MONTHS)
        va_e = tr_e + pd.DateOffset(months=VAL_MONTHS)
        te_e = va_e + pd.DateOffset(months=TEST_MONTHS)
        splits.append({"train": (cursor, tr_e), "val": (tr_e, va_e), "test": (va_e, te_e)})
        cursor += pd.DateOffset(months=TEST_MONTHS)
    return splits

splits = make_splits(feat.index)
print(f"[Walk-forward: {len(splits)} windows]\n")

all_trades = {thr: [] for thr in CONF_THRESHOLDS}
all_oos = {thr: [] for thr in CONF_THRESHOLDS}

for i, sp in enumerate(splits):
    tr_s, tr_e = sp["train"]; va_s, va_e = sp["val"]; te_s, te_e = sp["test"]
    tr_mask = (feat.index >= tr_s) & (feat.index < tr_e)
    va_mask = (feat.index >= va_s) & (feat.index < va_e)
    te_mask = (feat.index >= te_s) & (feat.index < te_e)

    edge_arr = np.maximum(
        np.nan_to_num(np.array(edge_long.values, dtype=np.float64, copy=True), 0),
        np.nan_to_num(np.array(edge_short.values, dtype=np.float64, copy=True), 0)
    )

    train_ds = DS(feat.values[tr_mask], action.values[tr_mask], edge_arr[tr_mask])
    val_ds = DS(feat.values[va_mask], action.values[va_mask], edge_arr[va_mask])
    if len(train_ds) < 1000: continue

    y_t = train_ds.y.numpy()
    counts = np.bincount(y_t, minlength=3).astype(float); counts[counts==0] = 1
    w = 1.0/counts; w = w/w.sum()*3
    cw = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    model = MLP256(n_feat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    criterion = nn.CrossEntropyLoss(weight=cw)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

    best_val = float("inf"); wait = 0; best_state = None
    for epoch in range(50):
        model.train()
        for X, y, e in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        vl = 0; nb = 0
        with torch.no_grad():
            for X, y, e in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                vl += criterion(model(X), y).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best_val:
            best_val = vl; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 7: break
    model.load_state_dict(best_state)

    # OOS
    model.eval()
    te_idx = feat.index[te_mask]
    X_test = feat.values[te_mask]
    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(np.array(X_test, dtype=np.float64, copy=True), 0.0), dtype=torch.float32).to(DEVICE)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    max_prob = probs.max(axis=1)
    pred_action = preds.astype(float) - 1
    el_test = edge_long.values[te_mask]
    es_test = edge_short.values[te_mask]

    # Print for first threshold only
    printed = False
    for thr in CONF_THRESHOLDS:
        conf_mask = max_prob >= thr if thr > 0 else np.ones(len(preds), dtype=bool)
        for j in range(len(te_idx)):
            if not conf_mask[j]: continue
            pa = pred_action[j]
            if pa == 0: continue
            direction = "long" if pa == 1 else "short"
            edge = el_test[j] if pa == 1 else es_test[j]
            all_trades[thr].append({"ts": te_idx[j], "dir": direction, "edge": edge,
                                     "conf": max_prob[j], "window": i+1})

        cm = conf_mask & (pred_action != 0)
        n_trades = cm.sum()
        if n_trades > 0:
            pnl_arr = np.where(pred_action[cm] == 1, el_test[cm], es_test[cm])
            wr = (pnl_arr > 0).mean(); avg_pnl = np.nanmean(pnl_arr)
        else:
            wr = 0; avg_pnl = 0
        all_oos[thr].append({"window": i+1, "test": f"{te_s.date()}~{te_e.date()}",
                              "n_trades": int(n_trades), "wr": wr, "avg_pnl": avg_pnl})

        if not printed:
            marker = "✓" if avg_pnl > 0 else "✗"
            print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} | trades={n_trades:5d} | "
                  f"WR={wr:.3f} | pnl={avg_pnl*100:+.4f}% {marker}")
            printed = True

# ── Equity curve per threshold ──
INITIAL = 10000
POS_SIZE = 1000

for thr in CONF_THRESHOLDS:
    print(f"\n{'='*80}")
    print(f"=== EQUITY CURVE (conf≥{thr}) ===\n")

    if not all_trades[thr]:
        print("  No trades.")
        continue

    trades_df = pd.DataFrame(all_trades[thr]).sort_values("ts").reset_index(drop=True)
    trades_df["edge"] = trades_df["edge"].astype(float)

    equity = [INITIAL]
    for _, t in trades_df.iterrows():
        equity.append(equity[-1] + POS_SIZE * t["edge"])

    final = equity[-1]
    total_return = (final - INITIAL) / INITIAL * 100
    n_trades = len(trades_df)
    avg_pnl = trades_df["edge"].mean()
    wr = (trades_df["edge"] > 0).mean()

    peak = INITIAL; max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > max_dd: max_dd = dd

    days = (trades_df["ts"].max() - trades_df["ts"].min()).days
    ann_return = total_return * 365 / days if days > 0 else 0
    trades_per_day = n_trades / days if days > 0 else 0

    print(f"  ${INITIAL:,} → ${final:,.0f} ({total_return:+.1f}%)")
    print(f"  Trades: {n_trades:,} ({trades_per_day:.1f}/day)")
    print(f"  WR: {wr:.1%}, Avg P&L: {avg_pnl*100:+.4f}%")
    print(f"  Max DD: {max_dd:.1%}")
    if days > 0:
        print(f"  Ann. return: {ann_return:+.1f}% ({days} days)")

    for d in ["long", "short"]:
        sub = trades_df[trades_df["dir"] == d]
        if len(sub) > 0:
            print(f"  {d.upper()}: n={len(sub):,} WR={(sub['edge']>0).mean():.1%} avg={sub['edge'].mean()*100:+.4f}%")

    # Quarterly
    trades_df["quarter"] = trades_df["ts"].dt.to_period("Q")
    quarterly = trades_df.groupby("quarter").agg(
        n_trades=("edge", "count"),
        avg_pnl=("edge", "mean"),
        wr=("edge", lambda x: (x > 0).mean()),
        total_pnl=("edge", "sum"),
    )
    print(f"\n  {'Quarter':8s} {'Trades':>6s} {'WR':>6s} {'Avg':>9s} {'Total$':>9s}")
    pos_q = 0; tot_q = 0
    for period, row in quarterly.iterrows():
        q_pnl = row["total_pnl"] * POS_SIZE
        tot_q += 1
        if q_pnl > 0: pos_q += 1
        m = "✓" if q_pnl > 0 else "✗"
        print(f"  {str(period):8s} {int(row['n_trades']):6d} {row['wr']:.1%} "
              f"{row['avg_pnl']*100:+.4f}% {q_pnl:+9,.0f} {m}")
    print(f"  Positive Q: {pos_q}/{tot_q}")

print(f"\n{'='*80}")
print("=== DONE ===")
