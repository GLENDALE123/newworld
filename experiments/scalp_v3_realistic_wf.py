"""v3 walk-forward — look-ahead 제거, 현실적 P&L 계산.

평가 방법:
A. 고정 청산: N바 후 close-to-close (fee 차감)
B. TP/SL 배리어: ATR 기반 TP/SL, 먼저 닿는 쪽 청산
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

print(f"=== v3 Realistic Walk-Forward: {SYMBOL} ===\n")

# ── Data ──
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
vwap_v = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
feat["vwap_dist"] = c / vwap_v - 1
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
bd_pivot.columns = [f"depth_{cc}" for cc in bd_pivot.columns]
for neg, pos, tag in [(-1.0, 1.0, "10"), (-2.0, 2.0, "20")]:
    cn, cp = f"depth_{neg}", f"depth_{pos}"
    if cn in bd_pivot.columns and cp in bd_pivot.columns:
        bd_pivot[f"depth_imb_{tag}"] = (bd_pivot[cn] - bd_pivot[cp]) / (bd_pivot[cn] + bd_pivot[cp] + 1e-10)
feat = feat.join(bd_pivot[[cc for cc in bd_pivot.columns if "imb" in cc]], how="left")
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

# ── Pre-compute realistic P&L arrays ──
print("[Pre-computing realistic P&L...]")
closes = kline["close"].values
highs = kline["high"].values
lows = kline["low"].values
atr_arr = feat["atr_12"].values

n = len(closes)

# A. Fixed exit: close-to-close at bar N
fwd_returns = {}
for bars in [3, 4, 6]:
    fwd = np.full(n, np.nan)
    for i in range(n - bars):
        fwd[i] = closes[i + bars] / closes[i] - 1
    fwd_returns[bars] = fwd

# B. TP/SL barrier exit (ATR-based)
def compute_barrier_pnl(tp_mult, sl_mult, max_bars):
    """For each bar, compute P&L if entering long or short with TP/SL."""
    pnl_long = np.full(n, np.nan)
    pnl_short = np.full(n, np.nan)

    for i in range(n - max_bars):
        a = atr_arr[i]
        if np.isnan(a) or a <= 0:
            continue
        entry = closes[i]
        tp_dist = tp_mult * a
        sl_dist = sl_mult * a

        # Long
        long_pnl = None
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if highs[j] >= entry + tp_dist:
                long_pnl = tp_dist / entry - FEE
                break
            if lows[j] <= entry - sl_dist:
                long_pnl = -sl_dist / entry - FEE
                break
        if long_pnl is None:
            long_pnl = (closes[min(i + max_bars, n-1)] - entry) / entry - FEE
        pnl_long[i] = long_pnl

        # Short
        short_pnl = None
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lows[j] <= entry - tp_dist:
                short_pnl = tp_dist / entry - FEE
                break
            if highs[j] >= entry + sl_dist:
                short_pnl = -sl_dist / entry - FEE
                break
        if short_pnl is None:
            short_pnl = (entry - closes[min(i + max_bars, n-1)]) / entry - FEE
        pnl_short[i] = short_pnl

    return pnl_long, pnl_short

barrier_pnl = {}
for tp, sl, bars in [(2.0, 1.0, 6), (1.5, 1.0, 6), (2.0, 1.5, 6)]:
    tag = f"tp{tp}_sl{sl}_h{bars}"
    pl, ps = compute_barrier_pnl(tp, sl, bars)
    barrier_pnl[tag] = {"long": pl, "short": ps}

print("  Done.\n")

# ── Dataset / Model ──
action = labels[f"action_{HOLD}"]
edge_long = labels[f"edge_long_{HOLD}"]
edge_short = labels[f"edge_short_{HOLD}"]
n_feat = feat.shape[1]

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
splits = []
cursor = pd.Timestamp("2023-01-01", tz=kline.index.tz)
end = kline.index.max()
while cursor + pd.DateOffset(months=8) <= end:
    tr_e = cursor + pd.DateOffset(months=6)
    va_e = tr_e + pd.DateOffset(months=1)
    te_e = va_e + pd.DateOffset(months=1)
    splits.append({"train": (cursor, tr_e), "val": (tr_e, va_e), "test": (va_e, te_e)})
    cursor += pd.DateOffset(months=1)
print(f"[Walk-forward: {len(splits)} windows]\n")

# Collect all OOS predictions
oos_records = []  # (index_pos, pred_action, conf)

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

    # OOS predictions
    model.eval()
    te_positions = np.where(te_mask)[0]
    X_test = feat.values[te_mask]
    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(np.array(X_test, dtype=np.float64, copy=True), 0.0), dtype=torch.float32).to(DEVICE)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    max_prob = probs.max(axis=1)
    pred_action = preds.astype(float) - 1

    for j in range(len(te_positions)):
        if pred_action[j] == 0: continue  # HOLD
        oos_records.append({
            "pos": te_positions[j],
            "pred": pred_action[j],  # +1 or -1
            "conf": max_prob[j],
            "window": i + 1,
        })

    # Brief status
    n_trades = (pred_action != 0).sum()
    print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} | trades={n_trades:5d}")

# ── Evaluate with realistic P&L ──
print(f"\n{'='*80}")
print("=== REALISTIC P&L EVALUATION ===\n")

oos_df = pd.DataFrame(oos_records)
if oos_df.empty:
    print("No trades.")
else:
    print(f"Total OOS predictions: {len(oos_df):,}\n")

    # Attach timestamps
    oos_df["ts"] = kline.index[oos_df["pos"].values]

    for conf_thr in [0.0, 0.35, 0.4]:
        sub = oos_df[oos_df["conf"] >= conf_thr] if conf_thr > 0 else oos_df
        if len(sub) == 0:
            print(f"  conf≥{conf_thr}: No trades\n")
            continue

        print(f"  ── conf≥{conf_thr} ({len(sub):,} trades) ──")

        # A. Fixed exit
        for exit_bars in [3, 6]:
            fwd = fwd_returns[exit_bars]
            pnls = []
            for _, row in sub.iterrows():
                pos = row["pos"]
                fwd_ret = fwd[pos]
                if np.isnan(fwd_ret): continue
                if row["pred"] == 1:  # LONG
                    pnls.append(fwd_ret - FEE)
                else:  # SHORT
                    pnls.append(-fwd_ret - FEE)
            pnls = np.array(pnls)
            if len(pnls) == 0: continue
            wr = (pnls > 0).mean()
            avg = pnls.mean()
            print(f"    Fixed {exit_bars}bar: WR={wr:.3f} avg_pnl={avg*100:+.4f}% n={len(pnls):,}")

        # B. TP/SL barrier
        for tag, data in barrier_pnl.items():
            pnls = []
            for _, row in sub.iterrows():
                pos = row["pos"]
                if row["pred"] == 1:
                    p = data["long"][pos]
                else:
                    p = data["short"][pos]
                if not np.isnan(p):
                    pnls.append(p)
            pnls = np.array(pnls)
            if len(pnls) == 0: continue
            wr = (pnls > 0).mean()
            avg = pnls.mean()
            print(f"    Barrier {tag}: WR={wr:.3f} avg_pnl={avg*100:+.4f}% n={len(pnls):,}")

        # C. Ideal (look-ahead, for comparison)
        el = edge_long.values
        es = edge_short.values
        ideal_pnls = []
        for _, row in sub.iterrows():
            pos = row["pos"]
            if row["pred"] == 1:
                p = el[pos]
            else:
                p = es[pos]
            if not np.isnan(p):
                ideal_pnls.append(p)
        ideal_pnls = np.array(ideal_pnls)
        if len(ideal_pnls) > 0:
            print(f"    [Ideal/lookahead]: WR={(ideal_pnls>0).mean():.3f} avg_pnl={ideal_pnls.mean()*100:+.4f}%")

        print()

    # ── Best realistic method: equity curve ──
    print(f"{'='*80}")
    print("=== EQUITY CURVE (Fixed 6-bar exit, conf≥0.0) ===\n")

    fwd6 = fwd_returns[6]
    trades = []
    for _, row in oos_df.iterrows():
        pos = row["pos"]
        fwd_ret = fwd6[pos]
        if np.isnan(fwd_ret): continue
        if row["pred"] == 1:
            pnl = fwd_ret - FEE
        else:
            pnl = -fwd_ret - FEE
        trades.append({"ts": kline.index[pos], "pnl": pnl, "dir": "long" if row["pred"]==1 else "short"})

    trades_df = pd.DataFrame(trades).sort_values("ts").reset_index(drop=True)
    INITIAL = 10000; POS_SIZE = 1000

    equity = [INITIAL]
    for _, t in trades_df.iterrows():
        equity.append(equity[-1] + POS_SIZE * t["pnl"])
    final = equity[-1]
    total_return = (final - INITIAL) / INITIAL * 100

    peak = INITIAL; max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > max_dd: max_dd = dd

    wr = (trades_df["pnl"] > 0).mean()
    avg_pnl = trades_df["pnl"].mean()
    days = (trades_df["ts"].max() - trades_df["ts"].min()).days

    print(f"  ${INITIAL:,} → ${final:,.0f} ({total_return:+.1f}%)")
    print(f"  Trades: {len(trades_df):,} ({len(trades_df)/max(days,1):.1f}/day)")
    print(f"  WR: {wr:.1%}, Avg P&L: {avg_pnl*100:+.4f}%")
    print(f"  Max DD: {max_dd:.1%}")
    if days > 0:
        print(f"  Ann. return: {total_return*365/days:+.1f}% ({days} days)")

    for d in ["long", "short"]:
        s = trades_df[trades_df["dir"] == d]
        if len(s) > 0:
            print(f"  {d.upper()}: n={len(s):,} WR={(s['pnl']>0).mean():.1%} avg={s['pnl'].mean()*100:+.4f}%")

    # Quarterly
    trades_df["quarter"] = trades_df["ts"].dt.to_period("Q")
    quarterly = trades_df.groupby("quarter").agg(
        n=("pnl", "count"), wr=("pnl", lambda x: (x>0).mean()),
        avg=("pnl", "mean"), total=("pnl", "sum"))
    print(f"\n  {'Q':8s} {'N':>6s} {'WR':>6s} {'Avg':>9s} {'Total$':>9s}")
    pos_q = 0
    for p, r in quarterly.iterrows():
        q_pnl = r["total"] * POS_SIZE
        if q_pnl > 0: pos_q += 1
        m = "✓" if q_pnl > 0 else "✗"
        print(f"  {str(p):8s} {int(r['n']):6d} {r['wr']:.1%} {r['avg']*100:+.4f}% {q_pnl:+9,.0f} {m}")
    print(f"  Positive Q: {pos_q}/{len(quarterly)}")

    # ── Also show TP/SL 2.0/1.0 equity ──
    print(f"\n{'='*80}")
    print("=== EQUITY CURVE (TP2.0/SL1.0 barrier, conf≥0.0) ===\n")
    bp = barrier_pnl["tp2.0_sl1.0_h6"]
    trades2 = []
    for _, row in oos_df.iterrows():
        pos = row["pos"]
        if row["pred"] == 1:
            p = bp["long"][pos]
        else:
            p = bp["short"][pos]
        if not np.isnan(p):
            trades2.append({"ts": kline.index[pos], "pnl": p, "dir": "long" if row["pred"]==1 else "short"})

    trades2_df = pd.DataFrame(trades2).sort_values("ts").reset_index(drop=True)
    equity2 = [INITIAL]
    for _, t in trades2_df.iterrows():
        equity2.append(equity2[-1] + POS_SIZE * t["pnl"])
    final2 = equity2[-1]
    ret2 = (final2 - INITIAL) / INITIAL * 100
    pk2 = INITIAL; dd2 = 0
    for e in equity2:
        if e > pk2: pk2 = e
        d = (pk2 - e) / pk2
        if d > dd2: dd2 = d
    wr2 = (trades2_df["pnl"] > 0).mean()
    days2 = (trades2_df["ts"].max() - trades2_df["ts"].min()).days

    print(f"  ${INITIAL:,} → ${final2:,.0f} ({ret2:+.1f}%)")
    print(f"  Trades: {len(trades2_df):,} ({len(trades2_df)/max(days2,1):.1f}/day)")
    print(f"  WR: {wr2:.1%}, Avg P&L: {trades2_df['pnl'].mean()*100:+.4f}%")
    print(f"  Max DD: {dd2:.1%}")
    if days2 > 0:
        print(f"  Ann. return: {ret2*365/days2:+.1f}% ({days2} days)")

    trades2_df["quarter"] = trades2_df["ts"].dt.to_period("Q")
    q2 = trades2_df.groupby("quarter").agg(
        n=("pnl","count"), wr=("pnl", lambda x:(x>0).mean()),
        avg=("pnl","mean"), total=("pnl","sum"))
    print(f"\n  {'Q':8s} {'N':>6s} {'WR':>6s} {'Avg':>9s} {'Total$':>9s}")
    pq2 = 0
    for p, r in q2.iterrows():
        qp = r["total"] * POS_SIZE
        if qp > 0: pq2 += 1
        m = "✓" if qp > 0 else "✗"
        print(f"  {str(p):8s} {int(r['n']):6d} {r['wr']:.1%} {r['avg']*100:+.4f}% {qp:+9,.0f} {m}")
    print(f"  Positive Q: {pq2}/{len(q2)}")

print(f"\n{'='*80}")
print("=== DONE ===")
