"""v3 라벨 + ScalpingMLP baseline — ETH 5m, walk-forward."""

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

print(f"=== v3 Baseline: {SYMBOL}, hold={HOLD}, device={DEVICE} ===\n")

# ── 1. Data + Labels ──
kline = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()

if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)

print("[Labels...]")
labels = generate_scalp_labels_v3(kline, max_holds=[HOLD], fee=FEE, min_rr=MIN_RR)

# ── 2. Features (탐색에서 유효했던 것들) ──
print("[Features...]")
c = kline["close"]; o = kline["open"]; h = kline["high"]; l = kline["low"]; v = kline["volume"]
ret = c.pct_change()

feat = pd.DataFrame(index=kline.index)

# Price structure
feat["ret_1"] = ret
feat["ret_3"] = c / c.shift(3) - 1
feat["ret_6"] = c / c.shift(6) - 1
feat["body_ratio"] = (c - o) / (h - l + 1e-10)
feat["body_ratio_3avg"] = feat["body_ratio"].rolling(3).mean()
feat["upper_wick"] = (h - np.maximum(o, c)) / (h - l + 1e-10)
feat["lower_wick"] = (np.minimum(o, c) - l) / (h - l + 1e-10)

# Volatility
feat["vol_20"] = ret.rolling(20).std()
feat["vol_60"] = ret.rolling(60).std()
feat["vol_squeeze"] = feat["vol_20"] / feat["vol_60"]
feat["atr_12"] = (h - l).rolling(12).mean()
feat["atr_ratio"] = (h - l) / feat["atr_12"]

# Volume
feat["vol_ratio"] = v / v.rolling(12).mean()

# Position
ma20 = c.rolling(20).mean()
ma50 = c.rolling(50).mean()
feat["ma_20_slope"] = (ma20 - ma20.shift(3)) / (ma20.shift(3) + 1e-10)
feat["ma_50_slope"] = (ma50 - ma50.shift(5)) / (ma50.shift(5) + 1e-10)
feat["range_pos_20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-10)
vwap = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
feat["vwap_dist"] = c / vwap - 1

# Tick bar CVD
tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
tick_5m = tick.resample("5min").agg({"buy_volume": "sum", "sell_volume": "sum", "cvd_raw": "sum", "trade_count": "sum"})
feat = feat.join(tick_5m[["cvd_raw", "trade_count"]], how="left")
feat["cvd_cumsum"] = feat["cvd_raw"].cumsum()
feat["cvd_slope_5"] = feat["cvd_cumsum"] - feat["cvd_cumsum"].shift(5)
feat["buy_ratio"] = tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"])
feat["taker_buy_pct"] = kline["taker_buy_volume"] / kline["volume"]

# Book depth
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

# Metrics
for col in ["sum_open_interest_value", "sum_taker_long_short_vol_ratio"]:
    if col in metrics.columns:
        feat[col] = metrics[col].reindex(kline.index, method="ffill")
if "sum_open_interest_value" in feat.columns:
    feat["oi_change_5"] = feat["sum_open_interest_value"].pct_change(5)
    feat.drop(columns=["sum_open_interest_value"], inplace=True)

# Funding
feat["funding_rate"] = funding["funding_rate"].reindex(kline.index, method="ffill") if "funding_rate" in funding.columns else np.nan

# Derived (탐색에서 유효했던 것)
feat["hammer_score"] = feat["lower_wick"].rolling(3).max() * (1 - feat["range_pos_20"])
feat["shooting_star"] = feat["upper_wick"].rolling(3).max() * feat["range_pos_20"]

# Drop raw intermediates
feat.drop(columns=["cvd_raw", "cvd_cumsum"], inplace=True, errors="ignore")

# Clean
feat = feat.replace([np.inf, -np.inf], np.nan)
print(f"  Features: {feat.shape[1]}")
print(f"  NaN rate: {feat.isna().mean().mean():.1%}")

# ── 3. Align ──
common = feat.index.intersection(labels.index)
feat = feat.loc[common]
labels = labels.loc[common]
action = labels[f"action_{HOLD}"]
edge_long = labels[f"edge_long_{HOLD}"]
edge_short = labels[f"edge_short_{HOLD}"]

# ── 4. Dataset ──
class ScalpV3Dataset(Dataset):
    def __init__(self, X, y, edge):
        self.X = torch.tensor(np.nan_to_num(np.array(X, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
        # y: -1,0,+1 → 0,1,2 for CrossEntropy
        y_arr = np.array(y, dtype=np.float64, copy=True)
        y_arr = np.nan_to_num(y_arr, nan=1.0)  # NaN → HOLD(1)
        self.y = torch.tensor(y_arr + 1, dtype=torch.long)  # -1→0, 0→1, +1→2
        self.edge = torch.tensor(np.nan_to_num(np.array(edge, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.edge[idx]

# ── 5. Model ──
class ScalpMLP3(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),  # short / hold / long
        )

    def forward(self, x):
        return self.net(x)

# ── 6. Walk-forward ──
print(f"\n[Walk-forward...]")

TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 1

def walk_forward_splits(index, start_year=2023):
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

splits = walk_forward_splits(feat.index)
print(f"  {len(splits)} windows (2023+)")

n_feat = feat.shape[1]
all_oos = []

for i, sp in enumerate(splits):
    tr_s, tr_e = sp["train"]
    va_s, va_e = sp["val"]
    te_s, te_e = sp["test"]

    tr_mask = (feat.index >= tr_s) & (feat.index < tr_e)
    va_mask = (feat.index >= va_s) & (feat.index < va_e)
    te_mask = (feat.index >= te_s) & (feat.index < te_e)

    # Best edge per bar for weighting
    edge = np.maximum(
        np.nan_to_num(np.array(edge_long.values, dtype=np.float64, copy=True), 0),
        np.nan_to_num(np.array(edge_short.values, dtype=np.float64, copy=True), 0)
    )

    train_ds = ScalpV3Dataset(feat.values[tr_mask], action.values[tr_mask], edge[tr_mask])
    val_ds = ScalpV3Dataset(feat.values[va_mask], action.values[va_mask], edge[va_mask])

    if len(train_ds) < 1000 or len(val_ds) < 100:
        continue

    print(f"\n  --- Window {i+1}/{len(splits)} | Test: {te_s.date()}~{te_e.date()} ---")
    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # Class weights (handle imbalance — HOLD is minority)
    y_train = train_ds.y.numpy()
    counts = np.bincount(y_train, minlength=3).astype(float)
    counts[counts == 0] = 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    model = ScalpMLP3(n_feat, hidden=128, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

    # Training
    best_val = float("inf")
    patience = 7
    wait = 0
    best_state = None

    for epoch in range(50):
        model.train()
        train_loss = 0; n_batch = 0
        for X, y, e in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        scheduler.step()

        model.eval()
        val_loss = 0; n_vb = 0
        with torch.no_grad():
            for X, y, e in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(X), y).item()
                n_vb += 1

        vl = val_loss / max(n_vb, 1)
        if vl < best_val:
            best_val = vl
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val:.4f} (epoch {epoch+1-patience if wait>=patience else epoch+1})")

    # ── OOS Evaluation ──
    model.eval()
    X_test = feat.values[te_mask]
    y_test = action.values[te_mask]
    el_test = edge_long.values[te_mask]
    es_test = edge_short.values[te_mask]

    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(np.array(X_test, dtype=np.float64, copy=True), 0.0), dtype=torch.float32).to(DEVICE)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()  # 0=short, 1=hold, 2=long

    # Map back: 0→-1, 1→0, 2→+1
    pred_action = preds.astype(float) - 1

    # Accuracy
    y_mapped = np.nan_to_num(y_test, nan=0) + 1  # to 0,1,2
    acc = (preds == y_mapped).mean()

    # Per-class accuracy
    for cls, cls_name in [(0, "SHORT"), (1, "HOLD"), (2, "LONG")]:
        mask = y_mapped == cls
        if mask.sum() > 0:
            cls_acc = (preds[mask] == cls).mean()
        else:
            cls_acc = 0

    # Trading simulation
    n_long = (pred_action == 1).sum()
    n_short = (pred_action == -1).sum()
    n_hold = (pred_action == 0).sum()
    n_total = len(pred_action)

    # When model says LONG, what's the actual edge?
    long_trades = pred_action == 1
    short_trades = pred_action == -1
    hold_pred = pred_action == 0

    # Actual optimal action alignment
    actual = np.nan_to_num(y_test, nan=0)
    correct_long = (long_trades & (actual == 1)).sum()
    correct_short = (short_trades & (actual == -1)).sum()
    correct_hold = (hold_pred & (actual == 0)).sum()
    wrong_long = (long_trades & (actual != 1)).sum()
    wrong_short = (short_trades & (actual != -1)).sum()

    # Edge-weighted P&L
    # When pred=long: use edge_long (positive if actual was long-favorable)
    pnl_long = el_test[long_trades]
    pnl_short = es_test[short_trades]

    avg_pnl_long = np.nanmean(pnl_long) if len(pnl_long) > 0 else 0
    avg_pnl_short = np.nanmean(pnl_short) if len(pnl_short) > 0 else 0
    total_trades = n_long + n_short
    trade_rate = total_trades / n_total

    # Overall average P&L per trade
    all_pnl = np.concatenate([pnl_long, pnl_short]) if total_trades > 0 else np.array([0])
    avg_pnl = np.nanmean(all_pnl)

    # Win rate (edge > 0)
    wr = (all_pnl > 0).mean() if len(all_pnl) > 0 else 0

    result = {
        "window": i + 1,
        "test": f"{te_s.date()}~{te_e.date()}",
        "acc": acc,
        "trade_rate": trade_rate,
        "n_long": n_long,
        "n_short": n_short,
        "n_hold": n_hold,
        "avg_pnl": avg_pnl,
        "avg_pnl_long": avg_pnl_long,
        "avg_pnl_short": avg_pnl_short,
        "wr": wr,
    }
    all_oos.append(result)

    marker = "✓" if avg_pnl > 0 else "✗"
    print(f"  Acc={acc:.3f} | trades={total_trades} ({trade_rate:.1%}) | "
          f"WR={wr:.3f} | avg_pnl={avg_pnl*100:+.4f}% | "
          f"L={avg_pnl_long*100:+.4f}% S={avg_pnl_short*100:+.4f}% {marker}")

# ── 7. Summary ──
print(f"\n{'='*80}")
print("=== SUMMARY ===\n")

if not all_oos:
    print("No results.")
else:
    df = pd.DataFrame(all_oos)
    print(f"Windows: {len(df)}")
    print(f"Avg accuracy: {df['acc'].mean():.3f}")
    print(f"Avg trade rate: {df['trade_rate'].mean():.1%}")
    print(f"Avg WR: {df['wr'].mean():.3f}")
    print(f"Avg P&L per trade: {df['avg_pnl'].mean()*100:+.4f}%")
    print(f"  Long:  {df['avg_pnl_long'].mean()*100:+.4f}%")
    print(f"  Short: {df['avg_pnl_short'].mean()*100:+.4f}%")

    positive = (df["avg_pnl"] > 0).sum()
    print(f"\nPositive windows: {positive}/{len(df)}")

    # Fee breakeven check
    print(f"\n핵심: avg_pnl > 0 이면 fee 차감 후에도 수익")
    print(f"  (v3 라벨의 edge는 이미 fee가 차감된 값)")
