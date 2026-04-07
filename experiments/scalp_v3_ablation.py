"""v3 개선 탐색 — 5가지 축 ablation.

Baseline: ScalpMLP3(128), CE loss, 28 features, hold=6, no confidence filter
Ablations:
  A. 모델 크기: 64 / 128 / 256 / 3-layer
  B. Loss: CE / Focal / Edge-weighted CE
  C. Confidence filter: trade only when max_prob > threshold
  D. min_rr: 1.5 / 2.0 / 2.5
  E. Feature selection: top-15 by importance vs all 28

단일 train/val/test split으로 빠르게 비교 (2023H1 train, 2023Q3 val, 2023Q4~2024Q1 test)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scalping.labeler_v3 import generate_scalp_labels_v3

SYMBOL = "ETHUSDT"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FEE = 0.0008

print(f"=== v3 Ablation Study: {SYMBOL} ===\n")

# ── Data (reuse baseline feature pipeline) ──
kline = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
funding = pd.read_parquet(f"data/merged/{SYMBOL}/funding_rate.parquet").set_index("timestamp").sort_index()

if tick.index.tz is None and kline.index.tz is not None:
    tick.index = tick.index.tz_localize(kline.index.tz)

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

feature_names = feat.columns.tolist()
print(f"Features: {len(feature_names)}")

# ── Dataset ──
class DS(Dataset):
    def __init__(self, X, y, edge):
        self.X = torch.tensor(np.nan_to_num(np.array(X, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
        y_arr = np.nan_to_num(np.array(y, dtype=np.float64, copy=True), nan=1.0)
        self.y = torch.tensor(y_arr + 1, dtype=torch.long)
        self.edge = torch.tensor(np.nan_to_num(np.array(edge, dtype=np.float64, copy=True), 0.0), dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.edge[i]

# ── Models ──
class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_layers=2, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden // 2), nn.GELU(), nn.LayerNorm(hidden // 2), nn.Dropout(dropout)]
            hidden = hidden // 2
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ── Focal Loss ──
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, inp, target):
        ce = F.cross_entropy(inp, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ── Train function ──
def train_eval(model, train_ds, val_ds, test_X, test_y, test_el, test_es,
               loss_fn, epochs=50, patience=7, lr=1e-3, conf_thresholds=None):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)

    best_val = float("inf"); wait = 0; best_state = None
    for epoch in range(epochs):
        model.train()
        for X, y, e in train_loader:
            X, y, e = X.to(DEVICE), y.to(DEVICE), e.to(DEVICE)
            out = model(X)
            if isinstance(loss_fn, str) and loss_fn == "edge_weighted":
                ce = F.cross_entropy(out, y, reduction='none')
                w = 1.0 + e * 100  # edge가 클수록 가중치 ↑
                loss = (ce * w).mean()
            else:
                loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        vl = 0; nb = 0
        with torch.no_grad():
            for X, y, e in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                vl += F.cross_entropy(model(X), y).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best_val:
            best_val = vl; wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(np.nan_to_num(np.array(test_X, dtype=np.float64, copy=True), 0.0), dtype=torch.float32).to(DEVICE)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    pred_action = preds.astype(float) - 1
    actual = np.nan_to_num(np.array(test_y, dtype=np.float64, copy=True), nan=0) + 1

    results = {}
    # Default (no confidence filter)
    thresholds = conf_thresholds or [0.0, 0.4, 0.5, 0.6, 0.7]
    for thr in thresholds:
        max_prob = probs.max(axis=1)
        conf_mask = max_prob >= thr if thr > 0 else np.ones(len(preds), dtype=bool)

        pa = pred_action[conf_mask]
        el = np.array(test_el, dtype=np.float64, copy=True)[conf_mask]
        es = np.array(test_es, dtype=np.float64, copy=True)[conf_mask]

        long_m = pa == 1; short_m = pa == -1
        n_trades = long_m.sum() + short_m.sum()
        n_total = len(pa)

        if n_trades == 0:
            results[thr] = {"thr": thr, "n": 0, "wr": 0, "pnl": 0, "rate": 0}
            continue

        pnl_arr = np.concatenate([el[long_m], es[short_m]])
        wr = (pnl_arr > 0).mean()
        avg_pnl = np.nanmean(pnl_arr)
        trade_rate = n_trades / n_total

        results[thr] = {"thr": thr, "n": int(n_trades), "wr": round(wr, 4),
                         "pnl": round(avg_pnl * 100, 4), "rate": round(trade_rate, 3)}

    return results, best_val

# ── Generate labels for different min_rr ──
print("[Generating labels...]")
labels_rr = {}
for rr in [1.5, 2.0, 2.5]:
    labels_rr[rr] = generate_scalp_labels_v3(kline, max_holds=[6], fee=FEE, min_rr=rr)

# ── Splits: longer test period for stability ──
# Train: 2023-01 ~ 2024-06 (18mo), Val: 2024-07 ~ 2024-09 (3mo), Test: 2024-10 ~ 2026-04 (18mo OOS)
tr_s = pd.Timestamp("2023-01-01", tz=kline.index.tz)
tr_e = pd.Timestamp("2024-07-01", tz=kline.index.tz)
va_e = pd.Timestamp("2024-10-01", tz=kline.index.tz)
te_e = kline.index.max()

tr_mask = (feat.index >= tr_s) & (feat.index < tr_e)
va_mask = (feat.index >= tr_e) & (feat.index < va_e)
te_mask = (feat.index >= va_e) & (feat.index <= te_e)

print(f"Train: {tr_mask.sum():,}, Val: {va_mask.sum():,}, Test: {te_mask.sum():,}")

# ── Run ablations ──
def run_ablation(name, model_cls, model_kwargs, loss_fn, rr=1.5, feat_cols=None):
    labs = labels_rr[rr]
    action = labs["action_6"]
    el = labs["edge_long_6"]
    es = labs["edge_short_6"]

    if feat_cols is None:
        X = feat.values
        n_feat = feat.shape[1]
    else:
        X = feat[feat_cols].values
        n_feat = len(feat_cols)

    edge = np.maximum(
        np.nan_to_num(np.array(el.values, dtype=np.float64, copy=True), 0),
        np.nan_to_num(np.array(es.values, dtype=np.float64, copy=True), 0)
    )

    train_ds = DS(X[tr_mask], action.values[tr_mask], edge[tr_mask])
    val_ds = DS(X[va_mask], action.values[va_mask], edge[va_mask])

    # Class weights
    y_t = train_ds.y.numpy()
    counts = np.bincount(y_t, minlength=3).astype(float)
    counts[counts == 0] = 1
    w = 1.0 / counts; w = w / w.sum() * 3
    cw = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    if loss_fn == "ce":
        lf = nn.CrossEntropyLoss(weight=cw)
    elif loss_fn == "focal":
        lf = FocalLoss(weight=cw, gamma=2.0)
    elif loss_fn == "edge_weighted":
        lf = "edge_weighted"
    else:
        lf = nn.CrossEntropyLoss(weight=cw)

    model = model_cls(n_feat, **model_kwargs)

    results, val_loss = train_eval(
        model, train_ds, val_ds,
        X[te_mask], action.values[te_mask], el.values[te_mask], es.values[te_mask],
        lf
    )

    print(f"\n  [{name}] val_loss={val_loss:.4f}")
    for thr, r in results.items():
        if r["n"] == 0:
            continue
        marker = " <<<" if r["pnl"] > 0.12 else " **" if r["pnl"] > 0.08 else ""
        print(f"    conf≥{thr:.1f}: trades={r['n']:,} rate={r['rate']:.1%} WR={r['wr']:.3f} pnl={r['pnl']:+.4f}%{marker}")
    return results

print(f"\n{'='*80}")
print("=== A. Model Size ===")
run_ablation("MLP-64", MLP, {"hidden": 64, "n_layers": 2}, "ce")
run_ablation("MLP-128 (baseline)", MLP, {"hidden": 128, "n_layers": 2}, "ce")
run_ablation("MLP-256", MLP, {"hidden": 256, "n_layers": 2}, "ce")
run_ablation("MLP-128x3L", MLP, {"hidden": 128, "n_layers": 3}, "ce")
run_ablation("MLP-256x3L", MLP, {"hidden": 256, "n_layers": 3}, "ce")

print(f"\n{'='*80}")
print("=== B. Loss Function ===")
run_ablation("CE (baseline)", MLP, {"hidden": 128, "n_layers": 2}, "ce")
run_ablation("Focal (γ=2)", MLP, {"hidden": 128, "n_layers": 2}, "focal")
run_ablation("Edge-weighted CE", MLP, {"hidden": 128, "n_layers": 2}, "edge_weighted")

print(f"\n{'='*80}")
print("=== C. min_rr ===")
run_ablation("rr=1.5 (baseline)", MLP, {"hidden": 128, "n_layers": 2}, "ce", rr=1.5)
run_ablation("rr=2.0", MLP, {"hidden": 128, "n_layers": 2}, "ce", rr=2.0)
run_ablation("rr=2.5", MLP, {"hidden": 128, "n_layers": 2}, "ce", rr=2.5)

print(f"\n{'='*80}")
print("=== D. Best combo candidates ===")
# Combine best from each axis
run_ablation("MLP-256 + Focal + rr=2.0", MLP, {"hidden": 256, "n_layers": 2}, "focal", rr=2.0)
run_ablation("MLP-256x3L + EdgeW + rr=2.0", MLP, {"hidden": 256, "n_layers": 3}, "edge_weighted", rr=2.0)
run_ablation("MLP-128x3L + Focal + rr=1.5", MLP, {"hidden": 128, "n_layers": 3}, "focal", rr=1.5)

print(f"\n{'='*80}")
print("=== DONE ===")
