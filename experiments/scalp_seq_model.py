"""시퀀스 모델: 1D-CNN over last 20 bars → TP hit prediction.

MLP는 현재 바 snapshot만 봄. 스캘핑 패턴은 시계열:
- "거래량 터진 후 2바 조용 → 브레이크아웃"
- "depth wall 형성 후 점진적 접근 → 반등"
- "vol squeeze → 급격 expansion"

1D-CNN이 이런 시퀀스 패턴을 잡을 수 있는지 검증.
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
FEE_PCT = 0.02; TP_PCT = 0.10; SL_PCT = 0.05; MAX_HOLD = 6
SEQ_LEN = 20  # 최근 20바 (100분)

print(f"=== Sequence Model: {SYMBOL}, seq_len={SEQ_LEN} ===\n")

# ── Data ──
k5 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_5m.parquet").set_index("timestamp").sort_index()
tick = pd.read_parquet(f"data/merged/{SYMBOL}/tick_bar.parquet").set_index("timestamp").sort_index()
bd = pd.read_parquet(f"data/merged/{SYMBOL}/book_depth.parquet")
metrics = pd.read_parquet(f"data/merged/{SYMBOL}/metrics.parquet").set_index("timestamp").sort_index()
k1 = pd.read_parquet(f"data/merged/{SYMBOL}/kline_1m.parquet").set_index("timestamp").sort_index()

for d in [tick, k1]:
    if d.index.tz is None and k5.index.tz is not None:
        d.index = d.index.tz_localize(k5.index.tz)

cc = k5["close"].values; hh = k5["high"].values; ll = k5["low"].values; vv = k5["volume"].values
n = len(cc)

# Direction
bd["timestamp"] = pd.to_datetime(bd["timestamp"])
if bd["timestamp"].dt.tz is None and k5.index.tz is not None:
    bd["timestamp"] = bd["timestamp"].dt.tz_localize(k5.index.tz)
bd_key = bd[bd["percentage"].isin([-1.0, 1.0])].copy()
bd_key["ts_5m"] = bd_key["timestamp"].dt.floor("5min")
bd_piv = bd_key.pivot_table(index="ts_5m", columns="percentage", values="notional", aggfunc="last")
depth_imb = pd.Series(np.nan, index=k5.index)
if -1.0 in bd_piv.columns and 1.0 in bd_piv.columns:
    depth_imb = ((bd_piv[-1.0] - bd_piv[1.0]) / (bd_piv[-1.0] + bd_piv[1.0] + 1e-10)).reindex(k5.index)

ret3 = (pd.Series(cc, index=k5.index) / pd.Series(cc, index=k5.index).shift(3) - 1).values
depth_dir = np.where(depth_imb > 0, 1, np.where(depth_imb < 0, -1, 0))
mr_dir = -np.sign(ret3)
agree = (depth_dir == mr_dir) & (depth_dir != 0)
direction = np.where(agree, depth_dir, 0).astype(float)

# ── Sequence features (per-bar, to form sliding window) ──
print("[Building sequence features...]")
ret = pd.Series(cc, index=k5.index).pct_change()
sf = pd.DataFrame(index=k5.index)

# Price action (normalized)
sf["ret"] = ret
sf["body"] = (k5["close"] - k5["open"]) / (k5["high"] - k5["low"] + 1e-10)
sf["upper_wick"] = (k5["high"] - np.maximum(k5["open"], k5["close"])) / (k5["high"] - k5["low"] + 1e-10)
sf["lower_wick"] = (np.minimum(k5["open"], k5["close"]) - k5["low"]) / (k5["high"] - k5["low"] + 1e-10)
sf["range_pct"] = (k5["high"] - k5["low"]) / k5["close"]

# Volume (ratio to recent avg for stationarity)
sf["vol_ratio"] = pd.Series(vv, index=k5.index) / pd.Series(vv, index=k5.index).rolling(20).mean()

# Trade activity
tick_5m = tick.resample("5min").agg({"buy_volume":"sum","sell_volume":"sum","cvd_raw" if "cvd_raw" in tick.columns else "buy_volume":"sum","trade_count":"sum"})
if "cvd_raw" not in tick.columns:
    tick["cvd_raw"] = tick["buy_volume"] - tick["sell_volume"]
    tick_5m = tick.resample("5min").agg({"buy_volume":"sum","sell_volume":"sum","cvd_raw":"sum","trade_count":"sum"})
sf["buy_ratio"] = (tick_5m["buy_volume"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"])).reindex(k5.index)
sf["tc_ratio"] = (tick_5m["trade_count"] / tick_5m["trade_count"].rolling(20).mean()).reindex(k5.index)
sf["cvd_norm"] = (tick_5m["cvd_raw"] / (tick_5m["buy_volume"] + tick_5m["sell_volume"] + 1e-10)).reindex(k5.index)

# Depth
sf["depth_imb"] = depth_imb

# Volatility state
sf["vol_accel"] = ret.rolling(5).std() / (ret.rolling(20).std() + 1e-10)

# 1m micro
k1_ret = k1["close"].pct_change()
sf["intra_range"] = (k1_ret.resample("5min").max() - k1_ret.resample("5min").min()).reindex(k5.index)

sf = sf.replace([np.inf, -np.inf], np.nan).fillna(0)
seq_features = sf.columns.tolist()
n_channels = len(seq_features)
print(f"  {n_channels} channels per bar, window={SEQ_LEN}")

# Static features (current bar only, for concat)
stat = pd.DataFrame(index=k5.index)
stat["vol_squeeze"] = ret.rolling(20).std() / (ret.rolling(60).std() + 1e-10)
stat["range_pos_20"] = (k5["close"] - k5["low"].rolling(20).min()) / (k5["high"].rolling(20).max() - k5["low"].rolling(20).min() + 1e-10)
stat["vwap_dist"] = k5["close"] / ((k5["close"]*k5["volume"]).rolling(20).sum() / (k5["volume"].rolling(20).sum()+1e-10)) - 1
stat["hour_sin"] = np.sin(2 * np.pi * k5.index.hour / 24)
stat["hour_cos"] = np.cos(2 * np.pi * k5.index.hour / 24)
if "sum_taker_long_short_vol_ratio" in metrics.columns:
    stat["taker_ls"] = metrics["sum_taker_long_short_vol_ratio"].reindex(k5.index, method="ffill")
if "sum_open_interest_value" in metrics.columns:
    oi = metrics["sum_open_interest_value"].reindex(k5.index, method="ffill")
    stat["oi_chg"] = oi.pct_change(5)
stat = stat.replace([np.inf, -np.inf], np.nan).fillna(0)
n_static = len(stat.columns)

# Vol filter
va = sf["vol_accel"]; tc = sf["tc_ratio"]
va_q80 = va.quantile(0.80); tc_q80 = tc.quantile(0.80)
entry_mask = (direction != 0) & (va > va_q80).values & (tc > tc_q80).values

# TP/SL labels
tp_hit = np.full(n, np.nan)
for i in range(n - MAX_HOLD):
    d = direction[i]
    if d == 0 or np.isnan(d): continue
    entry = cc[i]
    tp = entry * (1 + d * TP_PCT / 100); sl = entry * (1 - d * SL_PCT / 100)
    hit = 0
    for j in range(i+1, i+MAX_HOLD+1):
        if d == 1:
            if hh[j] >= tp: hit = 1; break
            if ll[j] <= sl: hit = -1; break
        else:
            if ll[j] <= tp: hit = 1; break
            if hh[j] >= sl: hit = -1; break
    tp_hit[i] = 1.0 if hit == 1 else 0.0

# ── Dataset ──
sf_vals = sf.values.astype(np.float32)
stat_vals = stat.values.astype(np.float32)

class SeqDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices  # positions in the original array
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        # Sequence: (SEQ_LEN, n_channels) → (n_channels, SEQ_LEN) for Conv1d
        seq = sf_vals[idx-SEQ_LEN:idx]  # (SEQ_LEN, C)
        seq = np.nan_to_num(seq, 0.0)
        static = np.nan_to_num(stat_vals[idx], 0.0)
        y = tp_hit[idx] if not np.isnan(tp_hit[idx]) else 0.0
        return (torch.tensor(seq.T, dtype=torch.float32),  # (C, SEQ_LEN)
                torch.tensor(static, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

# ── Model ──
class SeqTPModel(nn.Module):
    """1D-CNN on sequence + MLP on static → TP hit probability."""
    def __init__(self, n_channels, seq_len, n_static, hidden=128):
        super().__init__()
        # 1D CNN over time
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(64)

        # Combine: CNN output + static features
        self.head = nn.Sequential(
            nn.Linear(32 + n_static, hidden),
            nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.LayerNorm(hidden // 2), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, seq, static):
        # seq: (B, C, T)
        x = self.act(self.norm1(self.conv1(seq)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # (B, 32)
        combined = torch.cat([x, static], dim=-1)
        return self.head(combined).squeeze(-1)

# Also test baseline MLP for fair comparison
class BaselineMLP(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.LayerNorm(hidden//2), nn.Dropout(0.2),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        # Flatten last bar's seq features + static
        x = np.concatenate([sf_vals[idx], stat_vals[idx]])
        x = np.nan_to_num(x, 0.0)
        y = tp_hit[idx] if not np.isnan(tp_hit[idx]) else 0.0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ── Walk-forward ──
print("\n[Walk-forward...]")
splits = []
cursor = pd.Timestamp("2023-01-01", tz=k5.index.tz)
end = k5.index.max()
while cursor + pd.DateOffset(months=8) <= end:
    tr_e = cursor + pd.DateOffset(months=6)
    va_e = tr_e + pd.DateOffset(months=1)
    te_e = va_e + pd.DateOffset(months=1)
    splits.append({"train": (cursor, tr_e), "val": (tr_e, va_e), "test": (va_e, te_e)})
    cursor += pd.DateOffset(months=1)

def train_model(model, train_loader, val_loader, criterion, epochs=40, patience=7, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_v = float("inf"); wait = 0; best_s = None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            if len(batch) == 3:  # seq model
                seq, static, y = [b.to(DEVICE) for b in batch]
                out = model(seq, static)
            else:  # mlp
                x, y = [b.to(DEVICE) for b in batch]
                out = model(x)
            loss = criterion(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); vl = 0; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    seq, static, y = [b.to(DEVICE) for b in batch]
                    out = model(seq, static)
                else:
                    x, y = [b.to(DEVICE) for b in batch]
                    out = model(x)
                vl += criterion(out, y).item(); nb += 1
        vl /= max(nb,1)
        if vl < best_v:
            best_v = vl; wait = 0; best_s = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_s)
    return model, best_v

# Collect OOS predictions for both models
oos_seq = np.full(n, np.nan)
oos_mlp = np.full(n, np.nan)

for i, sp in enumerate(splits):
    tr_s, tr_e = sp["train"]; va_s, va_e = sp["val"]; te_s, te_e = sp["test"]

    def get_positions(start, end):
        mask = (k5.index >= start) & (k5.index < end) & entry_mask
        positions = np.where(mask)[0]
        # Filter: need SEQ_LEN history and valid label
        valid = [p for p in positions if p >= SEQ_LEN and not np.isnan(tp_hit[p])]
        return np.array(valid)

    tr_pos = get_positions(tr_s, tr_e)
    va_pos = get_positions(va_s, va_e)
    te_pos = get_positions(te_s, te_e)

    if len(tr_pos) < 200 or len(va_pos) < 50 or len(te_pos) == 0:
        continue

    pos_rate = np.mean([tp_hit[p] for p in tr_pos])
    pos_weight = torch.tensor([(1-pos_rate)/(pos_rate+1e-10)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Seq model ---
    seq_tr = SeqDataset(tr_pos); seq_va = SeqDataset(va_pos); seq_te = SeqDataset(te_pos)
    seq_model = SeqTPModel(n_channels, SEQ_LEN, n_static, hidden=128)
    seq_model, seq_vl = train_model(
        seq_model,
        DataLoader(seq_tr, batch_size=512, shuffle=True, num_workers=0),
        DataLoader(seq_va, batch_size=1024, shuffle=False, num_workers=0),
        criterion
    )

    # --- MLP baseline ---
    mlp_tr = MLPDataset(tr_pos); mlp_va = MLPDataset(va_pos)
    mlp_model = BaselineMLP(n_channels + n_static, hidden=128)
    mlp_model, mlp_vl = train_model(
        mlp_model,
        DataLoader(mlp_tr, batch_size=512, shuffle=True, num_workers=0),
        DataLoader(mlp_va, batch_size=1024, shuffle=False, num_workers=0),
        criterion
    )

    # --- OOS predictions ---
    seq_model.eval(); mlp_model.eval()
    with torch.no_grad():
        for j, pos in enumerate(te_pos):
            # Seq
            seq_input = torch.tensor(sf_vals[pos-SEQ_LEN:pos].T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            stat_input = torch.tensor(np.nan_to_num(stat_vals[pos], 0.0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            oos_seq[pos] = torch.sigmoid(seq_model(seq_input, stat_input)).cpu().item()

            # MLP
            x_mlp = torch.tensor(np.nan_to_num(np.concatenate([sf_vals[pos], stat_vals[pos]]), 0.0),
                                  dtype=torch.float32).unsqueeze(0).to(DEVICE)
            oos_mlp[pos] = torch.sigmoid(mlp_model(x_mlp)).cpu().item()

    # Window summary
    seq_probs = [oos_seq[p] for p in te_pos if not np.isnan(oos_seq[p])]
    mlp_probs = [oos_mlp[p] for p in te_pos if not np.isnan(oos_mlp[p])]
    print(f"  W{i+1:02d} {te_s.date()}~{te_e.date()} n={len(te_pos)} "
          f"seq_vl={seq_vl:.4f} mlp_vl={mlp_vl:.4f}")

# ── Compare ──
print(f"\n{'='*80}")
print("=== COMPARISON: Seq CNN vs MLP ===\n")

start_idx = k5.index.get_loc(k5.loc["2023-01-01":].index[0])

for model_name, probs_arr in [("MLP", oos_mlp), ("SeqCNN", oos_seq)]:
    print(f"\n  --- {model_name} ---")
    for prob_thr in [0.0, 0.50, 0.55, 0.60]:
        # Sequential sim
        equity = 10000; POS_SIZE = 1000
        trades = []; in_pos = False
        pos_bar = 0; pos_dir = 0; pos_entry = 0; pos_tp = 0; pos_sl = 0

        for idx in range(max(start_idx, SEQ_LEN), n):
            if in_pos:
                bars = idx - pos_bar
                closed = False
                if pos_dir == 1:
                    if hh[idx] >= pos_tp: pnl = TP_PCT/100 - FEE_PCT/100; closed = True
                    elif ll[idx] <= pos_sl: pnl = -SL_PCT/100 - FEE_PCT/100; closed = True
                    elif bars >= MAX_HOLD: pnl = (cc[idx]-pos_entry)/pos_entry - FEE_PCT/100; closed = True
                else:
                    if ll[idx] <= pos_tp: pnl = TP_PCT/100 - FEE_PCT/100; closed = True
                    elif hh[idx] >= pos_sl: pnl = -SL_PCT/100 - FEE_PCT/100; closed = True
                    elif bars >= MAX_HOLD: pnl = (pos_entry-cc[idx])/pos_entry - FEE_PCT/100; closed = True
                if closed:
                    equity += POS_SIZE * pnl
                    trades.append(pnl)
                    in_pos = False

            if not in_pos and entry_mask[idx] and direction[idx] != 0:
                prob = probs_arr[idx]
                if np.isnan(prob): continue
                if prob < prob_thr: continue
                pos_bar = idx; pos_dir = int(direction[idx]); pos_entry = cc[idx]
                pos_tp = pos_entry * (1 + pos_dir * TP_PCT / 100)
                pos_sl = pos_entry * (1 - pos_dir * SL_PCT / 100)
                in_pos = True

        if not trades:
            print(f"    prob≥{prob_thr:.2f}: No trades")
            continue
        trades = np.array(trades)
        wr = (trades > 0).mean()
        avg = trades.mean()
        ret_pct = (equity - 10000) / 10000 * 100
        pk = 10000; dd = 0; eq = 10000
        for t in trades:
            eq += POS_SIZE * t
            if eq > pk: pk = eq
            d = (pk - eq) / pk
            if d > dd: dd = d

        marker = " <<<" if avg > 0.0003 else ""
        print(f"    prob≥{prob_thr:.2f}: n={len(trades):,} WR={wr:.1%} avg={avg*100:+.4f}% "
              f"ret={ret_pct:+.1f}% DD={dd:.1%}{marker}")

print(f"\n{'='*80}")
print("=== DONE ===")
