#!/usr/bin/env python3
"""
Iteration 046: Grid Entry + Trailing Take-Profit Evaluation

Measure combined improvement from:
  1. Grid entry (3 tranches) vs immediate entry
  2. Trailing TP vs fixed exit
  3. Combined vs baseline
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, _kl_binary
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2, compute_atr
from execution.grid_entry import simulate_grid_entry, simulate_trailing_tp


def _strip_tz(df):
    if df.index.tz is not None:
        df = df.copy(); df.index = df.index.tz_localize(None)
    return df


def load_data_full(data_dir="data/merged/BTCUSDT", start="2020-06-01", end="2026-02-28"):
    kline = {}
    for tf in ["5m", "15m"]:
        path = os.path.join(data_dir, f"kline_{tf}.parquet")
        if os.path.exists(path):
            kline[tf] = _strip_tz(pd.read_parquet(path).set_index("timestamp").sort_index())[start:end]
    if "15m" in kline:
        kline["1h"] = kline["15m"].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    path_1m = os.path.join(data_dir, "kline_1m.parquet")
    if os.path.exists(path_1m):
        kline["1m"] = _strip_tz(pd.read_parquet(path_1m).set_index("timestamp").sort_index())[start:end]
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = _strip_tz(pd.read_parquet(path).set_index("timestamp").sort_index())[start:end]
    return kline, extras


def train_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048, lr=5e-4, device="cuda", patience=7):
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    opt = torch.optim.AdamW(list(model.parameters())+list(loss_fn.parameters()), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*(len(train_ds)//batch_size+1))
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    bv, ni, bs = float("inf"), 0, None
    for ep in range(epochs):
        model.train()
        for b in tl:
            b = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
            if np.random.random()<0.5:
                lam=np.random.beta(0.2,0.2);idx=torch.randperm(b["features"].size(0),device=device)
                for k in b:
                    if isinstance(b[k],torch.Tensor) and b[k].dtype==torch.float32: b[k]=lam*b[k]+(1-lam)*b[k][idx]
            o1=model(b["features"],b["account"]);o2=model(b["features"],b["account"])
            l1,l2=loss_fn(o1,b),loss_fn(o2,b)
            t=(l1["total"]+l2["total"])/2+1.0*_kl_binary(o1["label_probs"],o2["label_probs"],b["rar_mask"])
            opt.zero_grad();t.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step()
        model.eval()
        with torch.no_grad():
            vm=[loss_fn(model(bb["features"],bb["account"]),bb)["total"].item() for bb in [{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()} for b in vl]]
        v=np.mean(vm)
        if v<bv: bv,ni=v,0;bs={k:val.cpu().clone() for k,val in model.state_dict().items()}
        else:
            ni+=1
            if ni>=patience: break
    if bs: model.load_state_dict(bs)
    return model


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 046: Grid Entry + Trailing TP", flush=True)
    print("=" * 60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    print("\n[1/5] Loading data...", flush=True)
    kline, extras = load_data_full(start=START, end=END)

    print("[2/5] Features + labels...", flush=True)
    kline_feat = {k: v for k, v in kline.items() if k != "1m"}
    features = generate_features_v2(kline_data=kline_feat, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"), target_tf="15min", progress=False)
    tf = features.std().sort_values(ascending=False).head(30).index.tolist()
    sc = {}
    for lag in range(1, 8):
        for col in tf: sc[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in tf[:10]:
        for lag in [1, 2, 4]: sc[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)
    features = pd.concat([features, pd.DataFrame(sc, index=features.index)], axis=1)
    features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)

    lr_data = generate_multi_tbm_v2(kline_feat, fee_pct=0.0008, progress=False)
    labels = lr_data["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr_data:
            sw = lr_data[name].resample("15min").ffill() if name != "intraday" else lr_data[name]
            for col in sw.columns:
                if col not in labels.columns: labels[col] = sw[col]
    common = features.index.intersection(labels.index)
    X, L = features.loc[common], labels.loc[common]
    print(f"  Data: {X.shape}", flush=True)

    tc = sorted([c for c in L.columns if c.startswith("tbm_")])
    mc = sorted([c for c in L.columns if c.startswith("mae_")])
    fc = sorted([c for c in L.columns if c.startswith("mfe_")])
    rc = sorted([c for c in L.columns if c.startswith("rar_")])
    wc = sorted([c for c in L.columns if c.startswith("wgt_")])
    si = [{"style":c.replace("tbm_","").split("_")[0],"dir":c.replace("tbm_","").split("_")[1]} for c in tc]
    pt = {k:v for k,v in partition_features(list(X.columns)).items() if len(v)>0}
    n, ns = len(X), len(tc)
    ws = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute 15m ATR for trailing TP
    c15 = kline["15m"]["close"].values.astype(np.float64)
    h15 = kline["15m"]["high"].values.astype(np.float64)
    l15 = kline["15m"]["low"].values.astype(np.float64)
    atr_15m = compute_atr(h15, l15, c15, period=14)
    atr_series = pd.Series(atr_15m, index=kline["15m"].index)

    print("\n[3/5] Training model (window 3)...", flush=True)
    w = 2
    ts = w*(ws//3); te = ts+int(ws*2); ve = te+int(ws*0.5); test_e = min(ve+ws, n)
    torch.manual_seed(42); np.random.seed(42)
    Xn = X.values.astype(np.float32)
    ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
    wn = L[wc].values if wc else None
    tds = TradingDatasetV4(Xn[ts:te],L[tc].values[ts:te],L[mc].values[ts:te],L[fc].values[ts:te],L[rc].values[ts:te],ac[ts:te],wn[ts:te] if wn is not None else None)
    vds = TradingDatasetV4(Xn[te:ve],L[tc].values[te:ve],L[mc].values[te:ve],L[fc].values[te:ve],L[rc].values[te:ve],ac[te:ve],wn[te:ve] if wn is not None else None)
    model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                  expert_hidden=128, expert_output=96, fusion_dim=192, dropout=0.2)
    model = train_rdrop(model, tds, vds, epochs=50, device=device, patience=7)

    print("\n[4/5] Generating signals...", flush=True)
    X_test = X.iloc[ve:test_e]
    nt = len(X_test)
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device), torch.zeros(nt,4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p, mae_p = out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()

    tc_p = kline["15m"]["close"].reindex(X_test.index, method="ffill").values
    sv = kline["4h"]["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values
    is_long = np.array([s["dir"]=="long" for s in si])
    holds = np.array([{"scalp":1,"intraday":4,"daytrade":48,"swing":168}.get(s["style"],12) for s in si])
    dirs = np.where(is_long, 1.0, -1.0)

    if "1m" not in kline:
        print("  No 1m data!", flush=True); return

    c1m = kline["1m"]["close"].values
    h1m = kline["1m"]["high"].values
    l1m = kline["1m"]["low"].values
    v1m = kline["1m"]["volume"].values
    ts_1m = kline["1m"].index

    print("\n[5/5] Evaluating grid entry + trailing TP...", flush=True)
    baseline_pnl = []
    grid_pnl = []
    trail_pnl = []
    combined_pnl = []
    exit_reasons = []

    for i in range(0, nt - 1, 4):
        above = not np.isnan(sv[i]) and tc_p[i] > sv[i]
        lt = 0.40 if above else 0.55; st = 0.55 if above else 0.40
        thresh = np.where(is_long, lt, st)
        p = probs[i]
        ev = p * np.maximum(np.abs(mfe_p[i]), 0.001) - (1-p) * np.maximum(np.abs(mae_p[i]), 0.001) - 0.0008
        valid = (p >= thresh) & (ev > 0)
        if not valid.any(): continue
        ev_m = np.where(valid, ev, -np.inf); j = np.argmax(ev_m)
        if ev_m[j] <= 0: continue

        d = int(dirs[j])
        hold = int(holds[j]) * 15  # convert 15m bars to 1m bars
        sig_ts = X_test.index[i]

        # Find in 1m data
        idx_1m = ts_1m.searchsorted(sig_ts)
        if idx_1m >= len(ts_1m) - hold: continue

        # ATR at this point
        atr_idx = atr_series.index.searchsorted(sig_ts)
        if atr_idx > 0 and atr_idx < len(atr_series):
            atr_val = atr_series.iloc[atr_idx]
        else:
            atr_val = tc_p[i] * 0.005

        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = tc_p[i] * 0.005

        # Baseline: immediate entry, fixed exit at hold_bars
        base_entry = c1m[idx_1m]
        base_exit_idx = min(idx_1m + hold, len(c1m) - 1)
        base_exit = c1m[base_exit_idx]
        base_net = d * (base_exit - base_entry) / base_entry - 0.0008
        baseline_pnl.append(base_net * 100)

        # Grid entry
        ge = simulate_grid_entry(idx_1m, d, c1m, h1m, l1m, v1m, hold, grid_window=15)
        grid_entry_price = ge["avg_entry"]
        grid_exit = c1m[base_exit_idx]
        grid_net = d * (grid_exit - grid_entry_price) / grid_entry_price - 0.0008
        grid_pnl.append(grid_net * 100)

        # Trailing TP (with baseline entry)
        tp = simulate_trailing_tp(idx_1m, d, base_entry, c1m, h1m, l1m, hold,
                                   atr=atr_val, fee=0.0008)
        trail_pnl.append(tp["net_pct"])
        exit_reasons.append(tp["exit_reason"])

        # Combined: grid entry + trailing TP
        tp_c = simulate_trailing_tp(idx_1m, d, grid_entry_price, c1m, h1m, l1m, hold,
                                     atr=atr_val, fee=0.0008)
        combined_pnl.append(tp_c["net_pct"])

    # Results
    elapsed = time.time() - t0
    N = len(baseline_pnl)

    if N == 0:
        print("  No trades!", flush=True); return

    baseline_pnl = np.array(baseline_pnl)
    grid_pnl = np.array(grid_pnl)
    trail_pnl = np.array(trail_pnl)
    combined_pnl = np.array(combined_pnl)

    # Simulate equity curves
    def equity_curve(pnls, sizing=0.03):
        cap, pk = 100000.0, 100000.0
        for p in pnls:
            dd = (pk - cap) / pk if pk > 0 else 0
            cap += (p / 100) * cap * sizing * max(0.2, 1 - dd / 0.15)
            pk = max(pk, cap)
        return (cap - 100000) / 100000 * 100

    ret_base = equity_curve(baseline_pnl)
    ret_grid = equity_curve(grid_pnl)
    ret_trail = equity_curve(trail_pnl)
    ret_combined = equity_curve(combined_pnl)

    print(f"\n{'='*60}", flush=True)
    print(f"  ITERATION 046 RESULTS ({N} trades)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Baseline (immediate entry, fixed exit):  {ret_base:+.2f}%", flush=True)
    print(f"  Grid entry only:                         {ret_grid:+.2f}%  ({ret_grid-ret_base:+.2f}%)", flush=True)
    print(f"  Trailing TP only:                        {ret_trail:+.2f}%  ({ret_trail-ret_base:+.2f}%)", flush=True)
    print(f"  Combined (grid + trail):                 {ret_combined:+.2f}%  ({ret_combined-ret_base:+.2f}%)", flush=True)

    print(f"\n  Per-trade stats:", flush=True)
    print(f"    Baseline avg: {baseline_pnl.mean():+.3f}%  WR: {(baseline_pnl>0).mean()*100:.1f}%", flush=True)
    print(f"    Grid avg:     {grid_pnl.mean():+.3f}%  WR: {(grid_pnl>0).mean()*100:.1f}%", flush=True)
    print(f"    Trail avg:    {trail_pnl.mean():+.3f}%  WR: {(trail_pnl>0).mean()*100:.1f}%", flush=True)
    print(f"    Combined avg: {combined_pnl.mean():+.3f}%  WR: {(combined_pnl>0).mean()*100:.1f}%", flush=True)

    reason_counts = Counter(exit_reasons)
    print(f"\n  Exit reasons:", flush=True)
    for r, c in reason_counts.most_common():
        print(f"    {r:15s}: {c} ({c/N*100:.1f}%)", flush=True)

    print(f"\n  Time: {elapsed:.0f}s", flush=True)

    report = {
        "iteration": 46,
        "approach": "Grid entry (3 tranches) + Trailing TP",
        "n_trades": N,
        "returns": {"baseline": round(ret_base, 2), "grid": round(ret_grid, 2),
                     "trailing": round(ret_trail, 2), "combined": round(ret_combined, 2)},
        "improvements": {"grid": round(ret_grid - ret_base, 2),
                          "trailing": round(ret_trail - ret_base, 2),
                          "combined": round(ret_combined - ret_base, 2)},
        "exit_reasons": dict(reason_counts),
        "time_seconds": round(elapsed),
    }
    with open("reports/iteration_046.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to reports/iteration_046.json", flush=True)


if __name__ == "__main__":
    main()
