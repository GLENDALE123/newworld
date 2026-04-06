#!/usr/bin/env python3
"""
Iteration 048: Multi-TF Consensus Exit

Evaluate multi-TF exit vs fixed holding period.
Uses 5m/15m trend reversal instead of 1m trailing (which failed in iter 046).
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, _kl_binary
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2, compute_atr
from execution.multitf_exit import simulate_multitf_exit


def _strip_tz(df):
    if df.index.tz is not None:
        df = df.copy(); df.index = df.index.tz_localize(None)
    return df

def load_kline(data_dir="data/merged/BTCUSDT", start="2020-06-01", end="2026-02-28"):
    kline = {}
    for tf in ["5m", "15m"]:
        path = os.path.join(data_dir, f"kline_{tf}.parquet")
        if os.path.exists(path):
            kline[tf] = _strip_tz(pd.read_parquet(path).set_index("timestamp").sort_index())[start:end]
    if "15m" in kline:
        kline["1h"] = kline["15m"].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
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
            ni+=1;
            if ni>=patience: break
    if bs: model.load_state_dict(bs)
    return model


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 048: Multi-TF Consensus Exit", flush=True)
    print("="*60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    print("\n[1/4] Loading data...", flush=True)
    kline, extras = load_kline(start=START, end=END)

    print("[2/4] Features + labels...", flush=True)
    features = generate_features_v2(kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"), target_tf="15min", progress=False)
    tf = features.std().sort_values(ascending=False).head(30).index.tolist()
    sc = {}
    for lag in range(1, 8):
        for col in tf: sc[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in tf[:10]:
        for lag in [1, 2, 4]: sc[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)
    features = pd.concat([features, pd.DataFrame(sc, index=features.index)], axis=1)
    features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)

    lr_data = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr_data["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr_data:
            sw = lr_data[name].resample("15min").ffill() if name != "intraday" else lr_data[name]
            for col in sw.columns:
                if col not in labels.columns: labels[col] = sw[col]
    common = features.index.intersection(labels.index)
    X, L = features.loc[common], labels.loc[common]

    tc_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mc_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    fc_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rc_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wc_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    si = [{"style":c.replace("tbm_","").split("_")[0],"dir":c.replace("tbm_","").split("_")[1]} for c in tc_cols]
    pt = {k:v for k,v in partition_features(list(X.columns)).items() if len(v)>0}
    n, ns = len(X), len(tc_cols)
    ws = n // 4; device = "cuda" if torch.cuda.is_available() else "cpu"

    # ATR for SL
    c15 = kline["15m"]["close"].values.astype(np.float64)
    h15 = kline["15m"]["high"].values.astype(np.float64)
    l15 = kline["15m"]["low"].values.astype(np.float64)
    atr_arr = compute_atr(h15, l15, c15, period=14)
    atr_s = pd.Series(atr_arr, index=kline["15m"].index)

    # 5m data for exit
    c5m = kline["5m"]["close"].values
    ts_5m = kline["5m"].index.values
    c15m = kline["15m"]["close"].values
    h15m = kline["15m"]["high"].values
    l15m = kline["15m"]["low"].values
    ts_15m = kline["15m"].index.values

    print(f"  Data: {X.shape}", flush=True)

    print("\n[3/4] Training model (window 3, seed 42)...", flush=True)
    w = 2
    ts_w = w*(ws//3); te = ts_w+int(ws*2); ve = te+int(ws*0.5); test_e = min(ve+ws, n)
    torch.manual_seed(42); np.random.seed(42)
    Xn = X.values.astype(np.float32)
    ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
    wn = L[wc_cols].values if wc_cols else None
    tds = TradingDatasetV4(Xn[ts_w:te],L[tc_cols].values[ts_w:te],L[mc_cols].values[ts_w:te],
                            L[fc_cols].values[ts_w:te],L[rc_cols].values[ts_w:te],ac[ts_w:te],
                            wn[ts_w:te] if wn is not None else None)
    vds = TradingDatasetV4(Xn[te:ve],L[tc_cols].values[te:ve],L[mc_cols].values[te:ve],
                            L[fc_cols].values[te:ve],L[rc_cols].values[te:ve],ac[te:ve],
                            wn[te:ve] if wn is not None else None)
    model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                  expert_hidden=128, expert_output=96, fusion_dim=192, dropout=0.2)
    model = train_rdrop(model, tds, vds, epochs=50, device=device, patience=7)

    print("\n[4/4] Evaluating exits...", flush=True)
    X_test = X.iloc[ve:test_e]
    nt = len(X_test)
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device), torch.zeros(nt,4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p, mae_p = out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()

    tc_p = kline["15m"]["close"].reindex(X_test.index, method="ffill").values
    sv = kline["4h"]["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values
    il = np.array([s["dir"]=="long" for s in si])
    hs = np.array([{"scalp":1,"intraday":4,"daytrade":48,"swing":168}.get(s["style"],12) for s in si])
    ds = np.where(il, 1.0, -1.0)

    baseline_pnl = []
    multitf_pnl = []
    exit_reasons = []

    for i in range(0, nt - 1, 4):
        above = not np.isnan(sv[i]) and tc_p[i] > sv[i]
        lt = 0.40 if above else 0.55; st = 0.55 if above else 0.40
        thresh = np.where(il, lt, st)
        p = probs[i]
        ev = p * np.maximum(np.abs(mfe_p[i]), 0.001) - (1-p) * np.maximum(np.abs(mae_p[i]), 0.001) - 0.0008
        valid = (p >= thresh) & (ev > 0)
        if not valid.any(): continue
        ev_m = np.where(valid, ev, -np.inf); j = np.argmax(ev_m)
        if ev_m[j] <= 0: continue

        d = int(ds[j])
        hold = int(hs[j])
        sig_ts = X_test.index[i]

        # Find 15m bar index
        idx_15m = np.searchsorted(ts_15m, np.datetime64(sig_ts))
        if idx_15m >= len(c15m) - hold: continue

        entry_price = c15m[idx_15m]
        atr_idx = atr_s.index.searchsorted(sig_ts)
        atr_val = atr_s.iloc[min(atr_idx, len(atr_s)-1)]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = entry_price * 0.005

        # Baseline: fixed exit
        end_idx = min(idx_15m + hold, len(c15m) - 1)
        base_exit = c15m[end_idx]
        base_net = d * (base_exit - entry_price) / entry_price - 0.0008
        baseline_pnl.append(base_net * 100)

        # Multi-TF exit
        result = simulate_multitf_exit(
            idx_15m, d, entry_price,
            c5m, c15m, h15m, l15m,
            ts_5m, ts_15m,
            max_hold_15m=hold, atr=atr_val,
            sl_atr_mult=1.0, min_hold_15m=max(2, hold // 4),
        )
        multitf_pnl.append(result["net_pct"])
        exit_reasons.append(result["exit_reason"])

    # Results
    from collections import Counter
    N = len(baseline_pnl)
    baseline_pnl = np.array(baseline_pnl)
    multitf_pnl = np.array(multitf_pnl)

    def equity(pnls):
        cap, pk = 100000.0, 100000.0
        for p in pnls:
            dd = (pk-cap)/pk if pk>0 else 0
            cap += (p/100)*cap*0.03*max(0.2,1-dd/0.15); pk = max(pk,cap)
        return (cap-100000)/100000*100

    ret_base = equity(baseline_pnl)
    ret_mtf = equity(multitf_pnl)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  ITERATION 048 RESULTS ({N} trades)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Baseline (fixed exit):     {ret_base:+.2f}%", flush=True)
    print(f"  Multi-TF consensus exit:   {ret_mtf:+.2f}%  ({ret_mtf-ret_base:+.2f}%)", flush=True)
    print(f"\n  Per-trade:", flush=True)
    print(f"    Baseline avg: {baseline_pnl.mean():+.3f}%  WR: {(baseline_pnl>0).mean()*100:.1f}%", flush=True)
    print(f"    Multi-TF avg: {multitf_pnl.mean():+.3f}%  WR: {(multitf_pnl>0).mean()*100:.1f}%", flush=True)

    rc = Counter(exit_reasons)
    print(f"\n  Exit reasons:", flush=True)
    for r, c in rc.most_common():
        idx_r = [i for i, x in enumerate(exit_reasons) if x == r]
        avg = multitf_pnl[idx_r].mean()
        print(f"    {r:20s}: {c:4d} ({c/N*100:.1f}%)  avg={avg:+.3f}%", flush=True)

    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 48, "approach": "Multi-TF consensus exit (5m+15m)",
              "baseline_return": round(ret_base, 2), "multitf_return": round(ret_mtf, 2),
              "improvement": round(ret_mtf - ret_base, 2), "n_trades": N,
              "exit_reasons": dict(rc), "time": round(elapsed)}
    with open("reports/iteration_048.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
