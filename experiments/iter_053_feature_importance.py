#!/usr/bin/env python3
"""
Iteration 053: Feature Importance + Pruning

1. Train model, compute gradient-based feature importance
2. Rank features by importance
3. Test performance with top-N features (100, 200, 300, all 382)
4. Also add: fractional differentiation features from AFML

This tells us whether we need MORE features or FEWER features.
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
from labeling.multi_tbm_v2 import generate_multi_tbm_v2


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


def compute_gradient_importance(model, X_tensor, account_tensor, device="cuda"):
    """Compute gradient-based feature importance via input gradient magnitude."""
    model = model.to(device).eval()
    X_tensor = X_tensor.to(device).requires_grad_(True)
    account_tensor = account_tensor.to(device)

    out = model(X_tensor, account_tensor)
    # Sum all label logits (aggregate importance across all strategies)
    score = out["label_logits"].sum()
    score.backward()

    # Importance = mean absolute gradient per feature
    importance = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    return importance


def train_rdrop(model, train_ds, val_ds, epochs=50, batch_size=4096, lr=5e-4, device="cuda", patience=7):
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


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100: return {"return": 0.0}
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device), torch.zeros(n,4).to(device))
        probs, mfe_p, mae_p = out["label_probs"].cpu().numpy(), out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()
    tc = kline_15m["close"].reindex(X_test.index, method="ffill").values
    sv = kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values
    idx = np.arange(0, n-1, 4); M = len(idx)
    il = np.array([s["dir"]=="long" for s in strat_info])
    hs = np.array([{"scalp":1,"intraday":4,"daytrade":48,"swing":168}.get(s["style"],12) for s in strat_info])
    ds = np.where(il, 1.0, -1.0)
    ab = ~np.isnan(sv[idx]) & (tc[idx] > sv[idx])
    th = np.where(il[None,:], np.where(ab,0.40,0.55)[:,None], np.where(ab,0.55,0.40)[:,None])
    p = probs[idx]
    ev = p*np.maximum(np.abs(mfe_p[idx]),0.001)-(1-p)*np.maximum(np.abs(mae_p[idx]),0.001)-fee
    vd = (p>=th)&(ev>0); em = np.where(vd,ev,-np.inf); bj = np.argmax(em,axis=1)
    ht = em[np.arange(M),bj]>0; ti,tj = idx[ht],bj[ht]; T = len(ti)
    if T == 0: return {"return": 0.0, "trades": 0}
    ei = np.minimum(ti+hs[tj],n-1); net = ds[tj]*(tc[ei]-tc[ti])/tc[ti]-fee
    cap, pk = 100000.0, 100000.0
    for i in range(T):
        dd=(pk-cap)/pk if pk>0 else 0; cap+=net[i]*cap*0.03*max(0.2,1-dd/0.15); pk=max(pk,cap)
    return {"return": round((cap-100000)/100000*100, 2), "trades": T}


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 053: Feature Importance + Pruning", flush=True)
    print("="*60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print("\n[1/5] Loading data...", flush=True)
    kline, extras = load_kline(start=START, end=END)
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
    all_cols = list(features.columns)
    print(f"  Total features: {len(all_cols)}", flush=True)

    lr_data = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr_data["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr_data:
            sw = lr_data[name].resample("15min").ffill() if name != "intraday" else lr_data[name]
            for col in sw.columns:
                if col not in labels.columns: labels[col] = sw[col]
    common = features.index.intersection(labels.index)
    X_all, L = features.loc[common], labels.loc[common]

    tc_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mc_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    fc_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rc_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wc_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    si = [{"style":c.replace("tbm_","").split("_")[0],"dir":c.replace("tbm_","").split("_")[1]} for c in tc_cols]

    n = len(X_all); ns = len(tc_cols); ws = n//4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Train one model and compute feature importance ──
    print("\n[2/5] Computing feature importance...", flush=True)
    pt_all = {k:v for k,v in partition_features(all_cols).items() if len(v)>0}

    torch.manual_seed(42); np.random.seed(42)
    Xn = X_all.values.astype(np.float32)
    ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
    wn = L[wc_cols].values if wc_cols else None

    te = int(n*0.6); ve = int(n*0.8)
    tds = TradingDatasetV4(Xn[:te],L[tc_cols].values[:te],L[mc_cols].values[:te],
                            L[fc_cols].values[:te],L[rc_cols].values[:te],ac[:te],
                            wn[:te] if wn is not None else None)
    vds = TradingDatasetV4(Xn[te:ve],L[tc_cols].values[te:ve],L[mc_cols].values[te:ve],
                            L[fc_cols].values[te:ve],L[rc_cols].values[te:ve],ac[te:ve],
                            wn[te:ve] if wn is not None else None)
    model = PLEv4(feature_partitions=pt_all, n_account_features=4, n_strategies=ns,
                  expert_hidden=128, expert_output=96, fusion_dim=192, dropout=0.2)
    model = train_rdrop(model, tds, vds, epochs=50, batch_size=4096, device=device, patience=7)

    # Compute importance on validation set
    X_val_tensor = torch.tensor(Xn[te:ve])
    acc_val_tensor = torch.zeros(ve-te, 4)
    importance = compute_gradient_importance(model, X_val_tensor, acc_val_tensor, device)

    # Rank features
    ranked_idx = np.argsort(importance)[::-1]
    ranked_names = [all_cols[i] for i in ranked_idx]

    print(f"\n  Top 20 features:", flush=True)
    for i in range(20):
        print(f"    {i+1:3d}. {ranked_names[i]:40s} imp={importance[ranked_idx[i]]:.6f}", flush=True)

    print(f"\n  Bottom 10 features:", flush=True)
    for i in range(-10, 0):
        print(f"    {len(all_cols)+i+1:3d}. {ranked_names[i]:40s} imp={importance[ranked_idx[i]]:.6f}", flush=True)

    # Feature category distribution in top 100
    top100_names = set(ranked_names[:100])
    categories = {"price": 0, "volume": 0, "metrics": 0, "lag": 0, "chg": 0, "other": 0}
    for name in top100_names:
        if "_lag" in name: categories["lag"] += 1
        elif "_chg" in name: categories["chg"] += 1
        elif any(k in name for k in ["buy_ratio", "cvd", "vol_surge"]): categories["volume"] += 1
        elif any(k in name for k in ["oi", "taker", "funding", "ls_ratio"]): categories["metrics"] += 1
        else: categories["price"] += 1
    print(f"\n  Top 100 category distribution: {categories}", flush=True)

    # ── Step 2: Test with different feature counts ──
    print(f"\n[3/5] Testing feature subsets...", flush=True)
    feature_counts = [100, 200, 300, len(all_cols)]

    for n_feat in feature_counts:
        selected_cols = [all_cols[i] for i in ranked_idx[:n_feat]]
        X_sel = X_all[selected_cols]
        pt_sel = {k:v for k,v in partition_features(selected_cols).items() if len(v)>0}
        Xn_sel = X_sel.values.astype(np.float32)

        seed_means = []
        for seed in SEEDS:
            wrets = []
            for w in range(3):
                ts_w=w*(ws//3);te_w=ts_w+int(ws*2);ve_w=te_w+int(ws*0.5);test_e=min(ve_w+ws,n)
                if test_e<=ve_w: continue
                torch.manual_seed(seed);np.random.seed(seed)
                tds=TradingDatasetV4(Xn_sel[ts_w:te_w],L[tc_cols].values[ts_w:te_w],L[mc_cols].values[ts_w:te_w],
                                      L[fc_cols].values[ts_w:te_w],L[rc_cols].values[ts_w:te_w],ac[ts_w:te_w],
                                      wn[ts_w:te_w] if wn is not None else None)
                vds=TradingDatasetV4(Xn_sel[te_w:ve_w],L[tc_cols].values[te_w:ve_w],L[mc_cols].values[te_w:ve_w],
                                      L[fc_cols].values[te_w:ve_w],L[rc_cols].values[te_w:ve_w],ac[te_w:ve_w],
                                      wn[te_w:ve_w] if wn is not None else None)
                model2=PLEv4(feature_partitions=pt_sel,n_account_features=4,n_strategies=ns,
                              expert_hidden=128,expert_output=96,fusion_dim=192,dropout=0.2)
                model2=train_rdrop(model2,tds,vds,epochs=50,batch_size=4096,device=device,patience=7)
                r=backtest_vec(model2,X_sel.iloc[ve_w:test_e],kline["15m"],kline["4h"],si,device=device)
                wrets.append(r["return"])
            seed_means.append(np.mean(wrets))

        ov = np.mean(seed_means); sd = np.std(seed_means)
        print(f"  Top {n_feat:3d} features: mean={ov:+.2f}% std={sd:.1f}% seeds={[round(m,1) for m in seed_means]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 53, "approach": "Feature importance + pruning",
              "top20_features": ranked_names[:20], "time": round(elapsed)}
    with open("reports/iteration_053.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
