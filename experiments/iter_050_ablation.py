#!/usr/bin/env python3
"""
Iteration 050: Confidence Hold Ablation Study

Is +82% from confidence intelligence, or just "hold longer"?

Test:
  fixed×1.0  — baseline (current +31%)
  fixed×1.5  — always 50% longer
  fixed×2.0  — always double hold
  conf_scaled — clip(conf×2, 0.5, 2.0)
  random_scaled — clip(random×2, 0.5, 2.0) — null hypothesis

If fixed×2.0 ≈ conf_scaled → it's just "hold longer"
If conf_scaled >> fixed×2.0 → model confidence adds real value
If random_scaled ≈ conf_scaled → confidence is noise
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


def backtest_hold_mode(model, X_test, kline_15m, kline_4h, strat_info,
                        fee=0.0008, device="cuda", hold_mult=1.0,
                        use_confidence=False, use_random=False):
    n = len(X_test)
    if n < 100: return {"return": 0.0, "trades": 0}
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n, 4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p, mae_p = out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

    tc = kline_15m["close"].reindex(X_test.index, method="ffill").values
    sv = kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values
    idx = np.arange(0, n-1, 4); M = len(idx)
    il = np.array([s["dir"]=="long" for s in strat_info])
    base_holds = np.array([{"scalp":1,"intraday":4,"daytrade":48,"swing":168}.get(s["style"],12) for s in strat_info])
    ds = np.where(il, 1.0, -1.0)
    ab = ~np.isnan(sv[idx]) & (tc[idx] > sv[idx])
    th = np.where(il[None,:], np.where(ab,0.40,0.55)[:,None], np.where(ab,0.55,0.40)[:,None])
    p = probs[idx]
    ev = p*np.maximum(np.abs(mfe_p[idx]),0.001)-(1-p)*np.maximum(np.abs(mae_p[idx]),0.001)-fee
    vd = (p>=th)&(ev>0); em = np.where(vd,ev,-np.inf); bj = np.argmax(em,axis=1)
    ht = em[np.arange(M),bj]>0; ti,tj = idx[ht],bj[ht]; T = len(ti)
    if T == 0: return {"return": 0.0, "trades": 0, "wr": 0.0, "avg_hold": 0}

    # Compute holds
    if use_confidence:
        mults = np.clip(conf[ti] * 2, 0.5, 2.0)  # confidence → multiplier
    elif use_random:
        np.random.seed(999)  # deterministic random for fair comparison
        mults = np.clip(np.random.uniform(0, 1, T) * 2, 0.5, 2.0)
    else:
        mults = np.full(T, hold_mult)

    holds = np.maximum((base_holds[tj] * mults).astype(int), 1)
    ei = np.minimum(ti + holds, n-1)
    net = ds[tj]*(tc[ei]-tc[ti])/tc[ti]-fee

    # Also track avg confidence for analysis
    avg_conf = conf[ti].mean() if use_confidence else 0.0

    cap, pk = 100000.0, 100000.0
    for i in range(T):
        dd = (pk-cap)/pk if pk>0 else 0
        cap += net[i]*cap*0.03*max(0.2,1-dd/0.15); pk = max(pk,cap)

    return {"return": round((cap-100000)/100000*100, 2), "trades": T,
            "wr": round((net>0).mean()*100, 1),
            "avg_hold": round(float(holds.mean()), 1),
            "avg_conf": round(float(avg_conf), 3),
            "avg_mult": round(float(mults.mean()), 2)}


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 050: Confidence Hold Ablation", flush=True)
    print("="*60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print("\n[1/3] Loading data...", flush=True)
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

    lr_data = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr_data["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr_data:
            sw = lr_data[name].resample("15min").ffill() if name != "intraday" else lr_data[name]
            for col in sw.columns:
                if col not in labels.columns: labels[col] = sw[col]
    common = features.index.intersection(labels.index)
    X, L = features.loc[common], labels.loc[common]
    print(f"  Data: {X.shape}", flush=True)

    tc_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mc_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    fc_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rc_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wc_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    si = [{"style":c.replace("tbm_","").split("_")[0],"dir":c.replace("tbm_","").split("_")[1]} for c in tc_cols]
    pt = {k:v for k,v in partition_features(list(X.columns)).items() if len(v)>0}
    n, ns = len(X), len(tc_cols); ws = n//4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xn = X.values.astype(np.float32)
    ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
    wn = L[wc_cols].values if wc_cols else None

    modes = [
        {"name": "fixed×1.0", "mult": 1.0, "conf": False, "rand": False},
        {"name": "fixed×1.5", "mult": 1.5, "conf": False, "rand": False},
        {"name": "fixed×2.0", "mult": 2.0, "conf": False, "rand": False},
        {"name": "conf_scaled", "mult": 1.0, "conf": True, "rand": False},
        {"name": "random_scaled", "mult": 1.0, "conf": False, "rand": True},
    ]

    print(f"\n[2/3] {len(modes)} modes × {len(SEEDS)} seeds × 3 windows...", flush=True)

    for mode in modes:
        seed_data = []
        for seed in SEEDS:
            wrets = []; wdetails = []
            for w in range(3):
                ts_w=w*(ws//3);te=ts_w+int(ws*2);ve=te+int(ws*0.5);test_e=min(ve+ws,n)
                if test_e<=ve: continue
                torch.manual_seed(seed);np.random.seed(seed)
                tds=TradingDatasetV4(Xn[ts_w:te],L[tc_cols].values[ts_w:te],L[mc_cols].values[ts_w:te],
                                      L[fc_cols].values[ts_w:te],L[rc_cols].values[ts_w:te],ac[ts_w:te],
                                      wn[ts_w:te] if wn is not None else None)
                vds=TradingDatasetV4(Xn[te:ve],L[tc_cols].values[te:ve],L[mc_cols].values[te:ve],
                                      L[fc_cols].values[te:ve],L[rc_cols].values[te:ve],ac[te:ve],
                                      wn[te:ve] if wn is not None else None)
                model=PLEv4(feature_partitions=pt,n_account_features=4,n_strategies=ns,
                            expert_hidden=128,expert_output=96,fusion_dim=192,dropout=0.2)
                model=train_rdrop(model,tds,vds,epochs=50,device=device,patience=7)
                r=backtest_hold_mode(model,X.iloc[ve:test_e],kline["15m"],kline["4h"],si,
                                      device=device,hold_mult=mode["mult"],
                                      use_confidence=mode["conf"],use_random=mode["rand"])
                wrets.append(r["return"])
                wdetails.append(r)

            sm = np.mean(wrets)
            seed_data.append({"seed": seed, "windows": wrets, "mean": sm, "details": wdetails})

        seed_means = [d["mean"] for d in seed_data]
        ov = np.mean(seed_means); sd = np.std(seed_means)

        # Get avg hold and conf from first seed's details
        avg_hold = np.mean([d["avg_hold"] for d in seed_data[0]["details"]])
        avg_mult = np.mean([d["avg_mult"] for d in seed_data[0]["details"]])

        print(f"  {mode['name']:18s}: mean={ov:+.2f}% std={sd:.1f}% "
              f"avg_hold={avg_hold:.1f} avg_mult={avg_mult:.2f} "
              f"seeds={[round(m,1) for m in seed_means]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 50, "approach": "Confidence hold ablation study", "time": round(elapsed)}
    with open("reports/iteration_050.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
