#!/usr/bin/env python3
"""
Iteration 044: Longer Training + LR Schedule

Model is rock-stable (std 0.1%). Can we get higher returns by:
  1. Longer training: 80/100 epochs (vs 50)
  2. Higher patience: 12 (vs 7)
  3. Lower LR: 3e-4 (vs 5e-4)
  4. Combination of above

Using optimal arch: e128_o96_f192, d=0.20, α=1.0
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


def load_data_full(data_dir="data/merged/BTCUSDT", start="2020-06-01", end="2026-02-28"):
    kline = {}
    for tf in ["5m", "15m"]:
        path = os.path.join(data_dir, f"kline_{tf}.parquet")
        if os.path.exists(path):
            kline[tf] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]
    if "15m" in kline:
        kline["1h"] = kline["15m"].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]
    return kline, extras


def train_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                lr=5e-4, device="cuda", patience=7):
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters())+list(loss_fn.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs*(len(train_ds)//batch_size+1))
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    bv, ni, bs = float("inf"), 0, None
    for ep in range(epochs):
        model.train()
        for b in tl:
            b = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
            if np.random.random()<0.5:
                lam=np.random.beta(0.2,0.2); idx=torch.randperm(b["features"].size(0),device=device)
                for k in b:
                    if isinstance(b[k],torch.Tensor) and b[k].dtype==torch.float32: b[k]=lam*b[k]+(1-lam)*b[k][idx]
            o1=model(b["features"],b["account"]); o2=model(b["features"],b["account"])
            l1,l2=loss_fn(o1,b),loss_fn(o2,b)
            t=(l1["total"]+l2["total"])/2+1.0*_kl_binary(o1["label_probs"],o2["label_probs"],b["rar_mask"])
            optimizer.zero_grad(); t.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); scheduler.step()
        model.eval()
        with torch.no_grad():
            vm=[loss_fn(model(bb["features"],bb["account"]),bb)["total"].item() for bb in [{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()} for b in vl]]
        v=np.mean(vm)
        if v<bv: bv,ni=v,0; bs={k:val.cpu().clone() for k,val in model.state_dict().items()}
        else:
            ni+=1
            if ni>=patience: break
    if bs: model.load_state_dict(bs)
    return model, ep+1  # return actual epochs trained


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n=len(X_test)
    if n<100: return {"return":0.0,"trades":0,"wr":0.0}
    model=model.to(device).eval()
    with torch.no_grad():
        out=model(torch.tensor(X_test.values.astype(np.float32)).to(device),torch.zeros(n,4).to(device))
        probs,mfe_p,mae_p=out["label_probs"].cpu().numpy(),out["mfe_pred"].cpu().numpy(),out["mae_pred"].cpu().numpy()
    tc=kline_15m["close"].reindex(X_test.index,method="ffill").values
    sv=kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index,method="ffill").values
    idx=np.arange(0,n-1,4);M=len(idx)
    il=np.array([s["dir"]=="long" for s in strat_info])
    hs=np.array([{"scalp":1,"intraday":4,"daytrade":48,"swing":168}.get(s["style"],12) for s in strat_info])
    ds=np.where(il,1.0,-1.0)
    ab=~np.isnan(sv[idx])&(tc[idx]>sv[idx])
    th=np.where(il[None,:],np.where(ab,0.40,0.55)[:,None],np.where(ab,0.55,0.40)[:,None])
    p=probs[idx]; ev=p*np.maximum(np.abs(mfe_p[idx]),0.001)-(1-p)*np.maximum(np.abs(mae_p[idx]),0.001)-fee
    vd=(p>=th)&(ev>0); em=np.where(vd,ev,-np.inf); bj=np.argmax(em,axis=1)
    ht=em[np.arange(M),bj]>0; ti,tj=idx[ht],bj[ht]; T=len(ti)
    if T==0: return {"return":0.0,"trades":0,"wr":0.0}
    ei=np.minimum(ti+hs[tj],n-1); net=ds[tj]*(tc[ei]-tc[ti])/tc[ti]-fee
    cap,pk=100000.0,100000.0
    for i in range(T):
        dd=(pk-cap)/pk if pk>0 else 0; cap+=net[i]*cap*0.03*max(0.2,1-dd/0.15); pk=max(pk,cap)
    return {"return":round((cap-100000)/100000*100,2),"trades":T,"wr":round((net>0).mean()*100,1)}


def main():
    t0=time.time()
    print("="*60,flush=True)
    print("  ITERATION 044: Longer Training + LR Tuning",flush=True)
    print("="*60,flush=True)

    START,END="2020-06-01","2026-02-28"
    SEEDS=[42,123,777]

    print("\n[1/3] Loading data...",flush=True)
    kline,extras=load_data_full(start=START,end=END)
    features=generate_features_v2(kline_data=kline,tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"),funding=extras.get("funding_rate"),target_tf="15min",progress=False)
    tf=features.std().sort_values(ascending=False).head(30).index.tolist()
    sc={}
    for lag in range(1,8):
        for col in tf: sc[f"{col}_lag{lag}"]=features[col].shift(lag)
    for col in tf[:10]:
        for lag in [1,2,4]: sc[f"{col}_chg{lag}"]=features[col]-features[col].shift(lag)
    features=pd.concat([features,pd.DataFrame(sc,index=features.index)],axis=1)
    features=features.dropna().replace([np.inf,-np.inf],np.nan).fillna(0)

    lr_data=generate_multi_tbm_v2(kline,fee_pct=0.0008,progress=False)
    labels=lr_data["intraday"].copy()
    for name in ["scalp","daytrade","swing"]:
        if name in lr_data:
            sw=lr_data[name].resample("15min").ffill() if name!="intraday" else lr_data[name]
            for col in sw.columns:
                if col not in labels.columns: labels[col]=sw[col]
    common=features.index.intersection(labels.index)
    X,L=features.loc[common],labels.loc[common]
    print(f"  Data: {X.shape}",flush=True)

    tc=sorted([c for c in L.columns if c.startswith("tbm_")])
    mc=sorted([c for c in L.columns if c.startswith("mae_")])
    fc=sorted([c for c in L.columns if c.startswith("mfe_")])
    rc=sorted([c for c in L.columns if c.startswith("rar_")])
    wc=sorted([c for c in L.columns if c.startswith("wgt_")])
    si=[{"style":c.replace("tbm_","").split("_")[0],"dir":c.replace("tbm_","").split("_")[1]} for c in tc]
    pt={k:v for k,v in partition_features(list(X.columns)).items() if len(v)>0}
    n,ns=len(X),len(tc); ws=n//4; device="cuda" if torch.cuda.is_available() else "cpu"
    Xn=X.values.astype(np.float32); ac=np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
    wn=L[wc].values if wc else None

    configs=[
        {"name":"ep50_lr5e4_p7","epochs":50,"lr":5e-4,"patience":7},
        {"name":"ep80_lr5e4_p12","epochs":80,"lr":5e-4,"patience":12},
        {"name":"ep100_lr3e4_p12","epochs":100,"lr":3e-4,"patience":12},
        {"name":"ep80_lr3e4_p10","epochs":80,"lr":3e-4,"patience":10},
    ]

    print(f"\n[2/3] Testing {len(configs)} configs × {len(SEEDS)} seeds...",flush=True)

    for cfg in configs:
        sm=[]
        for seed in SEEDS:
            wr=[]
            for w in range(3):
                ts=w*(ws//3);te=ts+int(ws*2);ve=te+int(ws*0.5);test_e=min(ve+ws,n)
                if test_e<=ve: continue
                torch.manual_seed(seed);np.random.seed(seed)
                tds=TradingDatasetV4(Xn[ts:te],L[tc].values[ts:te],L[mc].values[ts:te],L[fc].values[ts:te],L[rc].values[ts:te],ac[ts:te],wn[ts:te] if wn is not None else None)
                vds=TradingDatasetV4(Xn[te:ve],L[tc].values[te:ve],L[mc].values[te:ve],L[fc].values[te:ve],L[rc].values[te:ve],ac[te:ve],wn[te:ve] if wn is not None else None)
                model=PLEv4(feature_partitions=pt,n_account_features=4,n_strategies=ns,expert_hidden=128,expert_output=96,fusion_dim=192,dropout=0.2)
                model,actual_ep=train_rdrop(model,tds,vds,epochs=cfg["epochs"],lr=cfg["lr"],device=device,patience=cfg["patience"])
                r=backtest_vec(model,X.iloc[ve:test_e],kline["15m"],kline["4h"],si,device=device)
                wr.append(r["return"])
            m=np.mean(wr); sm.append(m)
        ov=np.mean(sm);sd=np.std(sm)
        print(f"  {cfg['name']:25s}: mean={ov:+.2f}% std={sd:.1f}% seeds={[round(x,1) for x in sm]}",flush=True)

    elapsed=time.time()-t0
    print(f"\n  Time: {elapsed:.0f}s",flush=True)
    report={"iteration":44,"approach":"Longer training + LR tuning","time":round(elapsed)}
    with open("reports/iteration_044.json","w") as f: json.dump(report,f,indent=2)
    print(f"  Report saved",flush=True)

if __name__=="__main__": main()
