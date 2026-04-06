#!/usr/bin/env python3
"""
Iteration 051: Improved Confidence Head + Dynamic Hold

Changes:
  1. Confidence target: binary any_profitable → continuous sigmoid(best_RAR)
  2. Batch size: 2048 → 16384 (45× throughput)
  3. Retest dynamic hold ablation with improved confidence

If confidence now outputs varied values (not always ~1.0),
then confidence-scaled hold should differ from fixed×2.0.
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


def train_rdrop(model, train_ds, val_ds, epochs=50, batch_size=16384, lr=5e-4, device="cuda", patience=7):
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


def backtest_modes(model, X_test, kline_15m, kline_4h, strat_info,
                    fee=0.0008, device="cuda"):
    """Run all hold modes on same model predictions."""
    n = len(X_test)
    if n < 100: return {}
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n, 4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p, mae_p = out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()

    # Print confidence stats
    print(f"      Confidence: mean={conf.mean():.3f} std={conf.std():.3f} "
          f"min={conf.min():.3f} max={conf.max():.3f}", flush=True)

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
    if T == 0: return {m: {"return": 0.0} for m in ["fixed", "fixed2x", "conf_scaled"]}

    results = {}
    for mode in ["fixed", "fixed2x", "conf_scaled"]:
        if mode == "fixed":
            holds = base_holds[tj]
        elif mode == "fixed2x":
            holds = base_holds[tj] * 2
        else:  # conf_scaled
            mults = np.clip(conf[ti] * 2, 0.5, 2.0)
            holds = (base_holds[tj] * mults).astype(int)

        holds = np.maximum(holds, 1)
        ei = np.minimum(ti + holds, n-1)
        net = ds[tj]*(tc[ei]-tc[ti])/tc[ti]-fee
        cap, pk = 100000.0, 100000.0
        for i in range(T):
            dd = (pk-cap)/pk if pk>0 else 0
            cap += net[i]*cap*0.03*max(0.2,1-dd/0.15); pk = max(pk,cap)

        results[mode] = {
            "return": round((cap-100000)/100000*100, 2),
            "trades": T,
            "wr": round((net>0).mean()*100, 1),
            "avg_hold": round(float(holds.mean()), 1),
        }

    return results


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 051: Improved Confidence + Dynamic Hold", flush=True)
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

    print(f"\n[2/3] Training 3 seeds × 3 windows (BS=16384)...", flush=True)

    mode_results = {"fixed": [], "fixed2x": [], "conf_scaled": []}

    for seed in SEEDS:
        seed_rets = {"fixed": [], "fixed2x": [], "conf_scaled": []}
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
            model=train_rdrop(model,tds,vds,epochs=50,batch_size=16384,device=device,patience=7)

            print(f"    Seed {seed} W{w+1}:", flush=True)
            r = backtest_modes(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], si, device=device)
            for mode in mode_results:
                seed_rets[mode].append(r[mode]["return"])
                print(f"      {mode:15s}: {r[mode]['return']:+.2f}% (hold={r[mode]['avg_hold']:.0f})", flush=True)

        for mode in mode_results:
            sm = np.mean(seed_rets[mode])
            mode_results[mode].append(sm)
            print(f"  Seed {seed} {mode}: mean={sm:+.2f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  ITERATION 051 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for mode in mode_results:
        ov = np.mean(mode_results[mode])
        sd = np.std(mode_results[mode])
        print(f"  {mode:15s}: mean={ov:+.2f}% std={sd:.1f}% seeds={[round(m,1) for m in mode_results[mode]]}", flush=True)

    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 51, "approach": "Improved confidence (continuous target) + dynamic hold",
              "results": {m: {"mean": round(np.mean(mode_results[m]), 2),
                              "std": round(np.std(mode_results[m]), 2)}
                          for m in mode_results},
              "time": round(elapsed)}
    with open("reports/iteration_051.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
