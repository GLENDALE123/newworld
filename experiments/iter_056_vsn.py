#!/usr/bin/env python3
"""
Iteration 056: Variable Selection Network (TFT Component)

From "Temporal Fusion Transformers" (Lim et al. 2021):
  VSN learns which features are important at each timestep.
  Input: all features → softmax attention → weighted features → expert

Current PLE: features → partition → expert (static)
With VSN:    features → VSN(per-feature importance) → partition → expert

VSN architecture:
  For each feature i:
    h_i = GRN(x_i)  (Gated Residual Network)
  weights = softmax(Linear(concat(h_1..h_n)))
  selected = sum(weights_i * h_i)

Simplified version for speed:
  weights = softmax(Linear(features))  → per-feature gate
  selected = features * weights

This adds ~10K params but tells the model "ignore noisy features".
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4, Expert
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, _kl_binary, train_ple_v4
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2


class VariableSelectionNetwork(nn.Module):
    """Simplified VSN: learns per-feature importance weights."""
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        # x: (B, n_features)
        weights = F.softmax(self.gate(x), dim=-1)  # (B, n_features)
        return x * weights * x.shape[-1]  # scale to maintain magnitude


class PLEv4WithVSN(PLEv4):
    """PLE v4 with Variable Selection Network before experts."""

    def __init__(self, feature_partitions, n_account_features=4, n_strategies=128,
                 expert_hidden=128, expert_output=64, fusion_dim=128, dropout=0.2,
                 vsn_hidden=64):
        super().__init__(feature_partitions, n_account_features, n_strategies,
                         expert_hidden, expert_output, fusion_dim, dropout)
        # Add VSN for each partition
        self.vsns = nn.ModuleDict()
        for name, indices in feature_partitions.items():
            self.vsns[name] = VariableSelectionNetwork(len(indices), vsn_hidden)

    def forward(self, features, account):
        # Apply VSN before expert
        expert_outs = {}
        for name, indices in self.feature_partitions.items():
            idx = torch.tensor(indices, device=features.device)
            partition_features = features[:, idx]
            # VSN: dynamically weight features
            selected = self.vsns[name](partition_features)
            expert_outs[name] = self.experts[name](selected)

        account_enc = self.account_encoder(account)
        expert_list = list(expert_outs.values())
        expert_names = list(expert_outs.keys())
        expert_stacked = torch.stack(expert_list, dim=1)

        all_cat = torch.cat(expert_list + [account_enc], dim=-1)
        q = self.gate_query(all_cat)
        scores = []
        for name in expert_names:
            k = self.gate_keys[name](expert_outs[name])
            scores.append((q * k).sum(dim=-1, keepdim=True))

        gate_scores = torch.cat(scores, dim=-1)
        gate_weights = F.softmax(gate_scores / (q.shape[-1] ** 0.5), dim=-1)
        gated = torch.bmm(gate_weights.unsqueeze(1), expert_stacked).squeeze(1)
        fused = self.fusion(torch.cat([gated, account_enc], dim=-1))

        label_logits = self.label_head(fused)
        return {
            "label_logits": label_logits,
            "label_probs": torch.sigmoid(label_logits),
            "mae_pred": self.mae_head(fused),
            "mfe_pred": self.mfe_head(fused),
            "confidence": self.conf_head(fused).squeeze(-1),
            "gate_weights": gate_weights,
        }


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100: return {"return": 0.0, "trades": 0}
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
    p = probs[idx]; ev = p*np.maximum(np.abs(mfe_p[idx]),0.001)-(1-p)*np.maximum(np.abs(mae_p[idx]),0.001)-fee
    vd = (p>=th)&(ev>0); em = np.where(vd,ev,-np.inf); bj = np.argmax(em,axis=1)
    ht = em[np.arange(M),bj]>0; ti,tj = idx[ht],bj[ht]; T = len(ti)
    if T == 0: return {"return": 0.0, "trades": 0}
    ei = np.minimum(ti+hs[tj],n-1); net = ds[tj]*(tc[ei]-tc[ti])/tc[ti]-fee
    cap, pk = 100000.0, 100000.0
    for i in range(T):
        dd=(pk-cap)/pk if pk>0 else 0; cap+=net[i]*cap*0.03*max(0.2,1-dd/0.15); pk=max(pk,cap)
    return {"return": round((cap-100000)/100000*100, 2), "trades": T,
            "wr": round((net>0).mean()*100, 1)}


def train_model(ModelClass, train_ds, val_ds, partitions, ns, device, **kwargs):
    model = ModelClass(feature_partitions=partitions, n_account_features=4, n_strategies=ns,
                       expert_hidden=128, expert_output=64, fusion_dim=192, dropout=0.2, **kwargs).to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    opt = torch.optim.AdamW(list(model.parameters())+list(loss_fn.parameters()), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=50*(len(train_ds)//2048+1))
    tl = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    bv, ni, bs = float("inf"), 0, None
    for ep in range(50):
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
            if ni>=7: break
    if bs: model.load_state_dict(bs)
    return model


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 056: Variable Selection Network (VSN)", flush=True)
    print("="*60, flush=True)

    START, END = "2021-06-01", "2025-12-31"
    SEEDS = [42, 123, 777]

    print("\n[1/3] Loading data...", flush=True)
    kline = {}
    for tf in ["5m", "15m"]:
        kline[tf] = pd.read_parquet(f"data/merged/BTCUSDT/kline_{tf}.parquet").set_index("timestamp").sort_index()[START:END]
    kline["1h"] = kline["15m"].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    kline["4h"] = kline["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        p = f"data/merged/BTCUSDT/{name}.parquet"
        if os.path.exists(p): extras[name] = pd.read_parquet(p).set_index("timestamp").sort_index()[START:END]

    features = generate_features_v2(kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"), target_tf="15min", progress=False)
    tf_list = features.std().sort_values(ascending=False).head(30).index.tolist()
    sc = {}
    for lag in range(1, 8):
        for col in tf_list: sc[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in tf_list[:10]:
        for lag in [1, 2, 4]: sc[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)
    features = pd.concat([features, pd.DataFrame(sc, index=features.index)], axis=1).dropna().replace([np.inf,-np.inf],np.nan).fillna(0)

    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
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

    models_to_test = {
        "PLE_base": (PLEv4, {}),
        "PLE+VSN": (PLEv4WithVSN, {"vsn_hidden": 64}),
    }

    print(f"\n[2/3] Testing {len(models_to_test)} models × {len(SEEDS)} seeds...", flush=True)

    for model_name, (ModelClass, extra_kwargs) in models_to_test.items():
        seed_means = []
        for seed in SEEDS:
            wrets = []
            for w in range(3):
                ts=w*(ws//3);te=ts+int(ws*2);ve=te+int(ws*0.5);test_e=min(ve+ws,n)
                if test_e<=ve: continue
                torch.manual_seed(seed);np.random.seed(seed)
                tds=TradingDatasetV4(Xn[ts:te],L[tc_cols].values[ts:te],L[mc_cols].values[ts:te],
                                      L[fc_cols].values[ts:te],L[rc_cols].values[ts:te],ac[ts:te],
                                      wn[ts:te] if wn is not None else None)
                vds=TradingDatasetV4(Xn[te:ve],L[tc_cols].values[te:ve],L[mc_cols].values[te:ve],
                                      L[fc_cols].values[te:ve],L[rc_cols].values[te:ve],ac[te:ve],
                                      wn[te:ve] if wn is not None else None)
                model = train_model(ModelClass, tds, vds, pt, ns, device, **extra_kwargs)
                r = backtest_vec(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], si, device=device)
                wrets.append(r["return"])
            sm = np.mean(wrets); seed_means.append(sm)
            print(f"    {model_name} seed {seed}: {[round(x,1) for x in wrets]} → {sm:+.2f}%", flush=True)

        ov = np.mean(seed_means); sd = np.std(seed_means)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name:15s}: mean={ov:+.2f}% std={sd:.1f}% params={params:,}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 56, "approach": "Variable Selection Network (TFT)", "time": round(elapsed)}
    with open("reports/iteration_056.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
