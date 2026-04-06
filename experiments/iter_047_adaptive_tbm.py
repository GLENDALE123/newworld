#!/usr/bin/env python3
"""
Iteration 047: Adaptive TBM Barriers Per Regime

Current TBM v2: fixed TP/SL for all regimes (e.g., intraday TP=2×ATR, SL=1×ATR)
Problem: same barriers in trending vs ranging markets → suboptimal labels

Adaptive barriers:
  surge:    TP=3.0×ATR, SL=1.0×ATR (let winners run in momentum)
  dump:     TP=3.0×ATR, SL=1.0×ATR (same logic, opposite direction)
  range:    TP=1.5×ATR, SL=1.0×ATR (take quick profits in chop)
  volatile: TP=2.0×ATR, SL=1.5×ATR (wider SL to avoid whipsaws)

Compare adaptive vs fixed barriers on label quality + model performance.
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
from labeling.multi_tbm_v2 import (
    compute_atr, detect_regimes, _compute_strategy_tbm,
    STRATEGIES, HOLD_PERIODS, DIRECTIONS, REGIMES, FEE_PCT,
)


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
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = _strip_tz(pd.read_parquet(path).set_index("timestamp").sort_index())[start:end]
    return kline, extras


# ── Adaptive TBM barrier multipliers per regime ──
ADAPTIVE_BARRIERS = {
    # regime: (tp_mult_scale, sl_mult_scale) — multiplied with strategy's base TP/SL
    "surge":    (1.5, 1.0),   # wider TP in momentum
    "dump":     (1.5, 1.0),   # wider TP in momentum
    "range":    (0.75, 1.0),  # tighter TP in chop
    "volatile": (1.0, 1.5),   # wider SL to avoid whipsaws
}


def generate_adaptive_tbm(kline_data, fee_pct=FEE_PCT):
    """Generate labels with regime-adaptive TP/SL multipliers."""
    label_matrix = {}

    for strat_name, strat_cfg in STRATEGIES.items():
        source_tf = strat_cfg["source_tf"]
        if source_tf not in kline_data:
            continue

        df = kline_data[source_tf].copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()

        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)

        atr = compute_atr(h, l, c, period=strat_cfg["atr_period"])
        max_hold = HOLD_PERIODS[strat_name]
        regimes = detect_regimes(df["close"], atr, window=max(24, max_hold))

        for dir_name in DIRECTIONS:
            direction = 1 if dir_name == "long" else -1

            for regime in REGIMES:
                mask = regimes == regime
                tp_scale, sl_scale = ADAPTIVE_BARRIERS[regime]

                # Adaptive multipliers
                tp_mult = strat_cfg["tp_atr_mult"] * tp_scale
                sl_mult = strat_cfg["sl_atr_mult"] * sl_scale

                raw_tbm, raw_mae, raw_mfe, raw_rar, raw_weight = _compute_strategy_tbm(
                    h, l, c, atr,
                    tp_mult=tp_mult,
                    sl_mult=sl_mult,
                    max_hold=max_hold,
                    direction=direction,
                    fee_pct=fee_pct,
                )

                base = f"{strat_name}_{dir_name}_{regime}"
                label_matrix[f"tbm_{base}"] = np.where(mask, raw_tbm, np.nan)
                label_matrix[f"mae_{base}"] = np.where(mask, raw_mae, np.nan)
                label_matrix[f"mfe_{base}"] = np.where(mask, raw_mfe, np.nan)
                label_matrix[f"rar_{base}"] = np.where(mask, raw_rar, np.nan)
                label_matrix[f"wgt_{base}"] = np.where(mask, raw_weight, np.nan)

    # Build per-strategy DataFrames
    results = {}
    for strat_name, strat_cfg in STRATEGIES.items():
        source_tf = strat_cfg["source_tf"]
        if source_tf not in kline_data:
            continue
        df = kline_data[source_tf]
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()
        cols = {k: v for k, v in label_matrix.items() if strat_name in k}
        results[strat_name] = pd.DataFrame(cols, index=df.index)

    return results


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
    cap,pk = 100000.0,100000.0
    for i in range(T):
        dd=(pk-cap)/pk if pk>0 else 0; cap+=net[i]*cap*0.03*max(0.2,1-dd/0.15); pk=max(pk,cap)
    return {"return": round((cap-100000)/100000*100, 2), "trades": T,
            "wr": round((net>0).mean()*100, 1)}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 047: Adaptive TBM Barriers", flush=True)
    print("=" * 60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print("\n[1/5] Loading data...", flush=True)
    kline, extras = load_data_full(start=START, end=END)

    print("\n[2/5] Features...", flush=True)
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
    print(f"  Features: {features.shape}", flush=True)

    # Generate both label sets
    from labeling.multi_tbm_v2 import generate_multi_tbm_v2

    print("\n[3/5] Labels: fixed vs adaptive barriers...", flush=True)
    label_configs = {
        "fixed": generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False),
        "adaptive": generate_adaptive_tbm(kline, fee_pct=0.0008),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for lcfg_name, lr_data in label_configs.items():
        labels = lr_data["intraday"].copy() if "intraday" in lr_data else pd.DataFrame()
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
        ws = n // 4

        # Label quality stats
        for style in ["scalp", "intraday", "daytrade", "swing"]:
            rar_s = [c for c in rc_cols if style in c]
            if rar_s:
                vals = L[rar_s].values.flatten()
                valid = ~np.isnan(vals)
                if valid.sum() > 0:
                    wr = (vals[valid] > 0).mean() * 100
                    print(f"    {lcfg_name} {style}: WR={wr:.1f}% avgRAR={vals[valid].mean():.4f}", flush=True)

        print(f"\n[4/5] Training {lcfg_name} ({n} samples, {ns} strategies)...", flush=True)
        Xn = X.values.astype(np.float32)
        ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
        wn = L[wc_cols].values if wc_cols else None

        seed_means = []
        for seed in SEEDS:
            wrets = []
            for w in range(3):
                ts = w*(ws//3); te = ts+int(ws*2); ve = te+int(ws*0.5); test_e = min(ve+ws, n)
                if test_e <= ve: continue
                torch.manual_seed(seed); np.random.seed(seed)
                tds = TradingDatasetV4(Xn[ts:te],L[tc_cols].values[ts:te],L[mc_cols].values[ts:te],
                                        L[fc_cols].values[ts:te],L[rc_cols].values[ts:te],ac[ts:te],
                                        wn[ts:te] if wn is not None else None)
                vds = TradingDatasetV4(Xn[te:ve],L[tc_cols].values[te:ve],L[mc_cols].values[te:ve],
                                        L[fc_cols].values[te:ve],L[rc_cols].values[te:ve],ac[te:ve],
                                        wn[te:ve] if wn is not None else None)
                model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                              expert_hidden=128, expert_output=96, fusion_dim=192, dropout=0.2)
                model = train_rdrop(model, tds, vds, epochs=50, device=device, patience=7)
                r = backtest_vec(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], si, device=device)
                wrets.append(r["return"])
            sm = np.mean(wrets)
            seed_means.append(sm)
            print(f"    Seed {seed}: {[round(x,1) for x in wrets]} → mean={sm:+.2f}%", flush=True)

        overall = np.mean(seed_means)
        std = np.std(seed_means)
        print(f"  {lcfg_name}: overall={overall:+.2f}%, std={std:.1f}%\n", flush=True)

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 47, "approach": "Adaptive TBM barriers per regime", "time": round(elapsed)}
    with open("reports/iteration_047.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
