#!/usr/bin/env python3
"""
Iteration 054: Cross-Asset Features (ETH/SOL/BNB → BTC)

Add features from correlated assets:
  1. ETH/BTC ratio and its changes (rotation signal)
  2. SOL momentum vs BTC momentum (lead-lag)
  3. Cross-asset correlation changes (regime signal)
  4. Volume-weighted relative strength

These capture market-wide information that pure BTC features miss.
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
from ple.trainer_v4 import TradingDatasetV4, _kl_binary, train_ple_v4
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2

NUMTIC_DATA = "/home/henry/Projects/Numtic/data/merged"


def build_cross_asset_features(btc_close_15m, target_tf="15min"):
    """Build cross-asset features from ETH, SOL, BNB."""
    feats = {}
    btc = btc_close_15m

    for sym in ["ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        path = f"{NUMTIC_DATA}/{sym}/kline_15m.parquet"
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path).set_index("timestamp").sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        alt = df["close"].reindex(btc.index, method="ffill")
        prefix = sym.replace("USDT", "").lower()

        # 1. Price ratio and changes
        ratio = alt / btc.replace(0, np.nan)
        feats[f"xa_{prefix}_ratio"] = ratio
        for w in [4, 12, 48]:
            feats[f"xa_{prefix}_ratio_chg_{w}"] = ratio.pct_change(w)

        # 2. Relative momentum (alt vs BTC)
        for w in [4, 12, 48, 96]:
            btc_ret = btc.pct_change(w)
            alt_ret = alt.pct_change(w)
            feats[f"xa_{prefix}_rel_mom_{w}"] = alt_ret - btc_ret

        # 3. Rolling correlation
        btc_ret = btc.pct_change()
        alt_ret = alt.pct_change()
        for w in [24, 96]:
            feats[f"xa_{prefix}_corr_{w}"] = btc_ret.rolling(w).corr(alt_ret)

        # 4. Lead-lag: alt return predicting BTC (1-4 bar lead)
        for lag in [1, 2, 4]:
            feats[f"xa_{prefix}_lead_{lag}"] = alt_ret.shift(lag)

        # 5. Volume ratio
        alt_vol = df["volume"].reindex(btc.index, method="ffill")
        btc_vol_path = f"{NUMTIC_DATA}/BTCUSDT/kline_15m.parquet"
        btc_vol_df = pd.read_parquet(btc_vol_path).set_index("timestamp").sort_index()
        if btc_vol_df.index.tz is not None:
            btc_vol_df.index = btc_vol_df.index.tz_localize(None)
        btc_vol = btc_vol_df["volume"].reindex(btc.index, method="ffill")
        feats[f"xa_{prefix}_vol_ratio"] = alt_vol / btc_vol.replace(0, np.nan)

    result = pd.DataFrame(feats, index=btc.index)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


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
    p = probs[idx]
    ev = p*np.maximum(np.abs(mfe_p[idx]),0.001)-(1-p)*np.maximum(np.abs(mae_p[idx]),0.001)-fee
    vd = (p>=th)&(ev>0); em = np.where(vd,ev,-np.inf); bj = np.argmax(em,axis=1)
    ht = em[np.arange(M),bj]>0; ti,tj = idx[ht],bj[ht]; T = len(ti)
    if T == 0: return {"return": 0.0, "trades": 0}
    ei = np.minimum(ti+hs[tj],n-1); net = ds[tj]*(tc[ei]-tc[ti])/tc[ti]-fee
    cap, pk = 100000.0, 100000.0
    for i in range(T):
        dd=(pk-cap)/pk if pk>0 else 0; cap+=net[i]*cap*0.03*max(0.2,1-dd/0.15); pk=max(pk,cap)
    return {"return": round((cap-100000)/100000*100, 2), "trades": T,
            "wr": round((net>0).mean()*100, 1)}


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 054: Cross-Asset Features", flush=True)
    print("="*60, flush=True)

    START, END = "2021-06-01", "2025-12-31"  # SOL starts 2021
    SEEDS = [42, 123, 777]

    print("\n[1/4] Loading BTC data...", flush=True)
    kline = {}
    for tf in ["5m", "15m"]:
        kline[tf] = pd.read_parquet(f"data/merged/BTCUSDT/kline_{tf}.parquet").set_index("timestamp").sort_index()[START:END]
    kline["1h"] = kline["15m"].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    kline["4h"] = kline["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        p = f"data/merged/BTCUSDT/{name}.parquet"
        if os.path.exists(p):
            extras[name] = pd.read_parquet(p).set_index("timestamp").sort_index()[START:END]

    print(f"  15m: {len(kline['15m'])} bars", flush=True)

    print("\n[2/4] Building features...", flush=True)
    # Base features
    base_features = generate_features_v2(kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"), target_tf="15min", progress=False)

    # Sequence features
    tf_list = base_features.std().sort_values(ascending=False).head(30).index.tolist()
    sc = {}
    for lag in range(1, 8):
        for col in tf_list: sc[f"{col}_lag{lag}"] = base_features[col].shift(lag)
    for col in tf_list[:10]:
        for lag in [1, 2, 4]: sc[f"{col}_chg{lag}"] = base_features[col] - base_features[col].shift(lag)
    base_features = pd.concat([base_features, pd.DataFrame(sc, index=base_features.index)], axis=1)

    # Cross-asset features
    xa_features = build_cross_asset_features(kline["15m"]["close"])
    n_xa = xa_features.shape[1]
    print(f"  Base: {base_features.shape[1]}, Cross-asset: {n_xa}", flush=True)

    # Test both: with and without cross-asset
    configs = {
        "base_only": base_features,
        "with_xa": pd.concat([base_features, xa_features], axis=1),
    }

    print("\n[3/4] Building labels...", flush=True)
    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
            for col in sw.columns:
                if col not in labels.columns: labels[col] = sw[col]

    tc_cols = sorted([c for c in labels.columns if c.startswith("tbm_")])
    mc_cols = sorted([c for c in labels.columns if c.startswith("mae_")])
    fc_cols = sorted([c for c in labels.columns if c.startswith("mfe_")])
    rc_cols = sorted([c for c in labels.columns if c.startswith("rar_")])
    wc_cols = sorted([c for c in labels.columns if c.startswith("wgt_")])
    si = [{"style":c.replace("tbm_","").split("_")[0],"dir":c.replace("tbm_","").split("_")[1]} for c in tc_cols]
    ns = len(tc_cols)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[4/4] Walk-forward ({len(configs)} configs × {len(SEEDS)} seeds)...", flush=True)

    for cfg_name, features in configs.items():
        features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)
        common = features.index.intersection(labels.index)
        X, L = features.loc[common], labels.loc[common]
        pt = {k:v for k,v in partition_features(list(X.columns)).items() if len(v)>0}
        n = len(X); ws = n // 4
        Xn = X.values.astype(np.float32)
        ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
        wn = L[wc_cols].values if wc_cols else None

        print(f"\n  {cfg_name}: {X.shape}, partitions={[(k,len(v)) for k,v in pt.items()]}", flush=True)

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
                model=PLEv4(feature_partitions=pt,n_account_features=4,n_strategies=ns,
                            expert_hidden=128,expert_output=96,fusion_dim=192,dropout=0.2)
                train_ple_v4(model,tds,vds,epochs=50,batch_size=2048,device=device,patience=7,rdrop_alpha=1.0)
                r=backtest_vec(model,X.iloc[ve:test_e],kline["15m"],kline["4h"],si,device=device)
                wrets.append(r["return"])
            sm=np.mean(wrets); seed_means.append(sm)
            print(f"    Seed {seed}: {[round(x,1) for x in wrets]} → {sm:+.2f}%", flush=True)

        ov=np.mean(seed_means); sd=np.std(seed_means)
        print(f"  {cfg_name}: mean={ov:+.2f}% std={sd:.1f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 54, "approach": f"Cross-asset features (ETH/SOL/BNB, {n_xa} new features)", "time": round(elapsed)}
    with open("reports/iteration_054.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
