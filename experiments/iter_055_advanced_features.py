#!/usr/bin/env python3
"""
Iteration 055: Advanced Features — FracDiff + FFT + Cross-Asset

Three cutting-edge feature engineering techniques:

1. Fractional Differentiation (AFML Ch.5):
   - Price series: d≈0.3-0.5 preserves memory while achieving stationarity
   - Apply to close, volume, OI → stationary but information-preserving

2. FFT Frequency Features (StockMixer 2025):
   - Dominant frequency periods (cycles in price)
   - Spectral energy in different frequency bands
   - Phase of dominant cycle

3. Cross-Asset Lead-Lag (ETH/SOL):
   - ETH/BTC ratio momentum
   - SOL-BTC return correlation changes
   - Alt-coin volume relative to BTC
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
from validation.features.fracdiff import fracdiff

NUMTIC = "/home/henry/Projects/Numtic/data/merged"


def build_fracdiff_features(kline_15m, d=0.4):
    """Fractionally differentiate key price series."""
    feats = {}
    close = kline_15m["close"]
    volume = kline_15m["volume"]

    # FracDiff on close at multiple d values
    for d_val in [0.3, 0.5]:
        fd = fracdiff(close, d_val)
        feats[f"fd_close_{d_val}"] = fd
        # Lag features of fracdiff
        for lag in [1, 4, 12]:
            feats[f"fd_close_{d_val}_lag{lag}"] = fd.shift(lag)

    # FracDiff on volume
    fd_vol = fracdiff(volume, 0.5)
    feats["fd_volume_05"] = fd_vol

    # FracDiff on high-low range (volatility proxy)
    hl_range = kline_15m["high"] - kline_15m["low"]
    feats["fd_range_04"] = fracdiff(hl_range, 0.4)

    return pd.DataFrame(feats, index=close.index)


def build_fft_features(close, windows=[96, 384, 960]):
    """FFT frequency domain features — subsampled + batch FFT for speed.

    FFT features change slowly, so compute every `step` bars and forward-fill.
    Then use batch numpy FFT on the subsampled positions.
    """
    feats = {}
    values = close.values.astype(np.float64)
    n = len(values)

    for w in windows:
        step = max(w // 8, 4)  # compute every w/8 bars (e.g. w=96 -> every 12)
        positions = np.arange(w, n, step)
        n_pos = len(positions)
        if n_pos == 0:
            feats[f"fft_period_{w}"] = np.full(n, np.nan)
            feats[f"fft_spec_ratio_{w}"] = np.full(n, np.nan)
            feats[f"fft_phase_{w}"] = np.full(n, np.nan)
            continue

        # Build segment matrix using stride_tricks (zero-copy view)
        strides = (values.strides[0], values.strides[0])
        all_segs = np.lib.stride_tricks.as_strided(
            values, shape=(n - w + 1, w), strides=strides,
        )
        segs = all_segs[positions - w].copy()  # only the subsampled ones

        # Detrend
        ramp = np.linspace(0, 1, w)[None, :]
        segs -= segs[:, 0:1] + ramp * (segs[:, -1:] - segs[:, 0:1])

        # Batch FFT
        fft_vals = np.fft.rfft(segs, axis=1)
        mags = np.abs(fft_vals[:, 1:])

        dom_idx = np.argmax(mags, axis=1)
        quarter = max(mags.shape[1] // 4, 1)
        low_e = np.sum(mags[:, :quarter] ** 2, axis=1)
        high_e = np.sum(mags[:, -quarter:] ** 2, axis=1)
        total = low_e + high_e

        dp_sparse = w / (dom_idx + 1).astype(np.float64)
        sr_sparse = np.where(total > 0, low_e / total, 0.5)
        row_idx = np.arange(n_pos)
        ph_sparse = np.angle(fft_vals[row_idx, dom_idx + 1])

        # Scatter into full arrays and forward-fill
        dom_period = np.full(n, np.nan)
        spectral_ratio = np.full(n, np.nan)
        phase = np.full(n, np.nan)
        dom_period[positions] = dp_sparse
        spectral_ratio[positions] = sr_sparse
        phase[positions] = ph_sparse

        # Forward fill using pandas (fast)
        dom_period = pd.Series(dom_period).ffill().values
        spectral_ratio = pd.Series(spectral_ratio).ffill().values
        phase = pd.Series(phase).ffill().values

        feats[f"fft_period_{w}"] = dom_period
        feats[f"fft_spec_ratio_{w}"] = spectral_ratio
        feats[f"fft_phase_{w}"] = phase

    return pd.DataFrame(feats, index=close.index)


def build_cross_asset_features(btc_close):
    """Cross-asset features from ETH and SOL."""
    feats = {}

    for sym in ["ETHUSDT", "SOLUSDT"]:
        path = f"{NUMTIC}/{sym}/kline_15m.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).set_index("timestamp").sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        alt = df["close"].reindex(btc_close.index, method="ffill")
        prefix = sym[:3].lower()

        ratio = alt / btc_close.replace(0, np.nan)
        feats[f"xa_{prefix}_ratio"] = ratio
        for w in [4, 12, 48]:
            feats[f"xa_{prefix}_ratio_chg_{w}"] = ratio.pct_change(w)

        btc_ret = btc_close.pct_change()
        alt_ret = alt.pct_change()
        for w in [24, 96]:
            feats[f"xa_{prefix}_corr_{w}"] = btc_ret.rolling(w).corr(alt_ret)

        for lag in [1, 2, 4]:
            feats[f"xa_{prefix}_lead_{lag}"] = alt_ret.shift(lag)

    return pd.DataFrame(feats, index=btc_close.index)


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


def main():
    t0 = time.time()
    print("="*60, flush=True)
    print("  ITERATION 055: FracDiff + FFT + Cross-Asset", flush=True)
    print("="*60, flush=True)

    START, END = "2021-06-01", "2025-12-31"
    SEEDS = [42, 123, 777]

    print("\n[1/5] Loading data...", flush=True)
    kline = {}
    for tf in ["5m", "15m"]:
        kline[tf] = pd.read_parquet(f"data/merged/BTCUSDT/kline_{tf}.parquet").set_index("timestamp").sort_index()[START:END]
    kline["1h"] = kline["15m"].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    kline["4h"] = kline["1h"].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        p = f"data/merged/BTCUSDT/{name}.parquet"
        if os.path.exists(p): extras[name] = pd.read_parquet(p).set_index("timestamp").sort_index()[START:END]

    print(f"  15m: {len(kline['15m'])} bars", flush=True)

    print("\n[2/5] Building base features...", flush=True)
    base = generate_features_v2(kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"), target_tf="15min", progress=False)
    tf_list = base.std().sort_values(ascending=False).head(30).index.tolist()
    sc = {}
    for lag in range(1, 8):
        for col in tf_list: sc[f"{col}_lag{lag}"] = base[col].shift(lag)
    for col in tf_list[:10]:
        for lag in [1, 2, 4]: sc[f"{col}_chg{lag}"] = base[col] - base[col].shift(lag)
    base = pd.concat([base, pd.DataFrame(sc, index=base.index)], axis=1)
    print(f"  Base features: {base.shape[1]}", flush=True)

    print("\n[3/5] Building advanced features...", flush=True)
    t_fd = time.time()
    fd_feats = build_fracdiff_features(kline["15m"])
    print(f"  FracDiff: {fd_feats.shape[1]} features ({time.time()-t_fd:.1f}s)", flush=True)

    t_fft = time.time()
    fft_feats = build_fft_features(kline["15m"]["close"], windows=[96, 384])
    print(f"  FFT: {fft_feats.shape[1]} features ({time.time()-t_fft:.1f}s)", flush=True)

    t_xa = time.time()
    xa_feats = build_cross_asset_features(kline["15m"]["close"])
    print(f"  Cross-Asset: {xa_feats.shape[1]} features ({time.time()-t_xa:.1f}s)", flush=True)

    # Configs: base vs base+advanced
    configs = {
        "base": base,
        "base+fd": pd.concat([base, fd_feats], axis=1),
        "base+fft": pd.concat([base, fft_feats], axis=1),
        "base+xa": pd.concat([base, xa_feats], axis=1),
        "base+all": pd.concat([base, fd_feats, fft_feats, xa_feats], axis=1),
    }

    print("\n[4/5] Building labels...", flush=True)
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

    print(f"\n[5/5] Walk-forward ({len(configs)} configs × {len(SEEDS)} seeds)...", flush=True)

    for cfg_name, features in configs.items():
        features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)
        common = features.index.intersection(labels.index)
        X, L = features.loc[common], labels.loc[common]
        pt = {k:v for k,v in partition_features(list(X.columns)).items() if len(v)>0}
        n = len(X); ws = n // 4
        Xn = X.values.astype(np.float32)
        ac = np.zeros((n,4),dtype=np.float32); ac[:,0]=1.0
        wn = L[wc_cols].values if wc_cols else None

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

        ov=np.mean(seed_means); sd=np.std(seed_means)
        print(f"  {cfg_name:15s}: n={n:5d} feat={X.shape[1]:3d} mean={ov:+.2f}% std={sd:.1f}% seeds={[round(m,1) for m in seed_means]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 55, "approach": "FracDiff + FFT + Cross-Asset features", "time": round(elapsed)}
    with open("reports/iteration_055.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
