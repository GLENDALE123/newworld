#!/usr/bin/env python3
"""
Iteration 045: Precision Entry Evaluation

Measure how much entry improvement is possible by using 1m data
after the 15m model signal fires.

Steps:
  1. Train model on full history (optimal config)
  2. Generate all trade signals from backtest
  3. For each signal, evaluate precision entry on 1m data
  4. Compare: immediate entry vs VWAP pullback vs oracle
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, _kl_binary
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2
from execution.precision_entry import find_precision_entry
from torch.utils.data import DataLoader


def _strip_tz(df):
    """Remove timezone info from DataFrame index."""
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
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
    # 1m data (for precision entry)
    path_1m = os.path.join(data_dir, "kline_1m.parquet")
    if os.path.exists(path_1m):
        kline["1m"] = _strip_tz(pd.read_parquet(path_1m).set_index("timestamp").sort_index())[start:end]
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = _strip_tz(pd.read_parquet(path).set_index("timestamp").sort_index())[start:end]
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
    return model


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 045: Precision Entry Evaluation", flush=True)
    print("=" * 60, flush=True)

    # Use overlapping period: 1m data starts 2023-04, model data starts 2020-06
    START, END = "2020-06-01", "2026-02-28"
    # 1m data available from 2023-04

    print("\n[1/5] Loading data...", flush=True)
    kline, extras = load_data_full(start=START, end=END)
    has_1m = "1m" in kline
    print(f"  1m data: {'Yes, ' + str(len(kline['1m'])) + ' bars' if has_1m else 'No'}", flush=True)

    print("\n[2/5] Building features + labels...", flush=True)
    # Exclude 1m from feature generation (different tz, not used for features)
    kline_for_features = {k: v for k, v in kline.items() if k != "1m"}
    features = generate_features_v2(kline_data=kline_for_features, tick_bar=extras.get("tick_bar"),
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
    strat_info = [{"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]} for c in tc_cols]
    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    n, ns = len(X), len(tc_cols)
    ws = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[3/5] Training model (last window only)...", flush=True)
    # Train on window 3 (most recent, overlaps with 1m data)
    w = 2
    ts = w * (ws // 3); te = ts + int(ws * 2); ve = te + int(ws * 0.5); test_e = min(ve + ws, n)

    torch.manual_seed(42); np.random.seed(42)
    X_np = X.values.astype(np.float32)
    acc = np.zeros((n, 4), dtype=np.float32); acc[:, 0] = 1.0
    wgt = L[wc_cols].values if wc_cols else None

    tds = TradingDatasetV4(X_np[ts:te], L[tc_cols].values[ts:te], L[mc_cols].values[ts:te],
                            L[fc_cols].values[ts:te], L[rc_cols].values[ts:te], acc[ts:te],
                            wgt[ts:te] if wgt is not None else None)
    vds = TradingDatasetV4(X_np[te:ve], L[tc_cols].values[te:ve], L[mc_cols].values[te:ve],
                            L[fc_cols].values[te:ve], L[rc_cols].values[te:ve], acc[te:ve],
                            wgt[te:ve] if wgt is not None else None)

    model = PLEv4(feature_partitions=partitions, n_account_features=4, n_strategies=ns,
                  expert_hidden=128, expert_output=96, fusion_dim=192, dropout=0.2)
    model = train_rdrop(model, tds, vds, epochs=50, device=device, patience=7)

    print("\n[4/5] Generating trade signals...", flush=True)
    X_test = X.iloc[ve:test_e]
    n_test = len(X_test)
    model = model.to(device).eval()

    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n_test, 4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p = out["mfe_pred"].cpu().numpy()
        mae_p = out["mae_pred"].cpu().numpy()

    tc_prices = kline["15m"]["close"].reindex(X_test.index, method="ffill").values
    sv = kline["4h"]["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values

    is_long = np.array([s["dir"] == "long" for s in strat_info])
    holds = np.array([{"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(s["style"], 12) for s in strat_info])
    dirs = np.where(is_long, 1.0, -1.0)

    # Generate signals
    signals = []
    for i in range(0, n_test - 1, 4):
        above = not np.isnan(sv[i]) and tc_prices[i] > sv[i]
        lt = 0.40 if above else 0.55
        st = 0.55 if above else 0.40
        thresh = np.where(is_long, lt, st)

        p = probs[i]
        ev = p * np.maximum(np.abs(mfe_p[i]), 0.001) - (1 - p) * np.maximum(np.abs(mae_p[i]), 0.001) - 0.0008
        valid = (p >= thresh) & (ev > 0)
        if not valid.any():
            continue
        ev_masked = np.where(valid, ev, -np.inf)
        j = np.argmax(ev_masked)
        if ev_masked[j] <= 0:
            continue

        signals.append({
            "timestamp": X_test.index[i],
            "direction": int(dirs[j]),
            "strategy": f"{strat_info[j]['style']}_{strat_info[j]['dir']}",
            "prob": float(p[j]),
            "ev": float(ev_masked[j]),
            "hold_bars": int(holds[j]),
            "bar_idx": i,
        })

    signals_df = pd.DataFrame(signals)
    print(f"  Total signals: {len(signals_df)}", flush=True)
    if len(signals_df) > 0:
        print(f"  Long: {(signals_df['direction']==1).sum()}, Short: {(signals_df['direction']==-1).sum()}", flush=True)
        print(f"  Period: {signals_df['timestamp'].min()} to {signals_df['timestamp'].max()}", flush=True)

    # ── Precision Entry Analysis ──
    print("\n[5/5] Evaluating precision entries on 1m data...", flush=True)

    if not has_1m or len(signals_df) == 0:
        print("  No 1m data or no signals. Skipping.", flush=True)
        return

    close_1m = kline["1m"]["close"].values
    high_1m = kline["1m"]["high"].values
    low_1m = kline["1m"]["low"].values
    vol_1m = kline["1m"]["volume"].values
    ts_1m = kline["1m"].index

    improvements = []
    oracle_improvements = []
    strategies_used = []

    for _, sig in signals_df.iterrows():
        ts = sig["timestamp"]
        idx = ts_1m.searchsorted(ts)
        if idx >= len(ts_1m) - 15:
            continue

        entry = find_precision_entry(
            idx, sig["direction"], close_1m, high_1m, low_1m, vol_1m,
            window=15, max_adverse=0.003
        )
        improvements.append(entry["improvement_bps"])
        oracle_improvements.append(entry["oracle_improvement_bps"])
        strategies_used.append(entry["strategy"])

    if improvements:
        improvements = np.array(improvements)
        oracle_improvements = np.array(oracle_improvements)

        print(f"\n  === PRECISION ENTRY RESULTS ===", flush=True)
        print(f"  Signals analyzed: {len(improvements)}", flush=True)
        print(f"  Avg improvement:    {np.mean(improvements):+.1f} bps", flush=True)
        print(f"  Median improvement: {np.median(improvements):+.1f} bps", flush=True)
        print(f"  Oracle avg:         {np.mean(oracle_improvements):+.1f} bps", flush=True)
        print(f"  Oracle median:      {np.median(oracle_improvements):+.1f} bps", flush=True)
        print(f"  Positive entries:   {(improvements > 0).sum()}/{len(improvements)} ({(improvements > 0).mean()*100:.1f}%)", flush=True)

        from collections import Counter
        strategy_counts = Counter(strategies_used)
        print(f"\n  Strategy breakdown:", flush=True)
        for s, c in strategy_counts.most_common():
            idx_s = [i for i, x in enumerate(strategies_used) if x == s]
            avg = np.mean(improvements[idx_s])
            print(f"    {s:20s}: {c:4d} ({c/len(improvements)*100:.1f}%)  avg={avg:+.1f} bps", flush=True)

        # Translate to return impact
        # 1 bps = 0.01% per trade. With ~2000 trades/year and 3% sizing:
        # Impact = avg_bps * n_trades * sizing / 10000
        avg_bps = np.mean(improvements)
        n_trades = len(improvements)
        annual_impact = avg_bps * n_trades * 0.03 / 10000 * 100
        print(f"\n  Estimated annual return impact: {annual_impact:+.2f}% ({avg_bps:.1f} bps × {n_trades} trades × 3% size)", flush=True)
    else:
        print("  No 1m data overlap with signals.", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)

    report = {
        "iteration": 45,
        "approach": "Precision entry evaluation (1m data within 15m signal window)",
        "n_signals": len(improvements) if improvements else 0,
        "avg_improvement_bps": round(float(np.mean(improvements)), 1) if improvements else 0,
        "oracle_avg_bps": round(float(np.mean(oracle_improvements)), 1) if improvements else 0,
        "time_seconds": round(elapsed),
    }
    with open("reports/iteration_045.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to reports/iteration_045.json", flush=True)


if __name__ == "__main__":
    main()
