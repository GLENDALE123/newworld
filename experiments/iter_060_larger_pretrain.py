#!/usr/bin/env python3
"""
Iteration 058: Multi-Coin Pre-training + BTC Fine-tune

Phase 1 (Pre-train): Train PLE on top 5 coins simultaneously
  - Each coin generates its own features + TBM labels
  - Shared model learns universal crypto patterns
  - Experts see price/volume/metrics from ANY coin

Phase 2 (Fine-tune): Take pre-trained model, fine-tune on BTC only
  - Freeze experts, train only fusion + heads
  - Or full fine-tune with low LR

Expected benefit: Better generalization from diverse training data,
especially for regimes underrepresented in BTC-only data.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, _kl_binary, train_ple_v4
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2

DATA_DIR = "data/merged"
COINS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "BNBUSDT"]
START, END = "2021-06-01", "2025-12-31"  # common range for all 5


def load_coin_data(symbol, start=START, end=END):
    """Load kline + extras for one coin."""
    kline = {}
    for tf in ["5m", "15m"]:
        p = f"{DATA_DIR}/{symbol}/kline_{tf}.parquet"
        if os.path.exists(p):
            kline[tf] = pd.read_parquet(p).set_index("timestamp").sort_index()[start:end]
    if "15m" in kline:
        kline["1h"] = kline["15m"].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        p = f"{DATA_DIR}/{symbol}/{name}.parquet"
        if os.path.exists(p):
            extras[name] = pd.read_parquet(p).set_index("timestamp").sort_index()[start:end]
    return kline, extras


def build_dataset(kline, extras, n_features_target=None):
    """Build features + labels for one coin."""
    features = generate_features_v2(
        kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"),
        target_tf="15min", progress=False)

    # Sequence features
    tf_list = features.std().sort_values(ascending=False).head(30).index.tolist()
    sc = {}
    for lag in range(1, 8):
        for col in tf_list:
            sc[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in tf_list[:10]:
        for lag in [1, 2, 4]:
            sc[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)
    features = pd.concat([features, pd.DataFrame(sc, index=features.index)], axis=1)
    features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Labels
    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr["intraday"].copy() if "intraday" in lr else pd.DataFrame()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
            for col in sw.columns:
                if col not in labels.columns:
                    labels[col] = sw[col]

    common = features.index.intersection(labels.index)
    return features.loc[common], labels.loc[common]


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100: return {"return": 0.0}
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device), torch.zeros(n, 4).to(device))
        probs, mfe_p, mae_p = out["label_probs"].cpu().numpy(), out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()
    tc = kline_15m["close"].reindex(X_test.index, method="ffill").values
    sv = kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values
    idx = np.arange(0, n - 1, 4); M = len(idx)
    il = np.array([s["dir"] == "long" for s in strat_info])
    hs = np.array([{"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(s["style"], 12) for s in strat_info])
    ds = np.where(il, 1.0, -1.0)
    ab = ~np.isnan(sv[idx]) & (tc[idx] > sv[idx])
    th = np.where(il[None, :], np.where(ab, 0.40, 0.55)[:, None], np.where(ab, 0.55, 0.40)[:, None])
    p = probs[idx]; ev = p * np.maximum(np.abs(mfe_p[idx]), 0.001) - (1 - p) * np.maximum(np.abs(mae_p[idx]), 0.001) - fee
    vd = (p >= th) & (ev > 0); em = np.where(vd, ev, -np.inf); bj = np.argmax(em, axis=1)
    ht = em[np.arange(M), bj] > 0; ti, tj = idx[ht], bj[ht]; T = len(ti)
    if T == 0: return {"return": 0.0, "trades": 0}
    ei = np.minimum(ti + hs[tj], n - 1); net = ds[tj] * (tc[ei] - tc[ti]) / tc[ti] - fee
    cap, pk = 100000.0, 100000.0
    for i in range(T):
        dd = (pk - cap) / pk if pk > 0 else 0; cap += net[i] * cap * 0.03 * max(0.2, 1 - dd / 0.15); pk = max(pk, cap)
    return {"return": round((cap - 100000) / 100000 * 100, 2), "trades": T}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 060: Multi-Coin Pre-training", flush=True)
    print("=" * 60, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Phase 1: Build datasets for all coins ──
    print("\n[1/4] Building datasets for 5 coins...", flush=True)
    coin_data = {}
    feature_cols = None

    for sym in COINS:
        print(f"  {sym}...", end=" ", flush=True)
        kline, extras = load_coin_data(sym)
        if "15m" not in kline:
            print("SKIP (no 15m)", flush=True)
            continue
        X, L = build_dataset(kline, extras)
        # Align feature columns across coins
        if feature_cols is None:
            feature_cols = list(X.columns)
        else:
            common_cols = [c for c in feature_cols if c in X.columns]
            if len(common_cols) < len(feature_cols) * 0.8:
                print(f"SKIP (only {len(common_cols)}/{len(feature_cols)} common features)", flush=True)
                continue
            X = X.reindex(columns=feature_cols, fill_value=0)

        coin_data[sym] = {"X": X, "L": L, "kline": kline}
        print(f"{X.shape}", flush=True)

    if not coin_data:
        print("No coin data!", flush=True)
        return

    # Use common feature set
    pt = {k: v for k, v in partition_features(feature_cols).items() if len(v) > 0}

    tc_cols = sorted([c for c in coin_data["BTCUSDT"]["L"].columns if c.startswith("tbm_")])
    mc_cols = sorted([c for c in coin_data["BTCUSDT"]["L"].columns if c.startswith("mae_")])
    fc_cols = sorted([c for c in coin_data["BTCUSDT"]["L"].columns if c.startswith("mfe_")])
    rc_cols = sorted([c for c in coin_data["BTCUSDT"]["L"].columns if c.startswith("rar_")])
    wc_cols = sorted([c for c in coin_data["BTCUSDT"]["L"].columns if c.startswith("wgt_")])
    si = [{"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]} for c in tc_cols]
    ns = len(tc_cols)

    # ── Phase 2: Multi-coin pre-training ──
    print(f"\n[2/4] Pre-training on {len(coin_data)} coins...", flush=True)

    # Create combined dataset from all coins (first 60% of each)
    all_train_ds = []
    all_val_ds = []
    for sym, data in coin_data.items():
        X, L = data["X"], data["L"]
        # Ensure label columns match
        for col_list in [tc_cols, mc_cols, fc_cols, rc_cols, wc_cols]:
            for c in col_list:
                if c not in L.columns:
                    L[c] = np.nan

        n = len(X)
        s1, s2 = int(n * 0.6), int(n * 0.8)
        Xn = X[feature_cols].values.astype(np.float32)
        ac = np.zeros((n, 4), dtype=np.float32); ac[:, 0] = 1.0
        wn = L[wc_cols].values if wc_cols else None

        tds = TradingDatasetV4(Xn[:s1], L[tc_cols].values[:s1], L[mc_cols].values[:s1],
                                L[fc_cols].values[:s1], L[rc_cols].values[:s1], ac[:s1],
                                wn[:s1] if wn is not None else None)
        vds = TradingDatasetV4(Xn[s1:s2], L[tc_cols].values[s1:s2], L[mc_cols].values[s1:s2],
                                L[fc_cols].values[s1:s2], L[rc_cols].values[s1:s2], ac[s1:s2],
                                wn[s1:s2] if wn is not None else None)
        all_train_ds.append(tds)
        all_val_ds.append(vds)
        print(f"    {sym}: train={s1}, val={s2-s1}", flush=True)

    combined_train = ConcatDataset(all_train_ds)
    combined_val = ConcatDataset(all_val_ds)
    print(f"  Combined: train={len(combined_train)}, val={len(combined_val)}", flush=True)

    torch.manual_seed(42); np.random.seed(42)
    pretrained_model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                              expert_hidden=128, expert_output=64, fusion_dim=192, dropout=0.2, use_vsn=False)
    train_ple_v4(pretrained_model, combined_train, combined_val,
                  epochs=30, batch_size=2048, device=device, patience=5, rdrop_alpha=1.0, seed=42)

    # ── Phase 3: Fine-tune on BTC ──
    print(f"\n[3/4] Fine-tuning on BTC...", flush=True)
    btc = coin_data["BTCUSDT"]
    X_btc, L_btc = btc["X"], btc["L"]
    n = len(X_btc); ws = n // 4
    Xn = X_btc[feature_cols].values.astype(np.float32)
    ac = np.zeros((n, 4), dtype=np.float32); ac[:, 0] = 1.0
    wn = L_btc[wc_cols].values if wc_cols else None

    # Compare: pre-trained+finetune vs train-from-scratch
    methods = {
        "from_scratch": None,
        "pretrained_finetune": pretrained_model.state_dict(),
    }

    print(f"\n[4/4] Walk-forward comparison...", flush=True)
    for method_name, init_state in methods.items():
        seed_means = []
        for seed in [42, 123, 777]:
            wrets = []
            for w in range(3):
                ts = w * (ws // 3); te = ts + int(ws * 2); ve = te + int(ws * 0.5); test_e = min(ve + ws, n)
                if test_e <= ve: continue
                torch.manual_seed(seed); np.random.seed(seed)
                tds = TradingDatasetV4(Xn[ts:te], L_btc[tc_cols].values[ts:te], L_btc[mc_cols].values[ts:te],
                                        L_btc[fc_cols].values[ts:te], L_btc[rc_cols].values[ts:te], ac[ts:te],
                                        wn[ts:te] if wn is not None else None)
                vds = TradingDatasetV4(Xn[te:ve], L_btc[tc_cols].values[te:ve], L_btc[mc_cols].values[te:ve],
                                        L_btc[fc_cols].values[te:ve], L_btc[rc_cols].values[te:ve], ac[te:ve],
                                        wn[te:ve] if wn is not None else None)

                model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                              expert_hidden=128, expert_output=64, fusion_dim=192, dropout=0.2, use_vsn=False)
                if init_state is not None:
                    model.load_state_dict({k: v.clone() for k, v in init_state.items()})

                # Fine-tune with lower LR if pretrained
                lr = 1e-4 if init_state is not None else 5e-4
                train_ple_v4(model, tds, vds, epochs=50, batch_size=2048, device=device,
                              patience=7, rdrop_alpha=1.0, seed=seed)

                r = backtest_vec(model, X_btc.iloc[ve:test_e][feature_cols],
                                  btc["kline"]["15m"], btc["kline"]["4h"], si, device=device)
                wrets.append(r["return"])
            sm = np.mean(wrets); seed_means.append(sm)
            print(f"    {method_name} seed {seed}: {[round(x, 1) for x in wrets]} → {sm:+.2f}%", flush=True)

        ov = np.mean(seed_means); sd = np.std(seed_means)
        print(f"  {method_name}: mean={ov:+.2f}% std={sd:.1f}%\n", flush=True)

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s", flush=True)
    report = {"iteration": 60, "approach": "Multi-coin pre-training + BTC fine-tune", "time": round(elapsed)}
    with open("reports/iteration_060.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved", flush=True)


if __name__ == "__main__":
    main()
