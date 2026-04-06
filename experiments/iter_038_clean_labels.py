#!/usr/bin/env python3
"""
Iteration 038: Remove Scalp Labels + Feature Pruning

Scalp labels: WR=31%, avgRAR=-0.54 — complete noise.
Model wastes capacity trying to find patterns in unpredictable data.

Changes:
  1. Remove scalp strategy from label generation (3 styles instead of 4)
  2. Prune near-zero variance features (reduce 382 → ~300)
  3. Use optimal config: d=0.20, α=1.0 (from iter 037)

Expected: Higher precision, less overfitting, faster training.
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
        kline["1h"] = kline["15m"].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]
    return kline, extras


def train_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                lr=5e-4, device="cuda", patience=7, rdrop_alpha=1.0):
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * (len(train_ds) // batch_size + 1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val, no_improve, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]
            out1 = model(batch["features"], batch["account"])
            out2 = model(batch["features"], batch["account"])
            l1, l2 = loss_fn(out1, batch), loss_fn(out2, batch)
            total = (l1["total"] + l2["total"]) / 2 + rdrop_alpha * _kl_binary(
                out1["label_probs"], out2["label_probs"], batch["rar_mask"])
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            vm = []
            for b in val_loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                vm.append(loss_fn(model(b["features"], b["account"]), b)["total"].item())
            v = np.mean(vm)

        if v < best_val:
            best_val, no_improve = v, 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience: break

    if best_state: model.load_state_dict(best_state)
    return model


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0, "bh": 0.0}

    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n, 4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p = out["mfe_pred"].cpu().numpy()
        mae_p = out["mae_pred"].cpu().numpy()

    S = len(strat_info)
    tc = kline_15m["close"].reindex(X_test.index, method="ffill").values
    sv = kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values

    idx = np.arange(0, n - 1, 4)
    M = len(idx)
    is_long = np.array([s["dir"] == "long" for s in strat_info])
    holds = np.array([{"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(s["style"], 12) for s in strat_info])
    dirs = np.where(is_long, 1.0, -1.0)

    above = ~np.isnan(sv[idx]) & (tc[idx] > sv[idx])
    thresh = np.where(is_long[None, :], np.where(above, 0.40, 0.55)[:, None], np.where(above, 0.55, 0.40)[:, None])

    p = probs[idx]
    ev = p * np.maximum(np.abs(mfe_p[idx]), 0.001) - (1 - p) * np.maximum(np.abs(mae_p[idx]), 0.001) - fee
    valid = (p >= thresh) & (ev > 0)
    ev_masked = np.where(valid, ev, -np.inf)
    best_j = np.argmax(ev_masked, axis=1)
    has_trade = ev_masked[np.arange(M), best_j] > 0

    trade_idx = idx[has_trade]
    trade_j = best_j[has_trade]
    T = len(trade_idx)
    if T == 0:
        bh = (tc[-1] - tc[0]) / tc[0] * 100
        return {"return": 0.0, "bh": round(bh, 2), "trades": 0, "wr": 0.0, "n_long": 0, "n_short": 0}

    exit_idx = np.minimum(trade_idx + holds[trade_j], n - 1)
    net = dirs[trade_j] * (tc[exit_idx] - tc[trade_idx]) / tc[trade_idx] - fee

    capital, peak = 100000.0, 100000.0
    for i in range(T):
        dd = (peak - capital) / peak if peak > 0 else 0
        capital += net[i] * capital * 0.03 * max(0.2, 1 - dd / 0.15)
        peak = max(peak, capital)

    ret = (capital - 100000) / 100000 * 100
    bh = (tc[-1] - tc[0]) / tc[0] * 100
    return {"return": round(ret, 2), "bh": round(bh, 2), "trades": T,
            "wr": round((net > 0).mean() * 100, 1),
            "n_long": int((dirs[trade_j] > 0).sum()), "n_short": int((dirs[trade_j] < 0).sum())}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 038: Clean Labels + Feature Pruning", flush=True)
    print("=" * 60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print("\n[1/5] Loading data...", flush=True)
    kline, extras = load_data_full(start=START, end=END)

    print("\n[2/5] Building features...", flush=True)
    features = generate_features_v2(
        kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"),
        target_tf="15min", progress=False)
    top_feats = features.std().sort_values(ascending=False).head(30).index.tolist()
    seq_cols = {}
    for lag in range(1, 8):
        for col in top_feats:
            seq_cols[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in top_feats[:10]:
        for lag in [1, 2, 4]:
            seq_cols[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)
    features = pd.concat([features, pd.DataFrame(seq_cols, index=features.index)], axis=1)
    features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Feature pruning: remove near-zero variance
    n_before = features.shape[1]
    feature_std = features.std()
    keep = feature_std[feature_std > 1e-6].index
    features = features[keep]
    print(f"  Features: {n_before} → {features.shape[1]} (pruned {n_before - features.shape[1]})", flush=True)

    print("\n[3/5] Building labels (NO SCALP)...", flush=True)
    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)

    # Build labels WITHOUT scalp
    configs = {
        "no_scalp": ["intraday", "daytrade", "swing"],
        "with_scalp": ["scalp", "intraday", "daytrade", "swing"],  # baseline
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cfg_name, strategy_list in configs.items():
        print(f"\n[4/5] Testing config: {cfg_name} ({strategy_list})", flush=True)

        labels = pd.DataFrame()
        for name in strategy_list:
            if name not in lr:
                continue
            sw = lr[name].resample("15min").ffill() if name not in ["intraday"] else lr[name]
            for col in sw.columns:
                if col not in labels.columns:
                    labels[col] = sw[col]

        common = features.index.intersection(labels.index)
        X, L = features.loc[common], labels.loc[common]

        tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
        mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
        mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
        rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
        wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
        strat_info = [{"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]}
                      for c in tbm_cols]

        partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
        n = len(X)
        n_strat = len(tbm_cols)
        ws = n // 4
        X_np = X.values.astype(np.float32)
        acc = np.zeros((n, 4), dtype=np.float32)
        acc[:, 0] = 1.0
        wgt_np = L[wgt_cols].values if wgt_cols else None

        print(f"  Samples: {n}, Strategies: {n_strat}, Features: {X.shape[1]}", flush=True)

        # RAR stats
        for style in strategy_list:
            rar_s = [c for c in rar_cols if style in c]
            if rar_s:
                vals = L[rar_s].values.flatten()
                valid = ~np.isnan(vals)
                if valid.sum() > 0:
                    wr = (vals[valid] > 0).mean() * 100
                    print(f"    {style}: WR={wr:.1f}%, avgRAR={vals[valid].mean():.4f}", flush=True)

        seed_means = []
        for seed in SEEDS:
            wrets = []
            for w in range(3):
                ts = w * (ws // 3)
                te = ts + int(ws * 2)
                ve = te + int(ws * 0.5)
                test_e = min(ve + ws, n)
                if test_e <= ve: continue

                torch.manual_seed(seed)
                np.random.seed(seed)

                tds = TradingDatasetV4(X_np[ts:te], L[tbm_cols].values[ts:te], L[mae_cols].values[ts:te],
                                        L[mfe_cols].values[ts:te], L[rar_cols].values[ts:te], acc[ts:te],
                                        wgt_np[ts:te] if wgt_np is not None else None)
                vds = TradingDatasetV4(X_np[te:ve], L[tbm_cols].values[te:ve], L[mae_cols].values[te:ve],
                                        L[mfe_cols].values[te:ve], L[rar_cols].values[te:ve], acc[te:ve],
                                        wgt_np[te:ve] if wgt_np is not None else None)

                model = PLEv4(feature_partitions=partitions, n_account_features=4, n_strategies=n_strat,
                              expert_hidden=128, expert_output=64, fusion_dim=128, dropout=0.2)
                model = train_rdrop(model, tds, vds, epochs=50, device=device, patience=7, rdrop_alpha=1.0)
                r = backtest_vec(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], strat_info, device=device)
                wrets.append(r["return"])

            sm = np.mean(wrets)
            seed_means.append(sm)
            print(f"    Seed {seed}: {[round(x,1) for x in wrets]} → mean={sm:+.2f}%", flush=True)

        overall = np.mean(seed_means)
        std = np.std(seed_means)
        print(f"  {cfg_name}: overall={overall:+.2f}%, std={std:.1f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)

    report = {
        "iteration": 38,
        "approach": "Remove scalp labels (WR=31% avgRAR=-0.54) + feature pruning",
        "time_seconds": round(elapsed),
    }
    with open("reports/iteration_038.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to reports/iteration_038.json", flush=True)


if __name__ == "__main__":
    main()
