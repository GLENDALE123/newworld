#!/usr/bin/env python3
"""
Iteration 037: Dropout + R-Drop Alpha Sweep

From Iter 036:
  - d=0.2 + α=0.5: +35.96% (best)
  - d=0.2 + α=0.0: +33.13%
  - d=0.1 + α=0.5: +18.87%

Sweep:
  dropout: [0.15, 0.20, 0.25, 0.30]
  alpha:   [0.0, 0.3, 0.5, 1.0]

Using seed 42 only (most stable) + window 1 only for fast screening.
Then validate best 3 configs across all 3 windows.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4
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


def kl_divergence_binary(p1, p2, mask):
    eps = 1e-7
    p1, p2 = p1.clamp(eps, 1 - eps), p2.clamp(eps, 1 - eps)
    kl_1 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
    kl_2 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())
    return ((kl_1 + kl_2) / 2 * mask).sum() / mask.sum().clamp(1)


def train_with_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                      lr=5e-4, device="cuda", patience=7, rdrop_alpha=0.5):
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

            if rdrop_alpha > 0:
                out1 = model(batch["features"], batch["account"])
                out2 = model(batch["features"], batch["account"])
                l1 = loss_fn(out1, batch)
                l2 = loss_fn(out2, batch)
                task = (l1["total"] + l2["total"]) / 2
                rdrop = kl_divergence_binary(out1["label_probs"], out2["label_probs"], batch["rar_mask"])
                total = task + rdrop_alpha * rdrop
            else:
                out = model(batch["features"], batch["account"])
                losses = loss_fn(out, batch)
                total = losses["total"]

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch["features"], batch["account"])
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

        v = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        if v["total"] < best_val:
            best_val, no_improve = v["total"], 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def backtest(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0}

    model = model.to(device).eval()
    with torch.no_grad():
        out = model(
            torch.tensor(X_test.values.astype(np.float32)).to(device),
            torch.zeros(n, 4).to(device),
        )
        probs = out["label_probs"].cpu().numpy()
        mfe_p = out["mfe_pred"].cpu().numpy()
        mae_p = out["mae_pred"].cpu().numpy()

    close = kline_15m["close"]
    sma = kline_4h["close"].rolling(50).mean().resample("15min").ffill()
    tc = close.reindex(X_test.index, method="ffill").values
    sv = sma.reindex(X_test.index, method="ffill").values

    capital, peak, trades = 100000.0, 100000.0, []
    for i in range(0, n - 1, 4):
        above = not np.isnan(sv[i]) and tc[i] > sv[i]
        lt = 0.40 if above else 0.55
        st = 0.55 if above else 0.40
        best_ev, best_j = -1, -1
        for j in range(len(strat_info)):
            th = lt if strat_info[j]["dir"] == "long" else st
            if probs[i, j] < th: continue
            p = probs[i, j]
            ev = p * max(abs(mfe_p[i, j]), 0.001) - (1 - p) * max(abs(mae_p[i, j]), 0.001) - fee
            if ev > best_ev: best_ev, best_j = ev, j
        if best_j < 0 or best_ev <= 0: continue
        d = 1 if strat_info[best_j]["dir"] == "long" else -1
        hold = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(strat_info[best_j]["style"], 12)
        ei = min(i + hold, n - 1)
        net = d * (tc[ei] - tc[i]) / tc[i] - fee
        dd = (peak - capital) / peak if peak > 0 else 0
        sz = 0.03 * max(0.2, 1 - dd / 0.15)
        capital += net * capital * sz
        peak = max(peak, capital)
        trades.append({"net": net * 100, "dir": "L" if d == 1 else "S"})

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame({"net": [], "dir": []})
    ret = (capital - 100000) / 100000 * 100
    bh = (tc[-1] - tc[0]) / tc[0] * 100
    wr = (tdf["net"] > 0).mean() * 100 if len(tdf) > 0 else 0
    n_long = (tdf["dir"] == "L").sum() if len(tdf) > 0 else 0
    n_short = (tdf["dir"] == "S").sum() if len(tdf) > 0 else 0
    return {"return": round(ret, 2), "bh": round(bh, 2), "trades": len(tdf),
            "wr": round(wr, 1), "n_long": int(n_long), "n_short": int(n_short)}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 037: Dropout + R-Drop Alpha Sweep")
    print("=" * 60)

    START, END = "2020-06-01", "2026-02-28"

    print(f"\n[1/3] Loading data + features + labels...")
    kline, extras = load_data_full(start=START, end=END)
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

    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=False)
    labels = lr["intraday"].copy() if "intraday" in lr else pd.DataFrame()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
            for col in sw.columns:
                if col not in labels.columns:
                    labels[col] = sw[col]

    common = features.index.intersection(labels.index)
    X, L = features.loc[common], labels.loc[common]
    print(f"  Data: {X.shape}")

    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    strat_info = [{"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]}
                  for c in tbm_cols]

    n = len(X)
    n_strategies = len(tbm_cols)
    window_size = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase 1: Fast screen with seed 42, window 1 only
    DROPOUTS = [0.15, 0.20, 0.25, 0.30]
    ALPHAS = [0.0, 0.3, 0.5, 1.0]

    w = 0
    train_start = 0
    train_end = int(window_size * 2)
    val_end = train_end + int(window_size * 0.5)
    test_end = min(val_end + window_size, n)

    X_test = X.iloc[val_end:test_end]
    X_np = X.values.astype(np.float32)
    acc = np.zeros((n, 4), dtype=np.float32)
    acc[:, 0] = 1.0
    wgt_np = L[wgt_cols].values if wgt_cols else None

    print(f"\n[2/3] Phase 1: Fast screen (seed 42, window 1, {len(DROPOUTS)*len(ALPHAS)} configs)...")
    screen_results = {}

    for d in DROPOUTS:
        for a in ALPHAS:
            name = f"d{d:.2f}_a{a:.1f}"
            torch.manual_seed(42)
            np.random.seed(42)

            train_ds = TradingDatasetV4(
                X_np[train_start:train_end], L[tbm_cols].values[train_start:train_end],
                L[mae_cols].values[train_start:train_end], L[mfe_cols].values[train_start:train_end],
                L[rar_cols].values[train_start:train_end], acc[train_start:train_end],
                wgt_np[train_start:train_end] if wgt_np is not None else None)
            val_ds = TradingDatasetV4(
                X_np[train_end:val_end], L[tbm_cols].values[train_end:val_end],
                L[mae_cols].values[train_end:val_end], L[mfe_cols].values[train_end:val_end],
                L[rar_cols].values[train_end:val_end], acc[train_end:val_end],
                wgt_np[train_end:val_end] if wgt_np is not None else None)

            model = PLEv4(feature_partitions=partitions, n_account_features=4,
                          n_strategies=n_strategies, expert_hidden=128, expert_output=64,
                          fusion_dim=128, dropout=d)
            model = train_with_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                                      lr=5e-4, device=device, patience=7, rdrop_alpha=a)
            result = backtest(model, X_test, kline["15m"], kline["4h"], strat_info, device=device)
            screen_results[name] = {"dropout": d, "alpha": a, "w1": result}
            print(f"    {name}: {result['return']:+7.2f}%  trades={result['trades']}  WR={result['wr']:.1f}%")

    # Phase 2: Top 3 configs across all 3 windows with 3 seeds
    top3 = sorted(screen_results.items(), key=lambda x: x[1]["w1"]["return"], reverse=True)[:3]
    print(f"\n[3/3] Phase 2: Top 3 validation (3 windows × 3 seeds)...")
    SEEDS = [42, 123, 777]

    final_results = {}
    for name, cfg in top3:
        print(f"\n  Validating: {name} (d={cfg['dropout']}, α={cfg['alpha']})")
        seed_means = []

        for seed in SEEDS:
            window_rets = []
            for w in range(3):
                ts = w * (window_size // 3)
                te = ts + int(window_size * 2)
                ve = te + int(window_size * 0.5)
                test_e = min(ve + window_size, n)
                if test_e <= ve: continue

                torch.manual_seed(seed)
                np.random.seed(seed)

                train_ds = TradingDatasetV4(
                    X_np[ts:te], L[tbm_cols].values[ts:te], L[mae_cols].values[ts:te],
                    L[mfe_cols].values[ts:te], L[rar_cols].values[ts:te], acc[ts:te],
                    wgt_np[ts:te] if wgt_np is not None else None)
                val_ds = TradingDatasetV4(
                    X_np[te:ve], L[tbm_cols].values[te:ve], L[mae_cols].values[te:ve],
                    L[mfe_cols].values[te:ve], L[rar_cols].values[te:ve], acc[te:ve],
                    wgt_np[te:ve] if wgt_np is not None else None)

                model = PLEv4(feature_partitions=partitions, n_account_features=4,
                              n_strategies=n_strategies, expert_hidden=128, expert_output=64,
                              fusion_dim=128, dropout=cfg["dropout"])
                model = train_with_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                                          lr=5e-4, device=device, patience=7, rdrop_alpha=cfg["alpha"])
                X_w = X.iloc[ve:test_e]
                result = backtest(model, X_w, kline["15m"], kline["4h"], strat_info, device=device)
                window_rets.append(result["return"])

            mean = np.mean(window_rets)
            seed_means.append(mean)
            print(f"    Seed {seed}: {window_rets} → mean={mean:+.2f}%")

        overall = np.mean(seed_means)
        std = np.std(seed_means)
        final_results[name] = {
            "dropout": cfg["dropout"], "alpha": cfg["alpha"],
            "overall_mean": round(overall, 2), "seed_std": round(std, 2),
            "seed_means": {str(s): round(m, 2) for s, m in zip(SEEDS, seed_means)},
        }
        print(f"    Overall: mean={overall:+.2f}%, std={std:.2f}%")

    # Summary
    elapsed = time.time() - t0
    best = max(final_results.items(), key=lambda x: x[1]["overall_mean"])

    print(f"\n{'=' * 60}")
    print(f"  ITERATION 037 RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Screen results (seed 42, window 1):")
    for name, cfg in sorted(screen_results.items(), key=lambda x: x[1]["w1"]["return"], reverse=True):
        print(f"    {name}: {cfg['w1']['return']:+7.2f}%")

    print(f"\n  Final validation (3 windows × 3 seeds):")
    for name, cfg in final_results.items():
        print(f"    {name}: mean={cfg['overall_mean']:+.2f}%, std={cfg['seed_std']:.2f}%")

    print(f"\n  BEST: {best[0]} (d={best[1]['dropout']}, α={best[1]['alpha']}, "
          f"mean={best[1]['overall_mean']:+.2f}%)")
    print(f"  Time: {elapsed:.0f}s")

    report = {
        "iteration": 37,
        "approach": "Dropout + R-Drop alpha sweep (16 configs screen → top 3 full validation)",
        "screen_results": {k: {"dropout": v["dropout"], "alpha": v["alpha"],
                               "w1_return": v["w1"]["return"]}
                           for k, v in screen_results.items()},
        "final_results": final_results,
        "best_config": best[0],
        "best_dropout": best[1]["dropout"],
        "best_alpha": best[1]["alpha"],
        "mean": best[1]["overall_mean"],
    }

    with open("reports/iteration_037.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to reports/iteration_037.json")


if __name__ == "__main__":
    main()
