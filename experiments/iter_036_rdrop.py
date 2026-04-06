#!/usr/bin/env python3
"""
Iteration 036: R-Drop Regularization + Stronger Dropout

Problem from Iter 035:
  - Seed 42: +33.30%, Seed 123: -2.16%, Seed 777: +19.12%
  - 35% variance between seeds = overfitting to different minima
  - Model memorizes training noise differently per seed

R-Drop (Liang et al. 2021, Microsoft):
  - Two forward passes with different dropout masks
  - KL divergence loss forces consistency: P1 ≈ P2
  - If model is robust, different dropout should give same prediction
  - Proven effective on classification tasks (+1-3% across benchmarks)

  L_rdrop = L_task + α × (KL(P1||P2) + KL(P2||P1)) / 2

Additional changes:
  - Dropout 0.1 → 0.2 (stronger regularization)
  - α = 0.5 for R-Drop weight (tuned for classification)

Expected:
  - Lower seed variance (target: <5% between seeds)
  - Slightly lower peak but higher floor
  - More consistent walk-forward windows
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    """Symmetric KL divergence for binary probabilities with mask."""
    eps = 1e-7
    p1 = p1.clamp(eps, 1 - eps)
    p2 = p2.clamp(eps, 1 - eps)

    kl_1 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
    kl_2 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())

    kl = (kl_1 + kl_2) / 2  # symmetric
    return (kl * mask).sum() / mask.sum().clamp(1)


def train_ple_v4_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                        lr=5e-4, device="cuda", patience=7, rdrop_alpha=0.5):
    """Train PLE v4 with R-Drop regularization."""
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * (len(train_ds) // batch_size + 1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    best_val = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_rdrop = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Mixup
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]

            # R-Drop: two forward passes with different dropout masks
            out1 = model(batch["features"], batch["account"])
            out2 = model(batch["features"], batch["account"])

            # Standard losses from both passes
            losses1 = loss_fn(out1, batch)
            losses2 = loss_fn(out2, batch)
            task_loss = (losses1["total"] + losses2["total"]) / 2

            # R-Drop KL divergence on label probabilities
            rdrop_loss = kl_divergence_binary(
                out1["label_probs"], out2["label_probs"], batch["rar_mask"])

            total = task_loss + rdrop_alpha * rdrop_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_rdrop += rdrop_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch["features"], batch["account"])
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

        v = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        avg_rdrop = epoch_rdrop / max(n_batches, 1)

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  bce={v['L_label']:.3f}  "
              f"rdrop={avg_rdrop:.4f}  active={v['n_active']:.1f}  "
              f"prec={v['precision']:.2f}  no_trade={v['no_trade_pct']:.1%}")

        if v["total"] < best_val:
            best_val = v["total"]
            no_improve = 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def backtest(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008,
             size=0.03, device="cuda"):
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

    capital = 100000.0
    peak = capital
    trades = []

    for i in range(0, n - 1, 4):
        above = not np.isnan(sv[i]) and tc[i] > sv[i]
        lt = 0.40 if above else 0.55
        st = 0.55 if above else 0.40

        best_ev, best_j = -1, -1
        for j in range(len(strat_info)):
            th = lt if strat_info[j]["dir"] == "long" else st
            if probs[i, j] < th:
                continue
            p = probs[i, j]
            rew = max(abs(mfe_p[i, j]), 0.001)
            rsk = max(abs(mae_p[i, j]), 0.001)
            ev = p * rew - (1 - p) * rsk - fee
            if ev > best_ev:
                best_ev, best_j = ev, j

        if best_j < 0 or best_ev <= 0:
            continue

        d = 1 if strat_info[best_j]["dir"] == "long" else -1
        hold = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(
            strat_info[best_j]["style"], 12)
        ei = min(i + hold, n - 1)
        pnl = d * (tc[ei] - tc[i]) / tc[i]
        net = pnl - fee

        dd = (peak - capital) / peak if peak > 0 else 0
        sz = size * max(0.2, 1 - dd / 0.15)
        capital += net * capital * sz
        peak = max(peak, capital)
        trades.append({"net": net * 100, "dir": "L" if d == 1 else "S"})

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame({"net": [], "dir": []})
    ret = (capital - 100000) / 100000 * 100
    bh = (tc[-1] - tc[0]) / tc[0] * 100 if tc[0] > 0 else 0
    wr = (tdf["net"] > 0).mean() * 100 if len(tdf) > 0 else 0
    n_long = (tdf["dir"] == "L").sum() if len(tdf) > 0 else 0
    n_short = (tdf["dir"] == "S").sum() if len(tdf) > 0 else 0

    return {"return": round(ret, 2), "bh": round(bh, 2), "trades": len(tdf),
            "wr": round(wr, 1), "n_long": int(n_long), "n_short": int(n_short)}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 036: R-Drop + Dropout 0.2")
    print("=" * 60)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print(f"\n[1/4] Loading data + features + labels...")
    kline, extras = load_data_full(start=START, end=END)

    features = generate_features_v2(
        kline_data=kline, tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"), funding=extras.get("funding_rate"),
        target_tf="15min", progress=False,
    )
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
    X = features.loc[common]
    L = labels.loc[common]
    print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features, {L.shape[1]} labels")

    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])

    strat_info = []
    for c in tbm_cols:
        parts = c.replace("tbm_", "").split("_")
        strat_info.append({"style": parts[0], "dir": parts[1]})

    n = len(X)
    n_strategies = len(tbm_cols)
    window_size = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test both: R-Drop (dropout=0.2) vs baseline (dropout=0.1)
    configs = {
        "rdrop_d02": {"dropout": 0.2, "rdrop_alpha": 0.5},
        "rdrop_d01": {"dropout": 0.1, "rdrop_alpha": 0.5},
        "baseline_d02": {"dropout": 0.2, "rdrop_alpha": 0.0},
    }

    print(f"\n[2/4] Walk-forward with {len(SEEDS)} seeds × {len(configs)} configs...")

    all_results = {}
    for cfg_name in configs:
        all_results[cfg_name] = {s: [] for s in SEEDS}

    for w in range(3):
        train_start = w * (window_size // 3)
        train_end = train_start + int(window_size * 2)
        val_end = train_end + int(window_size * 0.5)
        test_end = min(val_end + window_size, n)
        if test_end <= val_end:
            continue

        X_test = X.iloc[val_end:test_end]
        X_np = X.values.astype(np.float32)
        acc = np.zeros((n, 4), dtype=np.float32)
        acc[:, 0] = 1.0
        wgt_np = L[wgt_cols].values if wgt_cols else None

        print(f"\n  --- Window {w+1}/3 ---")

        for cfg_name, cfg in configs.items():
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                train_ds = TradingDatasetV4(
                    X_np[train_start:train_end], L[tbm_cols].values[train_start:train_end],
                    L[mae_cols].values[train_start:train_end], L[mfe_cols].values[train_start:train_end],
                    L[rar_cols].values[train_start:train_end], acc[train_start:train_end],
                    wgt_np[train_start:train_end] if wgt_np is not None else None,
                )
                val_ds = TradingDatasetV4(
                    X_np[train_end:val_end], L[tbm_cols].values[train_end:val_end],
                    L[mae_cols].values[train_end:val_end], L[mfe_cols].values[train_end:val_end],
                    L[rar_cols].values[train_end:val_end], acc[train_end:val_end],
                    wgt_np[train_end:val_end] if wgt_np is not None else None,
                )

                model = PLEv4(
                    feature_partitions=partitions, n_account_features=4,
                    n_strategies=n_strategies, expert_hidden=128, expert_output=64,
                    fusion_dim=128, dropout=cfg["dropout"],
                )

                alpha = cfg["rdrop_alpha"]
                if alpha > 0:
                    print(f"    {cfg_name} seed={seed} (R-Drop α={alpha}, d={cfg['dropout']})...")
                    model = train_ple_v4_rdrop(
                        model, train_ds, val_ds, epochs=50, batch_size=2048,
                        lr=5e-4, device=device, patience=7, rdrop_alpha=alpha)
                else:
                    from ple.trainer_v4 import train_ple_v4
                    print(f"    {cfg_name} seed={seed} (no R-Drop, d={cfg['dropout']})...")
                    train_ple_v4(model, train_ds, val_ds, epochs=50, batch_size=2048,
                                  lr=5e-4, device=device, patience=7)

                result = backtest(model, X_test, kline["15m"], kline["4h"],
                                  strat_info, device=device)
                all_results[cfg_name][seed].append(result)
                print(f"      → {result['return']:+.2f}% ({result['trades']} trades, WR={result['wr']:.1f}%)")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ITERATION 036 RESULTS")
    print(f"{'=' * 60}")

    best_cfg = None
    best_score = -999

    for cfg_name in configs:
        seed_means = []
        print(f"\n  {cfg_name} (dropout={configs[cfg_name]['dropout']}, α={configs[cfg_name]['rdrop_alpha']}):")
        for seed in SEEDS:
            rets = [r["return"] for r in all_results[cfg_name][seed]]
            mean = np.mean(rets)
            seed_means.append(mean)
            print(f"    Seed {seed}: {rets} → mean={mean:+.2f}%")

        overall_mean = np.mean(seed_means)
        seed_std = np.std(seed_means)
        print(f"    Overall: mean={overall_mean:+.2f}%, seed_std={seed_std:.2f}%")

        # Score: mean return minus seed variance penalty
        score = overall_mean - seed_std * 0.5
        if score > best_score:
            best_score = score
            best_cfg = cfg_name

    print(f"\n  BEST CONFIG: {best_cfg} (score={best_score:+.2f})")
    print(f"  Time: {elapsed:.0f}s")

    # Save report
    report = {
        "iteration": 36,
        "approach": "R-Drop regularization + dropout tuning",
        "data_range": f"{START} to {END}",
        "best_config": best_cfg,
        "configs": {},
    }
    for cfg_name in configs:
        seed_results = {}
        for seed in SEEDS:
            rets = [r["return"] for r in all_results[cfg_name][seed]]
            seed_results[str(seed)] = {"windows": rets, "mean": round(np.mean(rets), 2)}
        all_means = [v["mean"] for v in seed_results.values()]
        report["configs"][cfg_name] = {
            "params": configs[cfg_name],
            "seeds": seed_results,
            "overall_mean": round(np.mean(all_means), 2),
            "seed_std": round(np.std(all_means), 2),
        }

    best_means = [v["mean"] for v in report["configs"][best_cfg]["seeds"].values()]
    report["mean"] = round(np.mean(best_means), 2)
    report["windows"] = "see per-seed results"

    with open("reports/iteration_036.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to reports/iteration_036.json")


if __name__ == "__main__":
    main()
