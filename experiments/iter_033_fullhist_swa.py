#!/usr/bin/env python3
"""
Iteration 033: Full-History Training (15m→1h resample) + Stochastic Weight Averaging

Key insights:
  1. 15m data spans 2020-01-01 to 2026-02-28 (6 years, 216K bars)
     1h data only spans 2024-04-01 to 2026-03-31 (2 years, 17K bars)
     → Resample 15m→1h to unlock full bull+bear market cycles

  2. SWA (Izmailov et al. 2018) averages weights over training trajectory
     → Finds wider optima that generalize better
     → Improves calibration (critical for probability outputs)
     → PyTorch native: torch.optim.swa_utils

  3. Cosine Annealing schedule (vs OneCycleLR)
     → Better exploration of loss landscape
     → Natural fit with SWA (average over cosine cycles)

Expected impact:
  - Bull market window should improve from -10% to positive
  - 3x more training data → more robust features
  - SWA → better generalization + calibration
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4
from ple.model_v3 import partition_features
from features.factory_v2 import generate_features_v2
from labeling.multi_tbm_v2 import generate_multi_tbm_v2


# ── Data Loading with 15m→1h Resample ─────────────────────────────────────

def load_data_full_history(data_dir="data/merged/BTCUSDT", start="2020-06-01", end="2026-02-28"):
    """Load data with 15m→1h resampling for full history coverage."""
    kline = {}

    # Load 15m (primary timeframe — full history)
    for tf in ["5m", "15m"]:
        path = os.path.join(data_dir, f"kline_{tf}.parquet")
        if os.path.exists(path):
            kline[tf] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]

    # 1h: prefer native file, but RESAMPLE from 15m if it covers more history
    path_1h = os.path.join(data_dir, "kline_1h.parquet")
    if os.path.exists(path_1h):
        native_1h = pd.read_parquet(path_1h).set_index("timestamp").sort_index()[start:end]
    else:
        native_1h = pd.DataFrame()

    # Resample 15m → 1h for full history
    if "15m" in kline:
        resampled_1h = kline["15m"].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        if len(resampled_1h) > len(native_1h):
            print(f"  Using resampled 1h: {len(resampled_1h)} bars (native: {len(native_1h)})")
            print(f"  Resampled range: {resampled_1h.index[0]} to {resampled_1h.index[-1]}")
            kline["1h"] = resampled_1h
        else:
            kline["1h"] = native_1h
    elif len(native_1h) > 0:
        kline["1h"] = native_1h

    # 4h: always resample from 1h
    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    # Optional data sources
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]

    return kline, extras


# ── SWA Training Loop ──────────────────────────────────────────────────────

def train_ple_v4_swa(model, train_ds, val_ds, epochs=60, batch_size=2048,
                      lr=5e-4, device="cuda", patience=10,
                      swa_start_pct=0.6, swa_lr=1e-4):
    """Train PLE v4 with Stochastic Weight Averaging.

    Training phases:
      1. Standard training with CosineAnnealingWarmRestarts (60% of epochs)
      2. SWA phase: average weights over remaining epochs (40%)

    SWA finds wider optima → better generalization + calibration.
    """
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)

    # Base optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # Phase 1: Cosine annealing with warm restarts
    swa_start_epoch = int(epochs * swa_start_pct)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(swa_start_epoch // 3, 5), T_mult=1, eta_min=swa_lr
    )

    # Phase 2: SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    best_val = float("inf")
    no_improve = 0
    best_state = None
    swa_active = False

    for epoch in range(epochs):
        # Switch to SWA phase
        if epoch >= swa_start_epoch and not swa_active:
            swa_active = True
            print(f"\n  >>> SWA phase starts at epoch {epoch+1} <<<")
            if best_state:
                model.load_state_dict(best_state)

        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Mixup (proven in iter 32)
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]

            out = model(batch["features"], batch["account"])
            losses = loss_fn(out, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update scheduler
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

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
        phase = "SWA" if swa_active else "STD"

        print(f"  E{epoch+1:02d} [{phase}]  loss={v['total']:.3f}  "
              f"bce={v['L_label']:.3f}  cal={v['L_cal']:.4f}  "
              f"active={v['n_active']:.1f}  prec={v['precision']:.2f}  "
              f"no_trade={v['no_trade_pct']:.1%}")

        if not swa_active:
            if v["total"] < best_val:
                best_val = v["total"]
                no_improve = 0
                best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience and epoch < swa_start_epoch - 1:
                    print(f"  Early stop standard phase at epoch {epoch+1}, switching to SWA")
                    swa_active = True
                    if best_state:
                        model.load_state_dict(best_state)

    # Update BN statistics for SWA model
    print("  Updating SWA batch norm statistics...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    return swa_model, best_state, best_val


# ── Walk-Forward Backtest ──────────────────────────────────────────────────

def backtest_window(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008,
                    size=0.03, device="cuda"):
    """Vectorized-ish backtest for a single window."""
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0, "bh": 0.0}

    model = model.to(device).eval()

    with torch.no_grad():
        out = model(
            torch.tensor(X_test.values.astype(np.float32)).to(device),
            torch.zeros(n, 4).to(device),
        )
        probs = out["label_probs"].cpu().numpy()
        mfe_p = out["mfe_pred"].cpu().numpy()
        mae_p = out["mae_pred"].cpu().numpy()

    # SMA filter
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

        best_ev = -1
        best_j = -1
        for j in range(len(strat_info)):
            th = lt if strat_info[j]["dir"] == "long" else st
            if probs[i, j] < th:
                continue
            p = probs[i, j]
            rew = max(abs(mfe_p[i, j]), 0.001)
            rsk = max(abs(mae_p[i, j]), 0.001)
            ev = p * rew - (1 - p) * rsk - fee
            if ev > best_ev:
                best_ev = ev
                best_j = j

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

    return {
        "return": round(ret, 2),
        "bh": round(bh, 2),
        "trades": len(tdf),
        "wr": round(wr, 1),
        "n_long": int(n_long),
        "n_short": int(n_short),
    }


# ── Main Experiment ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 033: Full-History (15m→1h) + SWA")
    print("=" * 60)

    # Use full history range — skip first 6 months for ATR warmup
    START = "2020-06-01"
    END = "2026-02-28"

    print(f"\n[1/5] Loading data ({START} to {END})...")
    kline, extras = load_data_full_history(start=START, end=END)
    for tf, df in kline.items():
        print(f"  {tf}: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    print(f"\n[2/5] Building features...")
    features = generate_features_v2(
        kline_data=kline,
        tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"),
        funding=extras.get("funding_rate"),
        target_tf="15min",
        progress=False,
    )

    # Sequence features
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
    print(f"  Features: {features.shape}")

    print(f"\n[3/5] Building TBM labels...")
    lr = generate_multi_tbm_v2(kline, fee_pct=0.0008, progress=True)

    labels = lr["intraday"].copy() if "intraday" in lr else pd.DataFrame()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
            for col in sw.columns:
                if col not in labels.columns:
                    labels[col] = sw[col]

    # Align
    common = features.index.intersection(labels.index)
    X = features.loc[common]
    L = labels.loc[common]
    print(f"  Aligned: {len(X)} rows, {L.shape[1]} label columns")

    # Feature partitions
    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    print(f"  Partitions: {list(partitions.keys())} ({sum(len(v) for v in partitions.values())} features)")

    # ── Walk-Forward Validation (3 windows) ──
    print(f"\n[4/5] Walk-forward validation (3 windows)...")
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
    print(f"  Total samples: {n}, strategies: {n_strategies}")

    # 3 rolling windows
    window_size = n // 4
    results = []

    for w in range(3):
        train_start = w * (window_size // 3)
        train_end = train_start + int(window_size * 2)
        val_end = train_end + int(window_size * 0.5)
        test_end = min(val_end + window_size, n)

        if test_end <= val_end or val_end <= train_end:
            continue

        print(f"\n  --- Window {w+1}/3 ---")
        print(f"  Train: {X.index[train_start]} to {X.index[train_end-1]} ({train_end - train_start} samples)")
        print(f"  Val:   {X.index[train_end]} to {X.index[val_end-1]} ({val_end - train_end} samples)")
        print(f"  Test:  {X.index[val_end]} to {X.index[test_end-1]} ({test_end - val_end} samples)")

        X_np = X.values.astype(np.float32)
        acc = np.zeros((n, 4), dtype=np.float32)
        acc[:, 0] = 1.0
        wgt_np = L[wgt_cols].values if wgt_cols else None

        torch.manual_seed(42 + w)
        np.random.seed(42 + w)

        train_ds = TradingDatasetV4(
            X_np[train_start:train_end],
            L[tbm_cols].values[train_start:train_end],
            L[mae_cols].values[train_start:train_end],
            L[mfe_cols].values[train_start:train_end],
            L[rar_cols].values[train_start:train_end],
            acc[train_start:train_end],
            wgt_np[train_start:train_end] if wgt_np is not None else None,
        )
        val_ds = TradingDatasetV4(
            X_np[train_end:val_end],
            L[tbm_cols].values[train_end:val_end],
            L[mae_cols].values[train_end:val_end],
            L[mfe_cols].values[train_end:val_end],
            L[rar_cols].values[train_end:val_end],
            acc[train_end:val_end],
            wgt_np[train_end:val_end] if wgt_np is not None else None,
        )

        model = PLEv4(
            feature_partitions=partitions,
            n_account_features=4,
            n_strategies=n_strategies,
            expert_hidden=128,
            expert_output=64,
            fusion_dim=128,
            dropout=0.1,
        )
        print(f"  Model: {model.count_parameters():,} params")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        swa_model, best_state, best_val = train_ple_v4_swa(
            model, train_ds, val_ds,
            epochs=60, batch_size=2048,
            lr=5e-4, device=device,
            patience=10, swa_start_pct=0.6, swa_lr=1e-4,
        )

        # Test with SWA model
        X_test = X.iloc[val_end:test_end]
        result = backtest_window(
            swa_model.module, X_test, kline["15m"], kline["4h"],
            strat_info, device=device,
        )

        # Also test with best standard model for comparison
        model.load_state_dict(best_state)
        result_std = backtest_window(
            model, X_test, kline["15m"], kline["4h"],
            strat_info, device=device,
        )

        print(f"\n  Window {w+1} Results:")
        print(f"    SWA:      {result['return']:+.2f}% ({result['trades']} trades, "
              f"WR={result['wr']:.1f}%, L={result['n_long']} S={result['n_short']})")
        print(f"    Standard: {result_std['return']:+.2f}% ({result_std['trades']} trades)")
        print(f"    B&H:      {result['bh']:+.2f}%")

        results.append({
            "window": w + 1,
            "swa": result,
            "standard": result_std,
            "period": f"{X.index[val_end]} to {X.index[test_end-1]}",
        })

    # ── Summary ──
    elapsed = time.time() - t0
    swa_returns = [r["swa"]["return"] for r in results]
    std_returns = [r["standard"]["return"] for r in results]
    swa_mean = np.mean(swa_returns)
    std_mean = np.mean(std_returns)

    print(f"\n{'=' * 60}")
    print(f"  ITERATION 033 RESULTS")
    print(f"{'=' * 60}")
    print(f"  SWA Returns:      {swa_returns} → mean={swa_mean:+.2f}%")
    print(f"  Standard Returns: {std_returns} → mean={std_mean:+.2f}%")
    print(f"  SWA Improvement:  {swa_mean - std_mean:+.2f}%")
    print(f"  Time: {elapsed:.0f}s")

    for i, r in enumerate(results):
        period_start = r["period"].split(" to ")[0][:10]
        period_end = r["period"].split(" to ")[1][:10]
        bh = r["swa"]["bh"]
        market = "BULL" if bh > 10 else ("BEAR" if bh < -10 else "SIDEWAYS")
        print(f"\n  W{i+1} ({period_start} ~ {period_end}) [{market}]")
        print(f"    SWA: {r['swa']['return']:+.2f}%  Std: {r['standard']['return']:+.2f}%  B&H: {bh:+.2f}%")
        print(f"    Trades: {r['swa']['trades']} (L:{r['swa']['n_long']} S:{r['swa']['n_short']})")

    # Save report
    report = {
        "iteration": 33,
        "approach": "Full-history (15m→1h resample, 2020-2026) + SWA (Stochastic Weight Averaging)",
        "data_range": f"{START} to {END}",
        "windows_swa": swa_returns,
        "windows_std": std_returns,
        "mean_swa": round(swa_mean, 2),
        "mean_std": round(std_mean, 2),
        "improvement": round(swa_mean - std_mean, 2),
        "details": results,
        "time_seconds": round(elapsed),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/iteration_033.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to reports/iteration_033.json")


if __name__ == "__main__":
    main()
