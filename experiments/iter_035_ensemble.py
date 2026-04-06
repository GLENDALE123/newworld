#!/usr/bin/env python3
"""
Iteration 035: Multi-Seed Ensemble (3 Models)

Rationale:
  - Single model precision: 0.21-0.23 → room for improvement
  - Different seeds explore different loss landscape regions
  - Averaging probabilities reduces noise and improves calibration
  - Proven technique in ML competitions and production systems

Setup:
  - Train 3 models with seeds [42, 123, 777]
  - Average probability outputs at inference
  - Compare: single best vs ensemble

Expected improvement:
  - More stable predictions → fewer false signals
  - Better calibration → confidence output more useful
  - Smoother equity curve → lower drawdown
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, train_ple_v4
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


def backtest(probs, mfe_p, mae_p, conf, X_test, kline_15m, kline_4h,
             strat_info, fee=0.0008, size=0.03):
    """Backtest with pre-computed predictions."""
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0}

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

    return {"return": round(ret, 2), "bh": round(bh, 2), "trades": len(tdf),
            "wr": round(wr, 1), "n_long": int(n_long), "n_short": int(n_short)}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 035: Multi-Seed Ensemble (3 Models)")
    print("=" * 60)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print(f"\n[1/5] Loading data...")
    kline, extras = load_data_full(start=START, end=END)

    print(f"\n[2/5] Building features...")
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
    print(f"  Features: {features.shape}")

    print(f"\n[3/5] Building labels...")
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
    print(f"  Aligned: {len(X)} rows")

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

    print(f"\n[4/5] Walk-forward with {len(SEEDS)}-seed ensemble...")
    results_single = {s: [] for s in SEEDS}
    results_ensemble = []

    for w in range(3):
        train_start = w * (window_size // 3)
        train_end = train_start + int(window_size * 2)
        val_end = train_end + int(window_size * 0.5)
        test_end = min(val_end + window_size, n)
        if test_end <= val_end:
            continue

        X_test = X.iloc[val_end:test_end]
        n_test = len(X_test)

        print(f"\n  --- Window {w+1}/3 ({n_test} test samples) ---")

        X_np = X.values.astype(np.float32)
        acc = np.zeros((n, 4), dtype=np.float32)
        acc[:, 0] = 1.0
        wgt_np = L[wgt_cols].values if wgt_cols else None

        all_probs = []
        all_mfe = []
        all_mae = []
        all_conf = []

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
                fusion_dim=128, dropout=0.1,
            )

            print(f"    Training seed {seed}...")
            train_ple_v4(model, train_ds, val_ds, epochs=50, batch_size=2048,
                          lr=5e-4, device=device, patience=7)

            # Inference
            model = model.to(device).eval()
            with torch.no_grad():
                out = model(
                    torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n_test, 4).to(device),
                )
                probs = out["label_probs"].cpu().numpy()
                mfe_p = out["mfe_pred"].cpu().numpy()
                mae_p = out["mae_pred"].cpu().numpy()
                conf = out["confidence"].cpu().numpy()

            all_probs.append(probs)
            all_mfe.append(mfe_p)
            all_mae.append(mae_p)
            all_conf.append(conf)

            # Single-seed backtest
            result = backtest(probs, mfe_p, mae_p, conf, X_test,
                              kline["15m"], kline["4h"], strat_info)
            results_single[seed].append(result)
            print(f"      Seed {seed}: {result['return']:+.2f}% ({result['trades']} trades, WR={result['wr']:.1f}%)")

        # Ensemble: average predictions
        ens_probs = np.mean(all_probs, axis=0)
        ens_mfe = np.mean(all_mfe, axis=0)
        ens_mae = np.mean(all_mae, axis=0)
        ens_conf = np.mean(all_conf, axis=0)

        # Ensemble backtest
        ens_result = backtest(ens_probs, ens_mfe, ens_mae, ens_conf, X_test,
                              kline["15m"], kline["4h"], strat_info)
        results_ensemble.append(ens_result)

        bh = ens_result["bh"]
        market = "BULL" if bh > 10 else ("BEAR" if bh < -10 else "SIDE")
        print(f"\n    ENSEMBLE: {ens_result['return']:+.2f}% ({ens_result['trades']} trades, "
              f"WR={ens_result['wr']:.1f}%, L={ens_result['n_long']} S={ens_result['n_short']}) [{market}]")

        # Prediction agreement analysis
        agree = 0
        total = 0
        for i in range(0, n_test, 4):
            for j in range(n_strategies):
                votes = sum(1 for p in all_probs if p[i, j] > 0.5)
                total += 1
                if votes == 0 or votes == len(SEEDS):
                    agree += 1
        print(f"    Agreement: {agree/total*100:.1f}% ({agree}/{total} predictions unanimous)")

    # Summary
    elapsed = time.time() - t0
    ens_returns = [r["return"] for r in results_ensemble]
    ens_mean = np.mean(ens_returns)

    print(f"\n{'=' * 60}")
    print(f"  ITERATION 035 RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  Ensemble ({len(SEEDS)} seeds):")
    print(f"    Windows: {ens_returns}")
    print(f"    Mean: {ens_mean:+.2f}%")

    for seed in SEEDS:
        rets = [r["return"] for r in results_single[seed]]
        print(f"\n  Single seed {seed}:")
        print(f"    Windows: {rets}")
        print(f"    Mean: {np.mean(rets):+.2f}%")

    best_single = max(SEEDS, key=lambda s: np.mean([r["return"] for r in results_single[s]]))
    best_single_mean = np.mean([r["return"] for r in results_single[best_single]])
    print(f"\n  Best single: seed {best_single} ({best_single_mean:+.2f}%)")
    print(f"  Ensemble vs best single: {ens_mean - best_single_mean:+.2f}%")
    print(f"  Time: {elapsed:.0f}s")

    # Save report
    report = {
        "iteration": 35,
        "approach": f"Multi-seed ensemble ({len(SEEDS)} seeds: {SEEDS})",
        "data_range": f"{START} to {END}",
        "windows": ens_returns,
        "mean": round(ens_mean, 2),
        "single_seed_results": {
            str(s): {"windows": [r["return"] for r in results_single[s]],
                     "mean": round(np.mean([r["return"] for r in results_single[s]]), 2)}
            for s in SEEDS
        },
        "ensemble_details": results_ensemble,
        "time_seconds": round(elapsed),
    }

    with open("reports/iteration_035.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to reports/iteration_035.json")


if __name__ == "__main__":
    main()
