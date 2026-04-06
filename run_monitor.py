#!/usr/bin/env python3
"""
ultraTM Alpha Monitor — Check if alpha is still alive

Loads saved models, tests on most recent data window.
Flags coins where alpha has decayed below threshold.

Usage:
  python run_monitor.py              # check all production coins
  python run_monitor.py --threshold 0.05  # minimum alpha threshold (%)
"""

import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strategy.portfolio_strategy import PortfolioSignalGenerator
from ultrathink.pipeline import UltraThink
from ple.model_v3 import partition_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/production_v5")
    parser.add_argument("--threshold", type=float, default=0.05, help="Min alpha %/trade")
    parser.add_argument("--window", default="2026-01-01", help="Test from this date")
    args = parser.parse_args()

    ut = UltraThink()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    holds_map = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}

    config_path = Path(args.model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    coins = config.get("coins", [])
    print(f"ultraTM Alpha Monitor — {len(coins)} coins")
    print(f"Test window: {args.window} ~ latest")
    print(f"Threshold: {args.threshold}%/trade\n")

    healthy = []
    decayed = []

    for sym in coins:
        model_path = Path(args.model_dir) / f"{sym.lower()}.pt"
        if not model_path.exists():
            print(f"  {sym}: NO MODEL")
            continue

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        pt = checkpoint["partitions"]
        ns = checkpoint["n_strategies"]
        si = checkpoint["strat_info"]
        feat_cols = checkpoint["feature_cols"]

        from ple.model_v4 import PLEv4
        model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                      expert_hidden=256, expert_output=128, fusion_dim=256,
                      dropout=0.2, use_vsn=False)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device).eval()

        try:
            X, L, kline, _ = ut.prepare(sym, args.window, "2026-12-31")
        except Exception:
            print(f"  {sym}: NO DATA")
            continue

        # Align features
        missing = [c for c in feat_cols if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        X = X[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)

        if len(X) < 500:
            print(f"  {sym}: insufficient data ({len(X)} rows)")
            continue

        with torch.no_grad():
            out = model(torch.tensor(X.values.astype(np.float32)).to(device),
                        torch.zeros(len(X), 4).to(device))
            probs = out["label_probs"].cpu().numpy()
            mfp = out["mfe_pred"].cpu().numpy()
            map_ = out["mae_pred"].cpu().numpy()

        tc_p = kline["15m"]["close"].reindex(X.index, method="ffill").values
        sv = kline["4h"]["close"].rolling(50).mean().resample("15min").ffill().reindex(
            X.index, method="ffill").values
        n = len(X)
        idx = np.arange(0, n - 1, 4)
        il = np.array([s["dir"] == "long" for s in si])
        hs = np.array([holds_map.get(s["style"], 12) for s in si])
        ds = np.where(il, 1.0, -1.0)
        ab = ~np.isnan(sv[idx]) & (tc_p[idx] > sv[idx])
        th = np.where(il[None, :], np.where(ab, 0.40, 0.55)[:, None],
                      np.where(ab, 0.55, 0.40)[:, None])
        p = probs[idx]
        ev = p * np.maximum(np.abs(mfp[idx]), 0.001) - (1 - p) * np.maximum(
            np.abs(map_[idx]), 0.001) - 0.0008
        vd = (p >= th) & (ev > 0)
        em = np.where(vd, ev, -np.inf)
        bj = np.argmax(em, axis=1)
        ht = em[np.arange(len(idx)), bj] > 0
        ti, tj = idx[ht], bj[ht]
        T = len(ti)

        if T < 20:
            print(f"  {sym}: {T} trades (too few)")
            continue

        ei = np.minimum(ti + hs[tj], n - 1)
        gross = ds[tj] * (tc_p[ei] - tc_p[ti]) / tc_p[ti]
        avg = gross.mean() * 100

        status = "✓ HEALTHY" if avg > args.threshold else "⚠ DECAYED"
        if avg > args.threshold:
            healthy.append(sym)
        else:
            decayed.append(sym)

        print(f"  {sym:12s}: {status}  alpha={avg:+.3f}%  trades={T}")

    print(f"\nSummary: {len(healthy)} healthy, {len(decayed)} decayed")
    if decayed:
        print(f"  ⚠ Consider retraining: {decayed}")


if __name__ == "__main__":
    main()
