#!/usr/bin/env python3
"""
ultraTM Auto-Retrain — Monthly model refresh

Alpha decays ~0.18%/month (iter 092). Monthly retraining maintains edge.
Uses latest data to train fresh models for all production coins.

Usage:
  python run_retrain.py                    # retrain all 5 coins
  python run_retrain.py --coins UNI BCH    # specific coins

Cron: 0 0 1 * * cd /path/to/ultraTM && python run_retrain.py
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultrathink.pipeline import UltraThink
from ple.model_v4 import PLEv4
from ple.trainer_v4 import TradingDatasetV4, train_ple_v4
from ple.model_v3 import partition_features

TOP5 = ["UNIUSDT", "DUSKUSDT", "CHRUSDT", "BCHUSDT", "XRPUSDT"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=TOP5)
    parser.add_argument("--output", default="models/production_v5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ut = UltraThink()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    print(f"ultraTM Retrain — {len(args.coins)} coins")
    print(f"Output: {args.output}\n")

    results = {}
    for sym in args.coins:
        sym = sym if sym.endswith("USDT") else sym + "USDT"
        t0 = time.time()

        try:
            X, L, kline, si = ut.prepare(sym, "2023-06-01", "2026-04-04")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")
            continue

        pt = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
        tc = sorted([c for c in L.columns if c.startswith("tbm_")])
        mc = sorted([c for c in L.columns if c.startswith("mae_")])
        fc = sorted([c for c in L.columns if c.startswith("mfe_")])
        rc = sorted([c for c in L.columns if c.startswith("rar_")])
        wc = sorted([c for c in L.columns if c.startswith("wgt_")])

        if not tc:
            continue

        ns = len(tc)
        n = len(X)
        s1 = int(n * 0.8)  # train on 80%

        Xn = X.values.astype(np.float32)
        ac = np.zeros((n, 4), dtype=np.float32)
        ac[:, 0] = 1.0
        wn = L[wc].values if wc else None

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        tds = TradingDatasetV4(Xn[:s1], L[tc].values[:s1], L[mc].values[:s1],
                                L[fc].values[:s1], L[rc].values[:s1], ac[:s1],
                                wn[:s1] if wn is not None else None)
        vds = TradingDatasetV4(Xn[s1:], L[tc].values[s1:], L[mc].values[s1:],
                                L[fc].values[s1:], L[rc].values[s1:], ac[s1:],
                                wn[s1:] if wn is not None else None)

        model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                      expert_hidden=256, expert_output=128, fusion_dim=256,
                      dropout=0.2, use_vsn=False)

        train_ple_v4(model, tds, vds, epochs=50, batch_size=2048,
                      device=device, patience=7, rdrop_alpha=1.0, seed=args.seed)

        path = os.path.join(args.output, f"{sym.lower()}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "partitions": pt,
            "n_strategies": ns,
            "feature_cols": list(X.columns),
            "strat_info": si,
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "data_end": str(X.index[-1]),
            "n_samples": n,
        }, path)

        dt = time.time() - t0
        results[sym] = {"params": model.count_parameters(), "time": round(dt)}
        print(f"  {sym}: saved ({model.count_parameters():,} params, {dt:.0f}s)")

    # Update config
    config = {
        "version": "v5_retrained",
        "retrained_at": time.strftime("%Y-%m-%d %H:%M"),
        "coins": list(results.keys()),
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. {len(results)} models retrained.")


if __name__ == "__main__":
    main()
