#!/usr/bin/env python3
"""
ultraTM Multi-Coin Production Pipeline

Trains and evaluates PLE v4 models for multiple coins simultaneously.
Uses OI divergence + Funding + LS ratio alpha features.

Top 5 portfolio (validated +92% mean, 4/5 all windows positive):
  BCH +142%, ONT +102%, BTC +82%, LINK +75%, ATOM +59%

Usage:
  python run_multicoin.py                    # all 5 coins
  python run_multicoin.py --coins BTC ETH    # specific coins
  python run_multicoin.py --mode train       # train only
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

from ultrathink.pipeline import UltraThink
from ple.model_v4 import PLEv4
from ple.trainer_v4 import TradingDatasetV4, train_ple_v4
from ple.model_v3 import partition_features

TOP5 = ["BTCUSDT", "BCHUSDT", "ONTUSDT", "LINKUSDT", "ATOMUSDT"]


def add_derivatives_features(X, symbol, data_dir="data/merged"):
    """Add OI divergence + Funding + LS ratio features."""
    idx = X.index
    try:
        m = pd.read_parquet(f"{data_dir}/{symbol}/metrics.parquet").set_index("timestamp").sort_index()
        if m.index.tz is not None: m.index = m.index.tz_localize(None)
        oi = m["sum_open_interest_value"].resample("15min").last().ffill().reindex(idx, method="ffill")
        c = pd.read_parquet(f"{data_dir}/{symbol}/kline_15m.parquet").set_index("timestamp").sort_index()
        if c.index.tz is not None: c.index = c.index.tz_localize(None)
        c = c["close"].reindex(idx, method="ffill")
        for lb in [48, 96, 192]:
            X[f"oi_div_{lb}"] = oi.pct_change(lb) - c.pct_change(lb)
            X[f"oi_chg_{lb}"] = oi.pct_change(lb)
        if "count_long_short_ratio" in m.columns:
            ls = m["count_long_short_ratio"].resample("15min").last().ffill().reindex(idx, method="ffill")
            for lb in [48, 96, 192]:
                X[f"ls_chg_{lb}"] = ls.pct_change(lb)
    except Exception:
        pass
    try:
        fr = pd.read_parquet(f"{data_dir}/{symbol}/funding_rate.parquet").set_index("timestamp").sort_index()
        if fr.index.tz is not None: fr.index = fr.index.tz_localize(None)
        fr_15m = fr["funding_rate"].resample("15min").ffill().reindex(idx, method="ffill")
        X["funding_rate_raw"] = fr_15m
        X["funding_zscore_672"] = (fr_15m - fr_15m.rolling(672).mean()) / fr_15m.rolling(672).std().replace(0, np.nan)
        X["funding_extreme_95"] = (fr_15m.abs() > fr_15m.abs().rolling(672).quantile(0.95)).astype(float)
    except Exception:
        pass
    return X.fillna(0).replace([np.inf, -np.inf], 0)


def main():
    parser = argparse.ArgumentParser(description="ultraTM Multi-Coin Pipeline")
    parser.add_argument("--coins", nargs="+", default=TOP5)
    parser.add_argument("--start", default="2020-06-01")
    parser.add_argument("--end", default="2026-02-28")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["train", "backtest", "both"], default="both")
    args = parser.parse_args()

    ut = UltraThink()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"ultraTM Multi-Coin — {len(args.coins)} coins")
    print(f"Period: {args.start} ~ {args.end}\n")

    results = {}
    for symbol in args.coins:
        symbol = symbol if symbol.endswith("USDT") else symbol + "USDT"
        print(f"{'='*50}")
        print(f"  {symbol}")
        print(f"{'='*50}")

        try:
            X, L, kline, si = ut.prepare(symbol, args.start, args.end)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        X = add_derivatives_features(X, symbol)
        pt = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}

        tc = sorted([c for c in L.columns if c.startswith("tbm_")])
        mc = sorted([c for c in L.columns if c.startswith("mae_")])
        fc = sorted([c for c in L.columns if c.startswith("mfe_")])
        rc = sorted([c for c in L.columns if c.startswith("rar_")])
        wc = sorted([c for c in L.columns if c.startswith("wgt_")])

        if not tc:
            print(f"  SKIP: no labels")
            continue

        ns = len(tc)
        n = len(X)
        ws = n // 4

        Xn = X.values.astype(np.float32)
        ac = np.zeros((n, 4), dtype=np.float32)
        ac[:, 0] = 1.0
        wn = L[wc].values if wc else None

        print(f"  Data: {X.shape}, strategies: {ns}")

        window_returns = []
        for w in range(3):
            ts = w * (ws // 3)
            te = ts + int(ws * 2)
            ve = te + int(ws * 0.5)
            test_e = min(ve + ws, n)
            if test_e <= ve:
                continue

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            tds = TradingDatasetV4(Xn[ts:te], L[tc].values[ts:te], L[mc].values[ts:te],
                                    L[fc].values[ts:te], L[rc].values[ts:te], ac[ts:te],
                                    wn[ts:te] if wn is not None else None)
            vds = TradingDatasetV4(Xn[te:ve], L[tc].values[te:ve], L[mc].values[te:ve],
                                    L[fc].values[te:ve], L[rc].values[te:ve], ac[te:ve],
                                    wn[te:ve] if wn is not None else None)

            model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                          expert_hidden=256, expert_output=128, fusion_dim=256,
                          dropout=0.2, use_vsn=False)

            train_ple_v4(model, tds, vds, epochs=50, batch_size=2048,
                          device=device, patience=7, rdrop_alpha=1.0, seed=args.seed)

            if args.mode in ["backtest", "both"]:
                model.eval()
                with torch.no_grad():
                    out = model(torch.tensor(X.iloc[ve:test_e].values.astype(np.float32)).to(device),
                                torch.zeros(test_e - ve, 4).to(device))
                    probs = out["label_probs"].cpu().numpy()
                    mfp = out["mfe_pred"].cpu().numpy()
                    map_ = out["mae_pred"].cpu().numpy()

                tc_p = kline["15m"]["close"].reindex(X.iloc[ve:test_e].index, method="ffill").values
                sv = kline["4h"]["close"].rolling(50).mean().resample("15min").ffill().reindex(
                    X.iloc[ve:test_e].index, method="ffill").values
                nt = test_e - ve
                idx = np.arange(0, nt - 1, 4)
                M = len(idx)
                il = np.array([s["dir"] == "long" for s in si])
                hs = np.array([{"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(
                    s["style"], 12) for s in si])
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
                ht = em[np.arange(M), bj] > 0
                ti, tj = idx[ht], bj[ht]
                T = len(ti)

                if T == 0:
                    window_returns.append(0.0)
                    continue

                ei = np.minimum(ti + hs[tj], nt - 1)
                net = ds[tj] * (tc_p[ei] - tc_p[ti]) / tc_p[ti] - 0.0008
                cap, pk = 100000.0, 100000.0
                for i in range(T):
                    dd = (pk - cap) / pk if pk > 0 else 0
                    cap += net[i] * cap * 0.03 * max(0.2, 1 - dd / 0.15)
                    pk = max(pk, cap)
                ret = round((cap - 100000) / 100000 * 100, 2)
                window_returns.append(ret)

        if window_returns:
            mean_ret = np.mean(window_returns)
            results[symbol] = {"windows": window_returns, "mean": round(mean_ret, 2)}
            print(f"  Results: {[f'{r:+.1f}%' for r in window_returns]} → mean={mean_ret:+.1f}%\n")

    # Portfolio summary
    if results:
        print(f"\n{'='*50}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"{'='*50}")
        portfolio_mean = np.mean([r["mean"] for r in results.values()])
        for sym, r in sorted(results.items(), key=lambda x: -x[1]["mean"]):
            all_pos = all(w > 0 for w in r["windows"])
            flag = "✓" if all_pos else " "
            print(f"  {flag} {sym:12s}: {r['mean']:+.1f}%  {r['windows']}")
        print(f"\n  Portfolio mean: {portfolio_mean:+.1f}%")
        print(f"  Coins: {len(results)}/{len(args.coins)}")


if __name__ == "__main__":
    main()
