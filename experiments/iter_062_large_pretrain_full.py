#!/usr/bin/env python3
"""
Iteration 062: Large Model Pre-train on Full Data (UltraThink pipeline)

Uses UltraThink cache pipeline for fast iteration.
Large model (728K) + 4-coin pre-train + BTC fine-tune on full data.

Expected: +46% baseline return + std <1% from pre-training.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultrathink.pipeline import UltraThink
from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v4 import TradingDatasetV4, train_ple_v4
from ple.model_v3 import partition_features


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100: return {"return": 0.0, "trades": 0}
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
    return {"return": round((cap - 100000) / 100000 * 100, 2), "trades": T,
            "wr": round((net > 0).mean() * 100, 1)}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 062: Large Pre-train Full Data", flush=True)
    print("=" * 60, flush=True)

    ut = UltraThink()
    START, END = "2020-06-01", "2026-02-28"
    COINS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"]
    SEEDS = [42, 123, 777]
    device = "cuda"

    # ── Load all coins via UltraThink (cached) ──
    print("\n[1/4] Loading coin data...", flush=True)
    coin_data = {}
    feat_cols = None
    for sym in COINS:
        try:
            X, L, kline, si = ut.prepare(sym, START, END)
        except Exception as e:
            print(f"  {sym}: SKIP ({e})", flush=True)
            continue
        if feat_cols is None:
            feat_cols = list(X.columns)
        else:
            missing = [c for c in feat_cols if c not in X.columns]
            for c in missing: X[c] = 0.0
            X = X[feat_cols]
        coin_data[sym] = {"X": X, "L": L, "kline": kline, "si": si}
        print(f"  {sym}: {X.shape}", flush=True)

    if "BTCUSDT" not in coin_data:
        print("No BTC data!", flush=True)
        return

    pt = {k: v for k, v in partition_features(feat_cols).items() if len(v) > 0}
    btc = coin_data["BTCUSDT"]
    tc = sorted([c for c in btc["L"].columns if c.startswith("tbm_")])
    mc = sorted([c for c in btc["L"].columns if c.startswith("mae_")])
    fc = sorted([c for c in btc["L"].columns if c.startswith("mfe_")])
    rc = sorted([c for c in btc["L"].columns if c.startswith("rar_")])
    wc = sorted([c for c in btc["L"].columns if c.startswith("wgt_")])
    ns = len(tc); si = btc["si"]

    # ── Combined dataset ──
    print("\n[2/4] Building combined dataset...", flush=True)
    all_t, all_v = [], []
    for sym, d in coin_data.items():
        X, L = d["X"], d["L"]
        for cl in [tc, mc, fc, rc, wc]:
            for c in cl:
                if c not in L.columns: L[c] = np.nan
        n = len(X); s1 = int(n * 0.6); s2 = int(n * 0.8)
        Xn = X.values.astype(np.float32); ac = np.zeros((n, 4), dtype=np.float32); ac[:, 0] = 1.0
        wn = L[wc].values if wc else None
        all_t.append(TradingDatasetV4(Xn[:s1], L[tc].values[:s1], L[mc].values[:s1], L[fc].values[:s1], L[rc].values[:s1], ac[:s1], wn[:s1] if wn is not None else None))
        all_v.append(TradingDatasetV4(Xn[s1:s2], L[tc].values[s1:s2], L[mc].values[s1:s2], L[fc].values[s1:s2], L[rc].values[s1:s2], ac[s1:s2], wn[s1:s2] if wn is not None else None))
    ct = ConcatDataset(all_t); cv = ConcatDataset(all_v)
    print(f"  Combined: {len(ct)} train, {len(cv)} val", flush=True)

    # ── Pre-train large model ──
    print("\n[3/4] Pre-training large model (e256_o128_f256)...", flush=True)
    torch.manual_seed(42); np.random.seed(42)
    pre = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                 expert_hidden=256, expert_output=128, fusion_dim=256, dropout=0.2, use_vsn=False)
    print(f"  Params: {pre.count_parameters():,}", flush=True)
    train_ple_v4(pre, ct, cv, epochs=30, batch_size=2048, device=device, patience=5, rdrop_alpha=1.0, seed=42)
    pre_state = {k: v.cpu().clone() for k, v in pre.state_dict().items()}

    # ── Walk-forward on BTC ──
    print("\n[4/4] Walk-forward BTC (scratch vs pretrain)...", flush=True)
    X_btc, L_btc, kl = btc["X"], btc["L"], btc["kline"]
    n = len(X_btc); ws = n // 4
    Xn = X_btc.values.astype(np.float32); ac = np.zeros((n, 4), dtype=np.float32); ac[:, 0] = 1.0
    wn = L_btc[wc].values if wc else None

    for method, init in [("scratch", None), ("pretrain", pre_state)]:
        sm = []
        for seed in SEEDS:
            wr = []
            for w in range(3):
                ts = w * (ws // 3); te = ts + int(ws * 2); ve = te + int(ws * 0.5); test_e = min(ve + ws, n)
                if test_e <= ve: continue
                torch.manual_seed(seed); np.random.seed(seed)
                tds = TradingDatasetV4(Xn[ts:te], L_btc[tc].values[ts:te], L_btc[mc].values[ts:te], L_btc[fc].values[ts:te], L_btc[rc].values[ts:te], ac[ts:te], wn[ts:te] if wn is not None else None)
                vds = TradingDatasetV4(Xn[te:ve], L_btc[tc].values[te:ve], L_btc[mc].values[te:ve], L_btc[fc].values[te:ve], L_btc[rc].values[te:ve], ac[te:ve], wn[te:ve] if wn is not None else None)
                m = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                          expert_hidden=256, expert_output=128, fusion_dim=256, dropout=0.2, use_vsn=False)
                if init: m.load_state_dict({k: v.clone() for k, v in init.items()})
                train_ple_v4(m, tds, vds, epochs=50, batch_size=2048, device=device, patience=7, rdrop_alpha=1.0, seed=seed)
                r = backtest_vec(m, X_btc.iloc[ve:test_e], kl["15m"], kl["4h"], si, device=device)
                wr.append(r["return"])
            s = np.mean(wr); sm.append(s)
            print(f"  {method} s{seed}: {[round(x, 1) for x in wr]} → {s:+.2f}%", flush=True)
        print(f"  {method}: mean={np.mean(sm):+.2f}% std={np.std(sm):.1f}%\n", flush=True)

    elapsed = time.time() - t0
    print(f"Time: {elapsed:.0f}s", flush=True)
    json.dump({"iteration": 62}, open("reports/iteration_062.json", "w"))
    print("Report saved", flush=True)


if __name__ == "__main__":
    main()
