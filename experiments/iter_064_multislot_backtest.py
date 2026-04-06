#!/usr/bin/env python3
"""
Iteration 064: Multi-Slot Backtest via Position Manager

Compare single-slot (current) vs multi-slot (3 positions) trading.
Multi-slot allows:
  - Simultaneous long scalp + short swing
  - Better capital utilization
  - Diversification across strategies

Uses current data + optimal model config.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultrathink.pipeline import UltraThink
from ple.model_v4 import PLEv4
from ple.trainer_v4 import TradingDatasetV4, train_ple_v4
from ple.model_v3 import partition_features
from execution.position_manager import PositionManager


def run_backtest(model, X_test, kline_15m, kline_4h, strat_info,
                  max_slots=1, leverage=1.0, fee=0.0008, device="cuda"):
    """Run backtest with Position Manager."""
    n = len(X_test)
    if n < 100:
        return {"return": 0.0}

    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n, 4).to(device))
        probs = out["label_probs"].cpu().numpy()
        mfe_p = out["mfe_pred"].cpu().numpy()
        mae_p = out["mae_pred"].cpu().numpy()

    tc = kline_15m["close"].reindex(X_test.index, method="ffill").values
    sv = kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values

    pm = PositionManager(max_slots=max_slots, initial_equity=100000,
                          max_total_exposure=0.3 * max_slots, fee_pct=fee)

    holds_map = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}

    for i in range(0, n - 1, 4):
        # Update existing positions
        pm.update_positions({"BTC": tc[i]}, bar_idx=i)

        # Check for new signals
        above = not np.isnan(sv[i]) and tc[i] > sv[i]
        lt = 0.40 if above else 0.55
        st = 0.55 if above else 0.40

        # Collect all valid signals (not just best)
        signals = []
        for j, info in enumerate(strat_info):
            th = lt if info["dir"] == "long" else st
            if probs[i, j] < th:
                continue
            p = probs[i, j]
            rew = max(abs(mfe_p[i, j]), 0.001)
            rsk = max(abs(mae_p[i, j]), 0.001)
            ev = p * rew - (1 - p) * rsk - fee
            if ev <= 0:
                continue
            d = 1 if info["dir"] == "long" else -1
            hold = holds_map.get(info["style"], 12)
            signals.append({
                "asset": "BTC", "direction": d, "probability": p,
                "mfe": rew, "mae": rsk, "strategy": f"{info['style']}_{info['dir']}",
                "hold_limit": hold, "ev": ev,
            })

        if not signals:
            continue

        # Position Manager selects best signals for available slots
        best = pm.get_best_signals(signals)

        for sig in best:
            # DD-based sizing
            dd = pm.state.drawdown
            size = 0.03 * max(0.2, 1 - dd / 0.15)

            pm.open_position(
                asset=sig["asset"],
                direction=sig["direction"],
                entry_price=tc[i],
                size_pct=size,
                leverage=leverage,
                strategy=sig["strategy"],
                hold_limit=sig["hold_limit"],
                bar_idx=i,
            )

    # Close remaining
    for sid in list(pm.positions.keys()):
        pm.close_position(sid, tc[-1], "end", n-1)

    return pm.summary()


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 064: Multi-Slot Backtest", flush=True)
    print("=" * 60, flush=True)

    ut = UltraThink()
    START, END = "2023-06-01", "2025-12-31"
    device = "cuda"

    print("\n[1/3] Loading data...", flush=True)
    X, L, kline, strat_info = ut.prepare("BTCUSDT", START, END)
    print(f"  Data: {X.shape}", flush=True)

    pt = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    tc_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mc_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    fc_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rc_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wc_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    ns = len(tc_cols)

    n = len(X); ws = n // 4
    Xn = X.values.astype(np.float32)
    ac = np.zeros((n, 4), dtype=np.float32); ac[:, 0] = 1.0
    wn = L[wc_cols].values if wc_cols else None

    configs = [
        {"name": "1-slot 1x", "slots": 1, "lev": 1.0},
        {"name": "3-slot 1x", "slots": 3, "lev": 1.0},
        {"name": "3-slot 2x", "slots": 3, "lev": 2.0},
        {"name": "5-slot 1x", "slots": 5, "lev": 1.0},
    ]

    print(f"\n[2/3] Training + backtesting (seed 42)...", flush=True)

    # Single train, multiple backtest configs
    w = 1  # use middle window
    ts = w*(ws//3); te = ts+int(ws*2); ve = te+int(ws*0.5); test_e = min(ve+ws, n)

    torch.manual_seed(42); np.random.seed(42)
    tds = TradingDatasetV4(Xn[ts:te], L[tc_cols].values[ts:te], L[mc_cols].values[ts:te],
                            L[fc_cols].values[ts:te], L[rc_cols].values[ts:te], ac[ts:te],
                            wn[ts:te] if wn is not None else None)
    vds = TradingDatasetV4(Xn[te:ve], L[tc_cols].values[te:ve], L[mc_cols].values[te:ve],
                            L[fc_cols].values[te:ve], L[rc_cols].values[te:ve], ac[te:ve],
                            wn[te:ve] if wn is not None else None)

    model = PLEv4(feature_partitions=pt, n_account_features=4, n_strategies=ns,
                  expert_hidden=128, expert_output=64, fusion_dim=192, dropout=0.2, use_vsn=False)
    train_ple_v4(model, tds, vds, epochs=50, batch_size=2048, device=device, patience=7, rdrop_alpha=1.0, seed=42)

    X_test = X.iloc[ve:test_e]

    print(f"\n[3/3] Comparing {len(configs)} slot configs...", flush=True)
    for cfg in configs:
        r = run_backtest(model, X_test, kline["15m"], kline["4h"], strat_info,
                          max_slots=cfg["slots"], leverage=cfg["lev"], device=device)
        print(f"  {cfg['name']:15s}: return={r['return_pct']:+.2f}% trades={r['trades']} "
              f"WR={r.get('win_rate',0):.1f}% MaxDD={r.get('max_drawdown',0):.1f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)
    json.dump({"iteration": 64, "approach": "Multi-slot backtest"},
              open("reports/iteration_064.json", "w"), indent=2)
    print("Report saved", flush=True)


if __name__ == "__main__":
    main()
