#!/usr/bin/env python3
"""
Iteration 037 v2: Vectorized Backtest + Fast Sweep

Optimizations vs v1:
  1. Backtest fully vectorized with numpy (no Python for loops)
  2. Screen phase: 20 epochs, no R-Drop (fast dropout screening)
  3. Validate phase: 50 epochs, R-Drop only on top 3 dropouts
"""

import os, sys, json, time
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


def _kl_binary(p1, p2, mask):
    eps = 1e-7
    p1, p2 = p1.clamp(eps, 1 - eps), p2.clamp(eps, 1 - eps)
    kl_1 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
    kl_2 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())
    return ((kl_1 + kl_2) / 2 * mask).sum() / mask.sum().clamp(1)


def train_fast(model, train_ds, val_ds, epochs=20, batch_size=2048,
               lr=5e-4, device="cuda", patience=5, rdrop_alpha=0.0):
    """Fast training: fewer epochs, optional R-Drop."""
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
                l1, l2 = loss_fn(out1, batch), loss_fn(out2, batch)
                total = (l1["total"] + l2["total"]) / 2 + rdrop_alpha * _kl_binary(
                    out1["label_probs"], out2["label_probs"], batch["rar_mask"])
            else:
                out = model(batch["features"], batch["account"])
                total = loss_fn(out, batch)["total"]

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            vm = []
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                vm.append(loss_fn(model(batch["features"], batch["account"]), batch)["total"].item())
            v = np.mean(vm)

        if v < best_val:
            best_val, no_improve = v, 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── VECTORIZED BACKTEST ────────────────────────────────────────────────────

def backtest_vectorized(model, X_test, kline_15m, kline_4h, strat_info,
                        fee=0.0008, size=0.03, device="cuda"):
    """Fully vectorized backtest — no Python for loops over bars/strategies."""
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0, "bh": 0.0}

    model = model.to(device).eval()
    with torch.no_grad():
        out = model(
            torch.tensor(X_test.values.astype(np.float32)).to(device),
            torch.zeros(n, 4).to(device))
        probs = out["label_probs"].cpu().numpy()     # (n, S)
        mfe_p = out["mfe_pred"].cpu().numpy()         # (n, S)
        mae_p = out["mae_pred"].cpu().numpy()         # (n, S)

    S = len(strat_info)
    close = kline_15m["close"]
    sma = kline_4h["close"].rolling(50).mean().resample("15min").ffill()
    tc = close.reindex(X_test.index, method="ffill").values
    sv = sma.reindex(X_test.index, method="ffill").values

    # Sample every 4 bars
    idx = np.arange(0, n - 1, 4)
    M = len(idx)

    # Direction mask: is_long[j] = True if strategy j is long
    is_long = np.array([s["dir"] == "long" for s in strat_info])  # (S,)
    hold_map = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}
    holds = np.array([hold_map.get(s["style"], 12) for s in strat_info])  # (S,)
    dirs = np.where(is_long, 1.0, -1.0)  # (S,)

    # SMA filter → thresholds per bar
    above = ~np.isnan(sv[idx]) & (tc[idx] > sv[idx])  # (M,)
    lt = np.where(above, 0.40, 0.55)  # (M,)  long threshold
    st = np.where(above, 0.55, 0.40)  # (M,)  short threshold

    # Threshold matrix: (M, S)
    thresh = np.where(is_long[None, :], lt[:, None], st[:, None])

    # Prob matrix at sampled bars: (M, S)
    p = probs[idx]
    mfe = np.abs(mfe_p[idx])
    mae = np.abs(mae_p[idx])

    # EV calculation: vectorized over (M, S)
    rew = np.maximum(mfe, 0.001)
    rsk = np.maximum(mae, 0.001)
    ev = p * rew - (1 - p) * rsk - fee  # (M, S)

    # Mask: prob >= threshold AND ev > 0
    valid = (p >= thresh) & (ev > 0)  # (M, S)

    # For each bar, pick best EV strategy
    ev_masked = np.where(valid, ev, -np.inf)  # (M, S)
    best_j = np.argmax(ev_masked, axis=1)  # (M,)
    best_ev = ev_masked[np.arange(M), best_j]  # (M,)
    has_trade = best_ev > 0  # (M,)

    # Compute PnL for selected strategies
    trade_idx = idx[has_trade]  # bar indices where we trade
    trade_j = best_j[has_trade]  # which strategy
    T = len(trade_idx)

    if T == 0:
        bh = (tc[-1] - tc[0]) / tc[0] * 100 if tc[0] > 0 else 0
        return {"return": 0.0, "bh": round(bh, 2), "trades": 0, "wr": 0.0,
                "n_long": 0, "n_short": 0}

    # Exit indices
    exit_idx = np.minimum(trade_idx + holds[trade_j], n - 1)

    # PnL
    entry_prices = tc[trade_idx]
    exit_prices = tc[exit_idx]
    trade_dirs = dirs[trade_j]  # (T,)
    pnl = trade_dirs * (exit_prices - entry_prices) / entry_prices  # (T,)
    net = pnl - fee  # (T,)

    # Sequential capital simulation (must be sequential due to compounding)
    capital = 100000.0
    peak = capital
    for i in range(T):
        dd = (peak - capital) / peak if peak > 0 else 0
        sz = size * max(0.2, 1 - dd / 0.15)
        capital += net[i] * capital * sz
        peak = max(peak, capital)

    ret = (capital - 100000) / 100000 * 100
    bh = (tc[-1] - tc[0]) / tc[0] * 100 if tc[0] > 0 else 0
    wr = (net > 0).mean() * 100
    n_long = int((trade_dirs > 0).sum())
    n_short = int((trade_dirs < 0).sum())

    return {"return": round(ret, 2), "bh": round(bh, 2), "trades": T,
            "wr": round(wr, 1), "n_long": n_long, "n_short": n_short}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ITERATION 037v2: Vectorized Sweep")
    print("=" * 60, flush=True)

    START, END = "2020-06-01", "2026-02-28"

    print("\n[1/4] Loading data...", flush=True)
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
    print(f"  Data: {X.shape}", flush=True)

    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    strat_info = [{"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]}
                  for c in tbm_cols]

    n, n_strat = len(X), len(tbm_cols)
    ws = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_np = X.values.astype(np.float32)
    acc = np.zeros((n, 4), dtype=np.float32)
    acc[:, 0] = 1.0
    wgt_np = L[wgt_cols].values if wgt_cols else None

    # ── Phase 1: Fast screen — dropout only, no R-Drop, 20 epochs, seed 42 ──
    DROPOUTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    print(f"\n[2/4] Phase 1: Dropout screen ({len(DROPOUTS)} values, 20 epochs, no R-Drop)...", flush=True)
    screen = {}

    for d in DROPOUTS:
        t1 = time.time()
        window_rets = []
        for w in range(3):
            ts = w * (ws // 3)
            te = ts + int(ws * 2)
            ve = te + int(ws * 0.5)
            test_e = min(ve + ws, n)
            if test_e <= ve: continue

            torch.manual_seed(42)
            np.random.seed(42)

            tds = TradingDatasetV4(X_np[ts:te], L[tbm_cols].values[ts:te], L[mae_cols].values[ts:te],
                                    L[mfe_cols].values[ts:te], L[rar_cols].values[ts:te], acc[ts:te],
                                    wgt_np[ts:te] if wgt_np is not None else None)
            vds = TradingDatasetV4(X_np[te:ve], L[tbm_cols].values[te:ve], L[mae_cols].values[te:ve],
                                    L[mfe_cols].values[te:ve], L[rar_cols].values[te:ve], acc[te:ve],
                                    wgt_np[te:ve] if wgt_np is not None else None)

            model = PLEv4(feature_partitions=partitions, n_account_features=4, n_strategies=n_strat,
                          expert_hidden=128, expert_output=64, fusion_dim=128, dropout=d)
            model = train_fast(model, tds, vds, epochs=20, device=device, patience=5, rdrop_alpha=0.0)
            r = backtest_vectorized(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], strat_info, device=device)
            window_rets.append(r["return"])

        mean = np.mean(window_rets)
        screen[d] = {"windows": window_rets, "mean": round(mean, 2)}
        dt = time.time() - t1
        print(f"  d={d:.2f}: {window_rets} → mean={mean:+.2f}% ({dt:.0f}s)", flush=True)

    # ── Phase 2: Top 3 dropout × R-Drop alphas, full 50 epochs, 3 seeds ──
    top3_d = sorted(screen.items(), key=lambda x: x[1]["mean"], reverse=True)[:3]
    ALPHAS = [0.0, 0.3, 0.5, 1.0]
    SEEDS = [42, 123, 777]

    print(f"\n[3/4] Phase 2: Top 3 dropout × {len(ALPHAS)} alphas × {len(SEEDS)} seeds (50 epochs)...", flush=True)
    final = {}

    for d, _ in top3_d:
        for a in ALPHAS:
            key = f"d{d:.2f}_a{a:.1f}"
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
                                  expert_hidden=128, expert_output=64, fusion_dim=128, dropout=d)
                    model = train_fast(model, tds, vds, epochs=50, device=device, patience=7, rdrop_alpha=a)
                    r = backtest_vectorized(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], strat_info, device=device)
                    wrets.append(r["return"])

                sm = np.mean(wrets)
                seed_means.append(sm)

            overall = np.mean(seed_means)
            std = np.std(seed_means)
            final[key] = {"dropout": d, "alpha": a, "overall": round(overall, 2),
                          "std": round(std, 2), "seeds": {str(s): round(m, 2) for s, m in zip(SEEDS, seed_means)}}
            print(f"  {key}: mean={overall:+.2f}% std={std:.1f}% seeds={[round(m,1) for m in seed_means]}", flush=True)

    # ── Summary ──
    elapsed = time.time() - t0
    best = max(final.items(), key=lambda x: x[1]["overall"])

    print(f"\n{'=' * 60}")
    print(f"  ITERATION 037 RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Phase 1 Screen (dropout only):")
    for d in sorted(screen.keys()):
        print(f"    d={d:.2f}: mean={screen[d]['mean']:+.2f}%")
    print(f"\n  Phase 2 Final (top 3 × alphas × seeds):")
    for k, v in sorted(final.items(), key=lambda x: x[1]["overall"], reverse=True):
        print(f"    {k}: mean={v['overall']:+.2f}% std={v['std']:.1f}%")
    print(f"\n  BEST: {best[0]} (d={best[1]['dropout']}, α={best[1]['alpha']}, mean={best[1]['overall']:+.2f}%)")
    print(f"  Time: {elapsed:.0f}s", flush=True)

    report = {
        "iteration": 37,
        "approach": "Vectorized sweep: dropout screen → top3 × alpha × seeds",
        "screen": {str(k): v for k, v in screen.items()},
        "final": final,
        "best": best[0],
        "best_dropout": best[1]["dropout"],
        "best_alpha": best[1]["alpha"],
        "mean": best[1]["overall"],
    }
    with open("reports/iteration_037.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to reports/iteration_037.json", flush=True)


if __name__ == "__main__":
    main()
