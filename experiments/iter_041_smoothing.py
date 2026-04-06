#!/usr/bin/env python3
"""
Iteration 041: Label Smoothing + Larger Fusion

Problems:
  - W2 (bear/sideways) consistently negative (-4 to -5%)
  - Model may be overconfident on wrong signals

Changes:
  1. Label smoothing: target 0→0.05, 1→0.95 (prevents overconfident predictions)
  2. Fusion dim 128→192 (more capacity to learn from 382 features)
  3. Keep d=0.20, α=1.0, 32 strategies, 382 features (iter 038 baseline)

Compare: smoothing vs no-smoothing, 128 vs 192 fusion.
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
from ple.trainer_v4 import TradingDatasetV4, _kl_binary
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


class PLEv4LossSmoothed(PLEv4Loss):
    """PLEv4Loss with label smoothing on BCE targets."""
    def __init__(self, n_losses=4, smooth=0.05):
        super().__init__(n_losses)
        self.smooth = smooth

    def forward(self, outputs, batch):
        # Apply label smoothing to the target before BCE
        if self.smooth > 0:
            rar = batch["rar"]
            rar_mask = batch["rar_mask"]
            target = (rar > 0).float() * rar_mask
            # Smooth: 0 → smooth, 1 → 1-smooth
            target = target * (1 - 2 * self.smooth) + self.smooth
            # Temporarily replace rar for the parent's forward
            batch = dict(batch)  # shallow copy
            batch["rar"] = torch.where(rar_mask > 0,
                                        torch.where(rar > 0,
                                                    torch.ones_like(rar) * (1 / self.smooth),  # keep as positive
                                                    torch.ones_like(rar) * (-1)),
                                        rar)
        return super().forward(outputs, batch)


def train_rdrop(model, train_ds, val_ds, epochs=50, batch_size=2048,
                lr=5e-4, device="cuda", patience=7, rdrop_alpha=1.0,
                label_smooth=0.0):
    model = model.to(device)
    if label_smooth > 0:
        loss_fn = PLEv4Loss(n_losses=4).to(device)  # Use standard, smooth in targets
    else:
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

            # Label smoothing: modify targets before loss
            if label_smooth > 0:
                # tbm targets are already (tbm+1)/2, so 0=loss, 1=win
                # Smooth: 0→smooth, 1→1-smooth
                batch["tbm"] = batch["tbm"] * (1 - 2 * label_smooth) + label_smooth

            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]

            out1 = model(batch["features"], batch["account"])
            out2 = model(batch["features"], batch["account"])
            l1, l2 = loss_fn(out1, batch), loss_fn(out2, batch)
            total = (l1["total"] + l2["total"]) / 2 + rdrop_alpha * _kl_binary(
                out1["label_probs"], out2["label_probs"], batch["rar_mask"])
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            vm = []
            for b in val_loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                vm.append(loss_fn(model(b["features"], b["account"]), b)["total"].item())
            v = np.mean(vm)

        if v < best_val:
            best_val, no_improve = v, 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience: break

    if best_state: model.load_state_dict(best_state)
    return model


def backtest_vec(model, X_test, kline_15m, kline_4h, strat_info, fee=0.0008, device="cuda"):
    n = len(X_test)
    if n < 100:
        return {"return": 0.0, "trades": 0, "wr": 0.0, "bh": 0.0}
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test.values.astype(np.float32)).to(device),
                    torch.zeros(n, 4).to(device))
        probs, mfe_p, mae_p = out["label_probs"].cpu().numpy(), out["mfe_pred"].cpu().numpy(), out["mae_pred"].cpu().numpy()
    tc = kline_15m["close"].reindex(X_test.index, method="ffill").values
    sv = kline_4h["close"].rolling(50).mean().resample("15min").ffill().reindex(X_test.index, method="ffill").values
    idx = np.arange(0, n - 1, 4); M = len(idx)
    is_long = np.array([s["dir"] == "long" for s in strat_info])
    holds = np.array([{"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(s["style"], 12) for s in strat_info])
    dirs = np.where(is_long, 1.0, -1.0)
    above = ~np.isnan(sv[idx]) & (tc[idx] > sv[idx])
    thresh = np.where(is_long[None, :], np.where(above, 0.40, 0.55)[:, None], np.where(above, 0.55, 0.40)[:, None])
    p = probs[idx]
    ev = p * np.maximum(np.abs(mfe_p[idx]), 0.001) - (1 - p) * np.maximum(np.abs(mae_p[idx]), 0.001) - fee
    valid = (p >= thresh) & (ev > 0)
    ev_masked = np.where(valid, ev, -np.inf)
    best_j = np.argmax(ev_masked, axis=1)
    has_trade = ev_masked[np.arange(M), best_j] > 0
    trade_idx, trade_j = idx[has_trade], best_j[has_trade]
    T = len(trade_idx)
    if T == 0:
        return {"return": 0.0, "bh": round((tc[-1]-tc[0])/tc[0]*100, 2), "trades": 0, "wr": 0.0, "n_long": 0, "n_short": 0}
    exit_idx = np.minimum(trade_idx + holds[trade_j], n - 1)
    net = dirs[trade_j] * (tc[exit_idx] - tc[trade_idx]) / tc[trade_idx] - fee
    capital, peak = 100000.0, 100000.0
    for i in range(T):
        dd = (peak - capital) / peak if peak > 0 else 0
        capital += net[i] * capital * 0.03 * max(0.2, 1 - dd / 0.15)
        peak = max(peak, capital)
    ret = (capital - 100000) / 100000 * 100
    return {"return": round(ret, 2), "bh": round((tc[-1]-tc[0])/tc[0]*100, 2), "trades": T,
            "wr": round((net > 0).mean() * 100, 1),
            "n_long": int((dirs[trade_j] > 0).sum()), "n_short": int((dirs[trade_j] < 0).sum())}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  ITERATION 041: Label Smoothing + Larger Fusion", flush=True)
    print("=" * 60, flush=True)

    START, END = "2020-06-01", "2026-02-28"
    SEEDS = [42, 123, 777]

    print("\n[1/4] Loading data + features + labels...", flush=True)
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
    labels = lr["intraday"].copy()
    for name in ["scalp", "daytrade", "swing"]:
        if name in lr:
            sw = lr[name].resample("15min").ffill() if name != "intraday" else lr[name]
            for col in sw.columns:
                if col not in labels.columns:
                    labels[col] = sw[col]
    common = features.index.intersection(labels.index)
    X, L = features.loc[common], labels.loc[common]
    print(f"  Data: {X.shape}", flush=True)

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])
    strat_info = [{"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]} for c in tbm_cols]
    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}
    n, n_strat = len(X), len(tbm_cols)
    ws = n // 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_np = X.values.astype(np.float32)
    acc = np.zeros((n, 4), dtype=np.float32); acc[:, 0] = 1.0
    wgt_np = L[wgt_cols].values if wgt_cols else None

    # Test configs
    configs = {
        "baseline_f128": {"fusion": 128, "smooth": 0.0},
        "smooth_f128": {"fusion": 128, "smooth": 0.05},
        "baseline_f192": {"fusion": 192, "smooth": 0.0},
        "smooth_f192": {"fusion": 192, "smooth": 0.05},
    }

    print(f"\n[2/4] Testing {len(configs)} configs × {len(SEEDS)} seeds...", flush=True)

    for cfg_name, cfg in configs.items():
        seed_means = []
        for seed in SEEDS:
            wrets = []
            for w in range(3):
                ts = w * (ws // 3)
                te = ts + int(ws * 2)
                ve = te + int(ws * 0.5)
                test_e = min(ve + ws, n)
                if test_e <= ve: continue

                torch.manual_seed(seed); np.random.seed(seed)
                tds = TradingDatasetV4(X_np[ts:te], L[tbm_cols].values[ts:te], L[mae_cols].values[ts:te],
                                        L[mfe_cols].values[ts:te], L[rar_cols].values[ts:te], acc[ts:te],
                                        wgt_np[ts:te] if wgt_np is not None else None)
                vds = TradingDatasetV4(X_np[te:ve], L[tbm_cols].values[te:ve], L[mae_cols].values[te:ve],
                                        L[mfe_cols].values[te:ve], L[rar_cols].values[te:ve], acc[te:ve],
                                        wgt_np[te:ve] if wgt_np is not None else None)

                model = PLEv4(feature_partitions=partitions, n_account_features=4, n_strategies=n_strat,
                              expert_hidden=128, expert_output=64, fusion_dim=cfg["fusion"], dropout=0.2)
                model = train_rdrop(model, tds, vds, epochs=50, device=device, patience=7,
                                     rdrop_alpha=1.0, label_smooth=cfg["smooth"])
                r = backtest_vec(model, X.iloc[ve:test_e], kline["15m"], kline["4h"], strat_info, device=device)
                wrets.append(r["return"])

            sm = np.mean(wrets)
            seed_means.append(sm)

        overall = np.mean(seed_means)
        std = np.std(seed_means)
        print(f"  {cfg_name}: mean={overall:+.2f}% std={std:.1f}% seeds={[round(m,1) for m in seed_means]}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)

    report = {"iteration": 41, "approach": "Label smoothing + fusion dim sweep", "time_seconds": round(elapsed)}
    with open("reports/iteration_041.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to reports/iteration_041.json", flush=True)


if __name__ == "__main__":
    main()
