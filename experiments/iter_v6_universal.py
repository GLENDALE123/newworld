"""
PLE v6 Universal Model Training — Multi-coin regime-aware MoE

Core idea: ONE model trades ALL 244 coins.
Uses UltraThink pipeline for full 392 features.
Trains on multiple alpha coins simultaneously.

"범용모델을 개선한다."
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultrathink.pipeline import UltraThink
from ple.model_v3 import partition_features
from ple.model_v6 import PLEv6
from ple.loss_v4 import PLEv4Loss
from ple.trainer_v6 import (
    TradingDatasetV6, compute_regime_ids, _kl_binary, _worker_init,
)
from features.temporal_context import generate_temporal_features
from torch.utils.data import DataLoader, ConcatDataset


DATA_DIR = "data/merged"
TRAIN_COINS = [
    "BCHUSDT", "UNIUSDT", "CHRUSDT", "DUSKUSDT", "CRVUSDT",
    "XRPUSDT", "AVAXUSDT", "DOGEUSDT", "ONTUSDT",  # alpha coins
    "BTCUSDT", "ETHUSDT", "SOLUSDT",  # majors
    "BNBUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT",  # large alts
    "APTUSDT", "INJUSDT", "ARBUSDT", "SUIUSDT",  # more large alts
]

ut = UltraThink(data_dir=DATA_DIR)


def prepare_coin(
    coin: str,
    coin_id: int,
    start: str = "2020-01-01",
    end: str = "2026-12-31",
) -> TradingDatasetV6 | None:
    """Prepare a single coin's data for v6 training."""
    try:
        X, labels, kline, strat_info = ut.prepare(coin, start, end)
    except Exception as e:
        print(f"  {coin}: SKIP ({e})")
        return None

    if len(X) < 1000:
        print(f"  {coin}: SKIP (only {len(X)} rows)")
        return None

    # Temporal context
    kline_15m = kline.get("15m", list(kline.values())[0])
    temporal = generate_temporal_features(kline_15m)
    temporal_df = pd.DataFrame(temporal)
    temporal_df = temporal_df.reindex(X.index).replace([np.inf, -np.inf], np.nan)
    n_temporal = temporal_df.shape[1]

    # Regime IDs
    regime_ids = compute_regime_ids(kline_15m.reindex(X.index, method="ffill"), X.index)

    # Label columns
    tbm_cols = sorted([c for c in labels.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in labels.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in labels.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in labels.columns if c.startswith("rar_")])

    n = len(X)
    coin_ids = np.full(n, coin_id, dtype=np.int64)
    account = np.zeros((n, 4), dtype=np.float32)
    account[:, 0] = 1.0

    ds = TradingDatasetV6(
        features=np.nan_to_num(X.values, 0.0),
        tbm=labels[tbm_cols].values,
        mae=labels[mae_cols].values,
        mfe=labels[mfe_cols].values,
        rar=labels[rar_cols].values,
        regime_ids=regime_ids,
        coin_ids=coin_ids,
        temporal=temporal_df.values,
        account=account,
    )

    print(f"  {coin}: {n} rows, {X.shape[1]} features, {len(tbm_cols)} labels")
    return ds


def train_universal(device: str = "cuda", epochs: int = 50, batch_size: int = 2048):
    print("=" * 70)
    print("PLE v6 Universal Model — Multi-Coin Training")
    print("=" * 70)

    # 1. Prepare all coins
    print(f"\n[1] Preparing {len(TRAIN_COINS)} coins...")
    datasets = {}
    feature_cols = None
    n_labels = None

    for i, coin in enumerate(TRAIN_COINS):
        ds = prepare_coin(coin, coin_id=i)
        if ds is not None:
            datasets[coin] = ds
            if feature_cols is None:
                # Get feature columns from first successful coin
                X, _, _, _ = ut.prepare(coin, "2020-01-01", "2026-12-31")
                feature_cols = list(X.columns)
                n_labels = ds.rar.shape[1]

    if not datasets:
        print("No data prepared!")
        return

    print(f"\n  Prepared {len(datasets)}/{len(TRAIN_COINS)} coins")
    print(f"  Features: {len(feature_cols)}, Labels: {n_labels}")

    # 2. Split into train/val/test per coin (60/20/20)
    train_sets, val_sets, test_sets = [], [], []
    for coin, ds in datasets.items():
        n = len(ds)
        s1, s2 = int(n * 0.6), int(n * 0.8)
        train_sets.append(torch.utils.data.Subset(ds, range(0, s1)))
        val_sets.append(torch.utils.data.Subset(ds, range(s1, s2)))
        test_sets.append(torch.utils.data.Subset(ds, range(s2, n)))

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    test_ds = ConcatDataset(test_sets)

    print(f"\n  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 3. Create model
    partitions = partition_features(feature_cols)
    n_groups = 4  # scalp, intraday, daytrade, swing
    n_per_group = max(1, n_labels // n_groups)
    n_temporal = 32  # from temporal_context.py

    print(f"\n[2] Creating PLE v6 Universal...")
    print(f"  Partitions: {', '.join(f'{k}={len(v)}' for k, v in partitions.items())}")

    model = PLEv6(
        feature_partitions=partitions,
        n_coins=250,
        coin_embed_dim=32,
        regime_embed_dim=16,
        n_account_features=4,
        n_temporal_features=n_temporal,
        n_strategy_groups=n_groups,
        n_labels_per_group=n_per_group,
        expert_hidden=256,
        expert_output=128,
        n_moe_experts=8,
        moe_top_k=2,
        fusion_dim=256,
        dropout=0.2,
    )

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    # 4. Train
    print(f"\n[3] Training on {device}...")
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=5e-4, weight_decay=1e-4,
    )

    total_steps = epochs * (len(train_ds) // batch_size + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=total_steps,
    )

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=12,
        pin_memory=True, persistent_workers=True,
        generator=g, worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init,
    )

    best_val = float("inf")
    no_improve = 0
    best_state = None
    rdrop_alpha = 1.0
    moe_balance_weight = 0.01

    for epoch in range(epochs):
        model.train()
        epoch_moe_loss = 0.0
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

            def fwd(b):
                return model(b["features"], b["coin_id"], b["regime_id"],
                             b["account"], b["temporal"])

            # R-Drop
            out1 = fwd(batch)
            out2 = fwd(batch)
            l1, l2 = loss_fn(out1, batch), loss_fn(out2, batch)
            task_loss = (l1["total"] + l2["total"]) / 2
            rdrop_loss = _kl_binary(out1["label_probs"], out2["label_probs"],
                                     batch["rar_mask"])
            moe_lb = model.moe_load_balance_loss(out1["moe_indices"])
            total = task_loss + rdrop_alpha * rdrop_loss + moe_balance_weight * moe_lb

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_moe_loss += moe_lb.item()
            n_batches += 1

        # Validation
        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch["features"], batch["coin_id"], batch["regime_id"],
                            batch["account"], batch["temporal"])
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

        v = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        moe_avg = epoch_moe_loss / max(n_batches, 1)

        # Expert stats
        with torch.no_grad():
            sb = next(iter(val_loader))
            sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sb.items()}
            so = model(sb["features"], sb["coin_id"], sb["regime_id"],
                       sb["account"], sb["temporal"])
            gw = so["gate_weights"].cpu().numpy().mean(0)
            moe_idx = so["moe_indices"].cpu().numpy().flatten()
            expert_pct = np.bincount(moe_idx, minlength=8) / len(moe_idx) * 100

        gate_str = " ".join(f"{w:.2f}" for w in gw)
        expert_str = " ".join(f"{p:.0f}%" for p in expert_pct)

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  "
              f"bce={v['L_label']:.3f}  cal={v['L_cal']:.4f}  "
              f"active={v['n_active']:.1f}  prec={v['precision']:.2f}  "
              f"no_trade={v['no_trade_pct']:.1%}  "
              f"moe_lb={moe_avg:.3f}  "
              f"gate=[{gate_str}]  experts=[{expert_str}]")

        if v["total"] < best_val:
            best_val = v["total"]
            no_improve = 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= 7:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # 5. Test evaluation
    print(f"\n[4] Test evaluation...")
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    all_probs, all_rar, all_mask, all_regimes, all_coins = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["features"], batch["coin_id"], batch["regime_id"],
                        batch["account"], batch["temporal"])
            all_probs.append(out["label_probs"].cpu().numpy())
            all_rar.append(batch["rar"].cpu().numpy())
            all_mask.append(batch["rar_mask"].cpu().numpy())
            all_regimes.append(batch["regime_id"].cpu().numpy())
            all_coins.append(batch["coin_id"].cpu().numpy())

    probs = np.concatenate(all_probs)
    rar = np.concatenate(all_rar)
    mask = np.concatenate(all_mask)
    regimes = np.concatenate(all_regimes)
    coins = np.concatenate(all_coins)

    regime_names = {0: "surge", 1: "dump", 2: "range", 3: "volatile"}
    coin_names = list(datasets.keys())

    # Per-regime
    print("\n  Per-regime (top-1 strategy):")
    for rid, rname in regime_names.items():
        rmask = regimes == rid
        if rmask.sum() == 0:
            continue
        r_probs = probs[rmask]
        r_rar = rar[rmask]
        r_mask = mask[rmask]
        best_idx = (r_probs * r_mask).argmax(axis=1)
        n_s = len(r_probs)
        sel_rar = r_rar[np.arange(n_s), best_idx]
        fired = r_probs[np.arange(n_s), best_idx] > 0.5
        if fired.sum() > 0:
            wr = (sel_rar[fired] > 0).mean()
            avg = sel_rar[fired].mean()
            print(f"    {rname:10s}: n={rmask.sum():6d} fired={fired.sum():5d} "
                  f"WR={wr:.1%} avg_RAR={avg:.4f}")

    # Per-coin
    print("\n  Per-coin (top-1 strategy, prob>0.5):")
    for cid, cname in enumerate(coin_names):
        cmask = coins == cid
        if cmask.sum() == 0:
            continue
        c_probs = probs[cmask]
        c_rar = rar[cmask]
        c_mask = mask[cmask]
        best_idx = (c_probs * c_mask).argmax(axis=1)
        n_s = len(c_probs)
        sel_rar = c_rar[np.arange(n_s), best_idx]
        fired = c_probs[np.arange(n_s), best_idx] > 0.5
        if fired.sum() > 0:
            wr = (sel_rar[fired] > 0).mean()
            avg = sel_rar[fired].mean()

            # Top 20% EV filtered
            sel_probs = c_probs[np.arange(n_s), best_idx]
            mfe_vals = probs[cmask]  # simplified — use prob as EV proxy
            evs = sel_probs * np.abs(sel_rar)
            if len(evs[fired]) > 50:
                threshold = np.percentile(evs[fired], 80)
                top20 = fired & (evs >= threshold)
                if top20.sum() > 0:
                    wr20 = (sel_rar[top20] > 0).mean()
                    avg20 = sel_rar[top20].mean()
                    print(f"    {cname:12s}: all={fired.sum():4d} WR={wr:.1%} RAR={avg:.4f} "
                          f"| top20%={top20.sum():3d} WR={wr20:.1%} RAR={avg20:.4f}")
                    continue

            print(f"    {cname:12s}: fired={fired.sum():4d} WR={wr:.1%} RAR={avg:.4f}")

    # 6. Save model
    save_dir = "models/universal_v6"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "universal_v6.pt")
    torch.save({
        "model_state": model.state_dict(),
        "partitions": partitions,
        "n_labels": n_labels,
        "feature_cols": feature_cols,
        "coin_map": {coin: i for i, coin in enumerate(coin_names)},
        "n_temporal": 32,
        "config": {
            "expert_hidden": 256, "expert_output": 128,
            "fusion_dim": 256, "n_moe_experts": 8, "moe_top_k": 2,
        },
    }, save_path)
    print(f"\n  Model saved: {save_path}")

    config = {
        "version": "v6_universal",
        "coins": coin_names,
        "n_features": len(feature_cols),
        "n_labels": n_labels,
        "n_params": n_params,
        "best_val_loss": float(best_val),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return {"best_val": best_val, "n_coins": len(datasets)}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    result = train_universal(device=device, epochs=50, batch_size=2048)
    if result:
        print(f"\nFinal: val_loss={result['best_val']:.4f}, coins={result['n_coins']}")
