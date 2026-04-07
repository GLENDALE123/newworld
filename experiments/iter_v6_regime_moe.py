"""
Experiment: PLE v6 Regime-Aware MoE Training

Test the new architecture on alpha coins.
Compare v4 (point-in-time) vs v6 (regime-aware MoE).

Goal: Verify that regime conditioning and MoE routing improve:
  1. Strategy selection accuracy
  2. Per-trade alpha
  3. Multi-strategy diversity (not just one strategy winning)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.factory_v2 import generate_features_v2
from features.temporal_context import generate_temporal_features
from labeling.multi_tbm_v2 import generate_multi_tbm_v2
from ple.model_v3 import partition_features
from ple.model_v6 import PLEv6
from ple.trainer_v6 import prepare_data_v6, train_ple_v6, compute_regime_ids
from execution.regime_detector import RegimeDetector


DATA_DIR = "data/merged"
# Test with proven alpha coins first
TEST_COINS = ["BCHUSDT", "UNIUSDT", "CRVUSDT"]


def load_kline_multi_tf(symbol: str) -> dict[str, pd.DataFrame]:
    """Load multi-timeframe klines for a symbol."""
    klines = {}
    for tf in ["15m"]:
        path = os.path.join(DATA_DIR, symbol, f"kline_{tf}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df = df.sort_index()
            klines[tf] = df
    return klines


def resample_kline(df_15m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 15m klines to target timeframe."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in agg if c in df_15m.columns]
    return df_15m[cols].resample(target_tf).agg({c: agg[c] for c in cols}).dropna()


def run_experiment(coin: str, device: str = "cuda"):
    """Run v6 training experiment on a single coin."""
    print(f"\n{'='*60}")
    print(f"PLE v6 Experiment: {coin}")
    print(f"{'='*60}")

    # 1. Load data
    print("\n[1] Loading data...")
    klines = load_kline_multi_tf(coin)
    if "15m" not in klines:
        print(f"  No 15m data for {coin}")
        return None

    df_15m = klines["15m"]
    print(f"  15m: {len(df_15m)} bars ({df_15m.index[0]} → {df_15m.index[-1]})")

    # Create multi-TF klines from 15m
    klines["1h"] = resample_kline(df_15m, "1h")
    klines["4h"] = resample_kline(df_15m, "4h")
    print(f"  1h:  {len(klines['1h'])} bars")
    print(f"  4h:  {len(klines['4h'])} bars")

    # 2. Generate features (base + temporal)
    print("\n[2] Generating features...")
    features = generate_features_v2(klines, target_tf="15min", progress=True)
    print(f"  Base features: {features.shape}")

    # Temporal context
    temporal = generate_temporal_features(df_15m)
    temporal_df = pd.DataFrame(temporal)
    temporal_df = temporal_df.reindex(features.index)
    temporal_df = temporal_df.replace([np.inf, -np.inf], np.nan)
    n_temporal = temporal_df.shape[1]
    print(f"  Temporal features: {n_temporal}")
    print(f"  Total features: {features.shape[1] + n_temporal}")

    # 3. Generate labels
    print("\n[3] Generating labels...")
    labels_by_strat = generate_multi_tbm_v2(
        {"15m": df_15m, "1h": klines["1h"], "4h": klines["4h"]},
        progress=True,
    )

    # Merge all strategy labels into one DataFrame aligned to 15m
    label_dfs = []
    for strat_name, ldf in labels_by_strat.items():
        if strat_name in ("scalp", "intraday"):
            # These use 5m/15m source → already aligned or close
            label_dfs.append(ldf)
        else:
            # Resample to 15m (forward fill from coarser TF)
            resampled = ldf.resample("15min").ffill() if not ldf.empty else ldf
            label_dfs.append(resampled)

    labels = pd.concat(label_dfs, axis=1)
    labels = labels.loc[labels.index.isin(features.index)]
    print(f"  Labels: {labels.shape}")

    # 4. Compute regime IDs
    print("\n[4] Computing regimes...")
    common = features.index.intersection(labels.index)
    regime_ids = compute_regime_ids(df_15m.reindex(common, method="ffill"), common)
    regime_dist = pd.Series(regime_ids).value_counts().sort_index()
    regime_names = {0: "surge", 1: "dump", 2: "range", 3: "volatile"}
    for rid, count in regime_dist.items():
        pct = count / len(regime_ids) * 100
        print(f"  {regime_names.get(rid, rid):10s}: {count:6d} ({pct:.1f}%)")

    # 5. Prepare datasets
    print("\n[5] Preparing datasets...")
    train_ds, val_ds, test_ds, partitions = prepare_data_v6(
        features.loc[common],
        labels.loc[common],
        df_15m.reindex(common, method="ffill"),
        temporal_df.reindex(common),
        coin_id=0,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Partitions: {', '.join(f'{k}={len(v)}' for k, v in partitions.items())}")

    # 6. Create model
    n_labels = labels.filter(like="tbm_").shape[1]
    n_groups = 4  # scalp, intraday, daytrade, swing
    n_per_group = n_labels // n_groups if n_labels >= n_groups else n_labels

    print(f"\n[6] Creating PLE v6...")
    print(f"  n_labels={n_labels}, n_groups={n_groups}, n_per_group={n_per_group}")
    print(f"  n_temporal_features={n_temporal}")

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

    # 7. Train
    print(f"\n[7] Training on {device}...")
    result = train_ple_v6(
        model, train_ds, val_ds,
        epochs=30,
        batch_size=2048,
        lr=5e-4,
        device=device,
        patience=7,
        rdrop_alpha=1.0,
        moe_balance_weight=0.01,
        seed=42,
    )

    print(f"\n  Best validation loss: {result['best_val']:.4f}")

    # 8. Evaluate on test set
    print("\n[8] Test set evaluation...")
    model.eval()
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    all_probs = []
    all_rar = []
    all_mask = []
    all_regimes = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(
                batch["features"], batch["coin_id"],
                batch["regime_id"], batch["account"],
                batch["temporal"],
            )
            all_probs.append(out["label_probs"].cpu().numpy())
            all_rar.append(batch["rar"].cpu().numpy())
            all_mask.append(batch["rar_mask"].cpu().numpy())
            all_regimes.append(batch["regime_id"].cpu().numpy())

    probs = np.concatenate(all_probs)
    rar = np.concatenate(all_rar)
    mask = np.concatenate(all_mask)
    regimes = np.concatenate(all_regimes)

    # Analyze per-regime accuracy
    print(f"\n  Per-regime analysis:")
    for rid, rname in regime_names.items():
        rmask = regimes == rid
        if rmask.sum() == 0:
            continue
        r_probs = probs[rmask]
        r_rar = rar[rmask]
        r_mask = mask[rmask]

        # Strategy that model picks (highest prob)
        best_idx = (r_probs * r_mask).argmax(axis=1)
        n_samples = len(r_probs)
        selected_rar = r_rar[np.arange(n_samples), best_idx]
        fired = r_probs[np.arange(n_samples), best_idx] > 0.5
        if fired.sum() > 0:
            avg_rar = selected_rar[fired].mean()
            wr = (selected_rar[fired] > 0).mean()
            print(f"    {rname:10s}: n={rmask.sum():5d}, fired={fired.sum():4d}, "
                  f"WR={wr:.1%}, avg_RAR={avg_rar:.4f}")
        else:
            print(f"    {rname:10s}: n={rmask.sum():5d}, fired=0")

    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}
    for coin in TEST_COINS:
        try:
            r = run_experiment(coin, device)
            if r:
                results[coin] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for coin, r in results.items():
        print(f"  {coin}: val_loss={r['best_val']:.4f}")
