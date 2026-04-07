"""
PLE v6 Trainer: Regime-Aware MoE training loop.

Changes from v4 trainer:
  - Adds regime_id and temporal context to batch
  - MoE load balance auxiliary loss
  - Regime-conditioned R-Drop
  - Coin ID support for universal training
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ple.model_v6 import PLEv6
from ple.loss_v6 import PLEv6Loss
from ple.loss_v4 import PLEv4Loss
from ple.model_v3 import partition_features
from execution.regime_detector import RegimeDetector, Regime


REGIME_MAP = {"surge": 0, "dump": 1, "range": 2, "volatile": 3}


class TradingDatasetV6(Dataset):
    """Dataset with regime, coin, and temporal context."""

    def __init__(
        self,
        features: np.ndarray,
        tbm: np.ndarray,
        mae: np.ndarray,
        mfe: np.ndarray,
        rar: np.ndarray,
        regime_ids: np.ndarray,
        coin_ids: np.ndarray | None = None,
        temporal: np.ndarray | None = None,
        account: np.ndarray | None = None,
        wgt: np.ndarray | None = None,
    ):
        self.features = torch.tensor(np.nan_to_num(features, 0.0), dtype=torch.float32)
        self.tbm = torch.tensor(np.nan_to_num((tbm + 1) / 2, nan=0.0), dtype=torch.float32)
        self.tbm_mask = torch.tensor(~np.isnan(tbm), dtype=torch.float32)
        self.mae = torch.tensor(np.nan_to_num(mae, 0.0), dtype=torch.float32)
        self.mfe = torch.tensor(np.nan_to_num(mfe, 0.0), dtype=torch.float32)
        self.rar = torch.tensor(np.nan_to_num(rar, 0.0), dtype=torch.float32)
        self.reg_mask = torch.tensor(~np.isnan(mae), dtype=torch.float32)
        self.rar_mask = torch.tensor(~np.isnan(rar), dtype=torch.float32)
        self.wgt = torch.tensor(np.nan_to_num(wgt, nan=1.0), dtype=torch.float32) if wgt is not None else torch.ones_like(self.rar)
        self.regime_ids = torch.tensor(regime_ids, dtype=torch.long)
        self.coin_ids = torch.tensor(coin_ids, dtype=torch.long) if coin_ids is not None else torch.zeros(len(features), dtype=torch.long)
        self.temporal = torch.tensor(np.nan_to_num(temporal, 0.0), dtype=torch.float32) if temporal is not None else torch.zeros(len(features), 40)
        self.account = torch.tensor(account, dtype=torch.float32) if account is not None else torch.zeros(len(features), 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "account": self.account[idx],
            "regime_id": self.regime_ids[idx],
            "coin_id": self.coin_ids[idx],
            "temporal": self.temporal[idx],
            "tbm": self.tbm[idx],
            "tbm_mask": self.tbm_mask[idx],
            "mae": self.mae[idx],
            "mfe": self.mfe[idx],
            "rar": self.rar[idx],
            "reg_mask": self.reg_mask[idx],
            "rar_mask": self.rar_mask[idx],
            "wgt": self.wgt[idx],
        }


def compute_regime_ids(
    kline_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Compute regime ID for each bar using RegimeDetector."""
    detector = RegimeDetector()
    regimes = detector.detect(
        kline_df["high"].values,
        kline_df["low"].values,
        kline_df["close"].values,
        kline_df["volume"].values,
    )

    # Map Regime enum to int
    regime_ids = np.array([
        REGIME_MAP.get(r.value if isinstance(r, Regime) else str(r), 2)
        for r in regimes
    ])

    # Align to target index
    regime_series = pd.Series(regime_ids, index=kline_df.index if hasattr(kline_df, 'index') else range(len(regime_ids)))
    aligned = regime_series.reindex(target_index, method="ffill").fillna(2).astype(int)
    return aligned.values


def prepare_data_v6(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    kline_df: pd.DataFrame,
    temporal_df: pd.DataFrame | None = None,
    coin_id: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Prepare v6 datasets with regime and temporal context."""
    common = features_df.index.intersection(labels_df.index)
    X = np.nan_to_num(features_df.loc[common].values, 0.0, 0.0, 0.0)
    L = labels_df.loc[common]

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])

    n = len(X)
    s1, s2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))

    # Regime IDs
    regime_ids = compute_regime_ids(kline_df, common)

    # Temporal context
    if temporal_df is not None:
        temporal = temporal_df.reindex(common).values
        temporal = np.nan_to_num(temporal, 0.0)
    else:
        temporal = np.zeros((n, 40))

    # Coin IDs
    coin_ids = np.full(n, coin_id, dtype=np.int64)

    # Account state (default)
    account = np.zeros((n, 4), dtype=np.float32)
    account[:, 0] = 1.0  # normalized equity

    def ds(a, b):
        return TradingDatasetV6(
            X[a:b], L[tbm_cols].values[a:b], L[mae_cols].values[a:b],
            L[mfe_cols].values[a:b], L[rar_cols].values[a:b],
            regime_ids[a:b], coin_ids[a:b], temporal[a:b], account[a:b],
        )

    partitions = partition_features(list(features_df.columns))
    return ds(0, s1), ds(s1, s2), ds(s2, n), partitions


def _kl_binary(p1, p2, mask):
    """Symmetric KL divergence for binary probabilities (R-Drop)."""
    eps = 1e-7
    p1, p2 = p1.clamp(eps, 1 - eps), p2.clamp(eps, 1 - eps)
    kl_1 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
    kl_2 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())
    return ((kl_1 + kl_2) / 2 * mask).sum() / mask.sum().clamp(1)


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2**32))


def train_ple_v6(
    model: PLEv6,
    train_ds: TradingDatasetV6,
    val_ds: TradingDatasetV6,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 5e-4,
    device: str = "cuda",
    patience: int = 7,
    rdrop_alpha: float = 1.0,
    moe_balance_weight: float = 0.01,
    seed: int = 42,
):
    """Train PLE v6 with regime-aware MoE."""
    model = model.to(device)
    # Use v4 loss (plain BCE) — model predicts accurately, router handles selectivity
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    total_steps = epochs * (len(train_ds) // batch_size + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
    )

    g = torch.Generator()
    g.manual_seed(seed)

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

    for epoch in range(epochs):
        model.train()
        epoch_moe_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Mixup
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]
                # For long tensors (regime_id, coin_id), keep original
                # (can't interpolate categorical IDs)

            # Forward pass (v6 signature)
            def forward_v6(batch):
                return model(
                    batch["features"], batch["coin_id"],
                    batch["regime_id"], batch["account"],
                    batch["temporal"],
                )

            # R-Drop
            if rdrop_alpha > 0:
                out1 = forward_v6(batch)
                out2 = forward_v6(batch)
                l1 = loss_fn(out1, batch)
                l2 = loss_fn(out2, batch)
                task_loss = (l1["total"] + l2["total"]) / 2
                rdrop_loss = _kl_binary(out1["label_probs"], out2["label_probs"], batch["rar_mask"])
                total = task_loss + rdrop_alpha * rdrop_loss

                # MoE load balance loss
                moe_lb = model.moe_load_balance_loss(out1["moe_indices"])
                total = total + moe_balance_weight * moe_lb
                epoch_moe_loss += moe_lb.item()
            else:
                out1 = forward_v6(batch)
                losses = loss_fn(out1, batch)
                total = losses["total"]
                moe_lb = model.moe_load_balance_loss(out1["moe_indices"])
                total = total + moe_balance_weight * moe_lb
                epoch_moe_loss += moe_lb.item()

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            n_batches += 1

        # Validation
        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(
                    batch["features"], batch["coin_id"],
                    batch["regime_id"], batch["account"],
                    batch["temporal"],
                )
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

        v = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        moe_avg = epoch_moe_loss / max(n_batches, 1)

        # MoE routing stats
        with torch.no_grad():
            sb = next(iter(val_loader))
            sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sb.items()}
            so = model(sb["features"], sb["coin_id"], sb["regime_id"], sb["account"], sb["temporal"])
            gw = so["gate_weights"].cpu().numpy().mean(0)
            moe_idx = so["moe_indices"].cpu().numpy().flatten()
            expert_usage = np.bincount(moe_idx, minlength=len(model.moe_experts))
            expert_pct = expert_usage / expert_usage.sum() * 100

        gate_str = " ".join(f"{w:.2f}" for w in gw)
        expert_str = " ".join(f"{p:.0f}%" for p in expert_pct)

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  "
              f"bce={v['L_label']:.3f}  cal={v['L_cal']:.4f}  "
              f"active={v['n_active']:.1f}  prec={v['precision']:.2f}  "
              f"no_trade={v['no_trade_pct']:.1%}  "
              f"moe_lb={moe_avg:.4f}  "
              f"gate=[{gate_str}]  "
              f"experts=[{expert_str}]")

        if v["total"] < best_val:
            best_val = v["total"]
            no_improve = 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return {"best_val": best_val}
