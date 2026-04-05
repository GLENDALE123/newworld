"""
PLE Training Pipeline

Handles: data loading, multi-task loss, training loop, evaluation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ple.model import PLETradingModel


class MultiLabelDataset(Dataset):
    """Dataset for PLE: features + 400 labels (100 TBM + 100 MAE + 100 MFE + 100 RAR)."""

    def __init__(self, features: np.ndarray, tbm: np.ndarray, mae: np.ndarray,
                 mfe: np.ndarray, rar: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        # TBM: convert +1/-1 to 0/1, NaN to -1 (ignore in loss)
        tbm_clean = np.where(np.isnan(tbm), -1, (tbm + 1) / 2)  # +1->1, -1->0, NaN->-1
        self.tbm = torch.tensor(tbm_clean, dtype=torch.float32)
        # Regression targets: NaN to 0 with mask
        self.mae = torch.tensor(np.nan_to_num(mae, nan=0.0), dtype=torch.float32)
        self.mfe = torch.tensor(np.nan_to_num(mfe, nan=0.0), dtype=torch.float32)
        self.rar = torch.tensor(np.nan_to_num(rar, nan=0.0), dtype=torch.float32)
        # Masks: 1 where label exists, 0 where NaN
        self.tbm_mask = torch.tensor(~np.isnan(tbm), dtype=torch.float32)
        self.reg_mask = torch.tensor(~np.isnan(mae), dtype=torch.float32)  # same mask for all reg

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "tbm": self.tbm[idx],
            "mae": self.mae[idx],
            "mfe": self.mfe[idx],
            "rar": self.rar[idx],
            "tbm_mask": self.tbm_mask[idx],
            "reg_mask": self.reg_mask[idx],
        }


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting (Kendall et al. 2018).

    Each task has a learnable log_sigma parameter. The model learns to balance
    classification and regression losses automatically.
    """

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        # Learnable log(sigma^2) per task — initialized to 0 (equal weight)
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        # TBM classification loss (BCE with safe masking)
        tbm_pred = outputs["tbm_probs"]
        tbm_target = batch["tbm"].clamp(0, 1)  # ensure [0,1]
        tbm_mask = batch["tbm_mask"]

        # Only compute BCE where mask is active
        bce = F.binary_cross_entropy(
            tbm_pred * tbm_mask + 0.5 * (1 - tbm_mask),  # inactive → 0.5 (no gradient)
            tbm_target * tbm_mask + 0.5 * (1 - tbm_mask),
            reduction="none",
        )
        tbm_loss = (bce * tbm_mask).sum() / tbm_mask.sum().clamp(min=1)

        # Regression losses (MSE with mask)
        reg_mask = batch["reg_mask"]
        mae_loss = ((outputs["mae_pred"] - batch["mae"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(min=1)
        mfe_loss = ((outputs["mfe_pred"] - batch["mfe"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(min=1)
        rar_loss = ((outputs["rar_pred"] - batch["rar"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(min=1)

        # Uncertainty weighting: L_total = sum(1/(2*sigma_i^2) * L_i + log(sigma_i))
        losses = [tbm_loss, mae_loss, mfe_loss, rar_loss]
        total = torch.tensor(0.0, device=tbm_pred.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma_sq[i])
            total = total + precision * loss + self.log_sigma_sq[i]

        return {
            "total": total,
            "tbm_loss": tbm_loss,
            "mae_loss": mae_loss,
            "mfe_loss": mfe_loss,
            "rar_loss": rar_loss,
            "task_weights": torch.exp(-self.log_sigma_sq).detach(),
        }


import torch.nn.functional as F


def prepare_datasets(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[MultiLabelDataset, MultiLabelDataset, MultiLabelDataset]:
    """Split features + labels into train/val/test datasets."""
    # Align
    common = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common].values
    L = labels_df.loc[common]

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])

    tbm = L[tbm_cols].values
    mae = L[mae_cols].values
    mfe = L[mfe_cols].values
    rar = L[rar_cols].values

    # Replace inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Temporal split
    n = len(X)
    s1 = int(n * train_ratio)
    s2 = int(n * (train_ratio + val_ratio))

    train_ds = MultiLabelDataset(X[:s1], tbm[:s1], mae[:s1], mfe[:s1], rar[:s1])
    val_ds = MultiLabelDataset(X[s1:s2], tbm[s1:s2], mae[s1:s2], mfe[s1:s2], rar[s1:s2])
    test_ds = MultiLabelDataset(X[s2:], tbm[s2:], mae[s2:], mfe[s2:], rar[s2:])

    return train_ds, val_ds, test_ds


def train_ple(
    model: PLETradingModel,
    train_ds: MultiLabelDataset,
    val_ds: MultiLabelDataset,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "cuda",
    patience: int = 5,
) -> dict:
    """Train PLE model with early stopping."""
    model = model.to(device)
    loss_fn = UncertaintyWeightedLoss(n_tasks=4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "tbm_loss": [], "val_tbm_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["features"])
            losses = loss_fn(outputs, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(losses["total"].item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["features"])
                losses = loss_fn(outputs, batch)
                val_losses.append(losses["total"].item())

                # TBM accuracy (on valid labels only)
                preds = (outputs["tbm_probs"] > 0.5).float()
                mask = batch["tbm_mask"]
                correct = ((preds == batch["tbm"]) * mask).sum()
                total = mask.sum()
                val_correct += correct.item()
                val_total += total.item()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_tbm_acc"].append(val_acc)

        print(f"  Epoch {epoch+1:2d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  tbm_acc={val_acc:.3f}  lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best
    model.load_state_dict(best_state)
    return history
