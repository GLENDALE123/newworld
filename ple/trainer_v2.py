"""
PLE v2 Trainer with proper dataset, training loop, and monitoring.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ple.model_v2 import PLEv2
from ple.loss_v2 import PLEv2Loss


class TradingDataset(Dataset):
    """Dataset for PLE v2: features + 100 labels (TBM/MAE/MFE/RAR)."""

    def __init__(self, features: np.ndarray, tbm: np.ndarray, mae: np.ndarray,
                 mfe: np.ndarray, rar: np.ndarray):
        self.features = torch.tensor(np.nan_to_num(features, 0.0), dtype=torch.float32)

        # TBM: +1->1, -1->0, NaN stays as 0 (masked out)
        tbm_clean = np.nan_to_num((tbm + 1) / 2, nan=0.0)
        self.tbm = torch.tensor(tbm_clean, dtype=torch.float32)
        self.tbm_mask = torch.tensor(~np.isnan(tbm), dtype=torch.float32)

        self.mae = torch.tensor(np.nan_to_num(mae, 0.0), dtype=torch.float32)
        self.mfe = torch.tensor(np.nan_to_num(mfe, 0.0), dtype=torch.float32)
        self.rar = torch.tensor(np.nan_to_num(rar, 0.0), dtype=torch.float32)
        self.reg_mask = torch.tensor(~np.isnan(mae), dtype=torch.float32)
        self.rar_mask = torch.tensor(~np.isnan(rar), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "tbm": self.tbm[idx],
            "tbm_mask": self.tbm_mask[idx],
            "mae": self.mae[idx],
            "mfe": self.mfe[idx],
            "rar": self.rar[idx],
            "reg_mask": self.reg_mask[idx],
            "rar_mask": self.rar_mask[idx],
        }


def prepare_data(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[TradingDataset, TradingDataset, TradingDataset]:
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

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(X)
    s1 = int(n * train_ratio)
    s2 = int(n * (train_ratio + val_ratio))

    return (
        TradingDataset(X[:s1], tbm[:s1], mae[:s1], mfe[:s1], rar[:s1]),
        TradingDataset(X[s1:s2], tbm[s1:s2], mae[s1:s2], mfe[s1:s2], rar[s1:s2]),
        TradingDataset(X[s2:], tbm[s2:], mae[s2:], mfe[s2:], rar[s2:]),
    )


def train_ple_v2(
    model: PLEv2,
    train_ds: TradingDataset,
    val_ds: TradingDataset,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 5e-4,
    device: str = "cuda",
    patience: int = 7,
) -> dict:
    model = model.to(device)
    loss_fn = PLEv2Loss(n_losses=5).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * (len(train_ds) // batch_size + 1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_metrics = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["features"])
            losses = loss_fn(outputs, batch)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_metrics.append({k: v.item() if isinstance(v, torch.Tensor) else v
                                  for k, v in losses.items() if k != "task_weights"})

        # ── Validate ──
        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["features"])
                losses = loss_fn(outputs, batch)
                val_metrics.append({k: v.item() if isinstance(v, torch.Tensor) else v
                                    for k, v in losses.items() if k != "task_weights"})

        # ── Aggregate ──
        t = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0]}
        v = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}

        # Gate weights monitoring
        with torch.no_grad():
            task_w = loss_fn.log_sigma_sq.exp().neg().exp().cpu().numpy()

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  sel={v['L_select']:.3f}  "
              f"cal={v['L_calibrate']:.4f}  div={v['L_diversity']:.3f}  "
              f"eq={v['L_equity']:.3f}  conf={v['mean_confidence']:.3f}  "
              f"oracle={v['oracle_acc']:.3f}  w={task_w}")

        history.append({"epoch": epoch + 1, **v})

        # Early stopping
        if v["total"] < best_val:
            best_val = v["total"]
            no_improve = 0
            best_state = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_val}
