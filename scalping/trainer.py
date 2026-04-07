"""Scalping model trainer with walk-forward validation."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scalping.model import ScalpingMLP


class ScalpDataset(Dataset):
    def __init__(self, features, label_long, label_short,
                 mfe_long, mfe_short, mae_long, mae_short,
                 bars_long, bars_short):
        def _c(arr):
            return np.array(arr, dtype=np.float64, copy=True)
        self.features = torch.tensor(np.nan_to_num(_c(features), 0.0), dtype=torch.float32)
        # Labels: +1/-1 → 1/0 for BCE
        ll = _c(label_long); ls = _c(label_short)
        self.mask_long = torch.tensor(~np.isnan(ll), dtype=torch.float32)
        self.mask_short = torch.tensor(~np.isnan(ls), dtype=torch.float32)
        self.label_long = torch.tensor((np.nan_to_num(ll, nan=0.0) + 1) / 2, dtype=torch.float32)
        self.label_short = torch.tensor((np.nan_to_num(ls, nan=0.0) + 1) / 2, dtype=torch.float32)
        self.mfe_long = torch.tensor(np.nan_to_num(_c(mfe_long), 0.0), dtype=torch.float32)
        self.mfe_short = torch.tensor(np.nan_to_num(_c(mfe_short), 0.0), dtype=torch.float32)
        self.mae_long = torch.tensor(np.nan_to_num(_c(mae_long), 0.0), dtype=torch.float32)
        self.mae_short = torch.tensor(np.nan_to_num(_c(mae_short), 0.0), dtype=torch.float32)
        self.bars_long = torch.tensor(np.nan_to_num(_c(bars_long), 0.0), dtype=torch.float32)
        self.bars_short = torch.tensor(np.nan_to_num(_c(bars_short), 0.0), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label_long": self.label_long[idx],
            "label_short": self.label_short[idx],
            "mfe_long": self.mfe_long[idx],
            "mfe_short": self.mfe_short[idx],
            "mae_long": self.mae_long[idx],
            "mae_short": self.mae_short[idx],
            "bars_long": self.bars_long[idx],
            "bars_short": self.bars_short[idx],
            "mask_long": self.mask_long[idx],
            "mask_short": self.mask_short[idx],
        }


def scalp_loss(out, batch):
    """Combined loss: BCE for direction + MSE for MFE/MAE/bars."""
    bce = nn.functional.binary_cross_entropy

    # Direction loss (BCE)
    loss_dir = (
        bce(out["prob_long"], batch["label_long"], reduction="none") * batch["mask_long"]
    ).sum() / batch["mask_long"].sum().clamp(1) + (
        bce(out["prob_short"], batch["label_short"], reduction="none") * batch["mask_short"]
    ).sum() / batch["mask_short"].sum().clamp(1)

    # MFE/MAE regression (only on valid labels)
    mse = nn.functional.mse_loss

    loss_mfe = (
        ((out["mfe_long"] - batch["mfe_long"]) ** 2 * batch["mask_long"]).sum() / batch["mask_long"].sum().clamp(1) +
        ((out["mfe_short"] - batch["mfe_short"]) ** 2 * batch["mask_short"]).sum() / batch["mask_short"].sum().clamp(1)
    )

    loss_mae = (
        ((out["mae_long"] - batch["mae_long"]) ** 2 * batch["mask_long"]).sum() / batch["mask_long"].sum().clamp(1) +
        ((out["mae_short"] - batch["mae_short"]) ** 2 * batch["mask_short"]).sum() / batch["mask_short"].sum().clamp(1)
    )

    loss_bars = (
        ((out["bars_long"] - batch["bars_long"]) ** 2 * batch["mask_long"]).sum() / batch["mask_long"].sum().clamp(1) +
        ((out["bars_short"] - batch["bars_short"]) ** 2 * batch["mask_short"]).sum() / batch["mask_short"].sum().clamp(1)
    )

    total = loss_dir + 0.5 * loss_mfe + 0.5 * loss_mae + 0.2 * loss_bars
    return {
        "total": total,
        "dir": float(loss_dir),
        "mfe": float(loss_mfe),
        "mae": float(loss_mae),
        "bars": float(loss_bars),
    }


def train_scalp_model(
    model: ScalpingMLP,
    train_ds: ScalpDataset,
    val_ds: ScalpDataset,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cuda",
    patience: int = 7,
    seed: int = 42,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=epochs * (len(train_ds) // batch_size + 1),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(wid):
        np.random.seed(seed + wid)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )

    best_val = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["features"])
            losses = scalp_loss(out, batch)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["features"])
                losses = scalp_loss(out, batch)
                val_losses.append(losses)

        avg = {k: float(np.mean([float(l[k]) for l in val_losses])) for k in val_losses[0]}

        if avg["total"] < best_val:
            best_val = avg["total"]
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:02d} loss={avg['total']:.4f} "
                  f"dir={avg['dir']:.4f} mfe={avg['mfe']:.4f} "
                  f"mae={avg['mae']:.4f} bars={avg['bars']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return {"best_val": best_val, "epochs": epoch + 1}
