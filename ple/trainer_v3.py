"""PLE v3 Trainer."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ple.model_v3 import PLEv3, partition_features
from ple.loss_v3 import PLEv3Loss


class TradingDatasetV3(Dataset):
    def __init__(self, features: np.ndarray, tbm: np.ndarray, mae: np.ndarray,
                 mfe: np.ndarray, rar: np.ndarray, account: np.ndarray | None = None):
        self.features = torch.tensor(np.nan_to_num(features, 0.0), dtype=torch.float32)
        self.tbm = torch.tensor(np.nan_to_num((tbm + 1) / 2, nan=0.0), dtype=torch.float32)
        self.tbm_mask = torch.tensor(~np.isnan(tbm), dtype=torch.float32)
        self.mae = torch.tensor(np.nan_to_num(mae, 0.0), dtype=torch.float32)
        self.mfe = torch.tensor(np.nan_to_num(mfe, 0.0), dtype=torch.float32)
        self.rar = torch.tensor(np.nan_to_num(rar, 0.0), dtype=torch.float32)
        self.reg_mask = torch.tensor(~np.isnan(mae), dtype=torch.float32)
        self.rar_mask = torch.tensor(~np.isnan(rar), dtype=torch.float32)

        if account is not None:
            self.account = torch.tensor(np.nan_to_num(account, 0.0), dtype=torch.float32)
        else:
            self.account = torch.zeros(len(features), 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "account": self.account[idx],
            "tbm": self.tbm[idx],
            "tbm_mask": self.tbm_mask[idx],
            "mae": self.mae[idx],
            "mfe": self.mfe[idx],
            "rar": self.rar[idx],
            "reg_mask": self.reg_mask[idx],
            "rar_mask": self.rar_mask[idx],
        }


def prepare_data_v3(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[TradingDatasetV3, TradingDatasetV3, TradingDatasetV3, dict]:
    common = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common].values
    L = labels_df.loc[common]

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(X)
    s1 = int(n * train_ratio)
    s2 = int(n * (train_ratio + val_ratio))

    # Simulate account state (equity_pct, drawdown, consec_losses, position_count)
    # In production, these would be real account data
    account = np.zeros((n, 4), dtype=np.float32)
    account[:, 0] = 1.0  # equity at 100%

    partitions = partition_features(list(features_df.columns))

    def make_ds(start, end):
        return TradingDatasetV3(
            X[start:end],
            L[tbm_cols].values[start:end],
            L[mae_cols].values[start:end],
            L[mfe_cols].values[start:end],
            L[rar_cols].values[start:end],
            account[start:end],
        )

    return make_ds(0, s1), make_ds(s1, s2), make_ds(s2, n), partitions


def train_ple_v3(
    model: PLEv3,
    train_ds: TradingDatasetV3,
    val_ds: TradingDatasetV3,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 5e-4,
    device: str = "cuda",
    patience: int = 7,
) -> dict:
    model = model.to(device)
    loss_fn = PLEv3Loss(n_losses=5).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=epochs * (len(train_ds) // batch_size + 1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val = float("inf")
    no_improve = 0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch["features"], batch["account"])
            losses = loss_fn(outputs, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch["features"], batch["account"])
                losses = loss_fn(outputs, batch)
                val_metrics.append({k: v.item() if isinstance(v, torch.Tensor) else v
                                    for k, v in losses.items() if k != "task_weights"})

        v = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}

        # Gate monitoring
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
            sample_out = model(sample_batch["features"], sample_batch["account"])
            gw = sample_out["gate_weights"].cpu().numpy()
            gate_str = " ".join(f"{w:.2f}" for w in gw.mean(axis=0))

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  "
              f"reg={v['oracle_regime_acc']:.2f}  tf={v['oracle_tf_acc']:.2f}  rr={v['oracle_rr_acc']:.2f}  "
              f"conf={v['confidence']:.3f}  gate=[{gate_str}]  "
              f"ent={v['gate_entropy']:.3f}")

        history.append(v)

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
    return {"history": history, "best_val": best_val}
