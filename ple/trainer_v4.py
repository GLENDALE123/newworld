"""PLE v4 Trainer: Multi-label training loop."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ple.model_v4 import PLEv4
from ple.loss_v4 import PLEv4Loss
from ple.model_v3 import partition_features


class TradingDatasetV4(Dataset):
    def __init__(self, features, tbm, mae, mfe, rar, account=None, wgt=None):
        self.features = torch.tensor(np.nan_to_num(features, 0.0), dtype=torch.float32)
        self.tbm = torch.tensor(np.nan_to_num((tbm + 1) / 2, nan=0.0), dtype=torch.float32)
        self.tbm_mask = torch.tensor(~np.isnan(tbm), dtype=torch.float32)
        self.mae = torch.tensor(np.nan_to_num(mae, 0.0), dtype=torch.float32)
        self.mfe = torch.tensor(np.nan_to_num(mfe, 0.0), dtype=torch.float32)
        self.rar = torch.tensor(np.nan_to_num(rar, 0.0), dtype=torch.float32)
        self.reg_mask = torch.tensor(~np.isnan(mae), dtype=torch.float32)
        self.rar_mask = torch.tensor(~np.isnan(rar), dtype=torch.float32)
        self.wgt = torch.tensor(np.nan_to_num(wgt, nan=1.0), dtype=torch.float32) if wgt is not None else torch.ones_like(self.rar)
        self.account = torch.tensor(account, dtype=torch.float32) if account is not None else torch.zeros(len(features), 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx], "account": self.account[idx],
            "tbm": self.tbm[idx], "tbm_mask": self.tbm_mask[idx],
            "mae": self.mae[idx], "mfe": self.mfe[idx],
            "rar": self.rar[idx], "reg_mask": self.reg_mask[idx],
            "rar_mask": self.rar_mask[idx], "wgt": self.wgt[idx],
        }


def prepare_data_v4(features_df, labels_df, train_ratio=0.6, val_ratio=0.2):
    common = features_df.index.intersection(labels_df.index)
    X = np.nan_to_num(features_df.loc[common].values, 0.0, 0.0, 0.0)
    L = labels_df.loc[common]

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])

    n = len(X)
    s1, s2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    account = np.zeros((n, 4), dtype=np.float32)
    account[:, 0] = 1.0

    def ds(a, b):
        return TradingDatasetV4(X[a:b], L[tbm_cols].values[a:b], L[mae_cols].values[a:b],
                                 L[mfe_cols].values[a:b], L[rar_cols].values[a:b], account[a:b])

    partitions = partition_features(list(features_df.columns))
    return ds(0, s1), ds(s1, s2), ds(s2, n), partitions


def train_ple_v4(model, train_ds, val_ds, epochs=50, batch_size=2048,
                  lr=5e-4, device="cuda", patience=7):
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * (len(train_ds) // batch_size + 1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Mixup: blend pairs of samples (50% chance per batch)
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]

            out = model(batch["features"], batch["account"])
            losses = loss_fn(out, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch["features"], batch["account"])
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

            # Gate weights
            sb = next(iter(val_loader))
            sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sb.items()}
            so = model(sb["features"], sb["account"])
            gw = so["gate_weights"].cpu().numpy().mean(0)

        v = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        gate_str = " ".join(f"{w:.2f}" for w in gw)

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  "
              f"bce={v['L_label']:.3f}  cal={v['L_cal']:.4f}  "
              f"eq={v['L_equity']:.3f}  "
              f"active={v['n_active']:.1f}  prec={v['precision']:.2f}  "
              f"no_trade={v['no_trade_pct']:.1%}  "
              f"gate=[{gate_str}]")

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
