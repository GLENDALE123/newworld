"""
PLE v7b Trainer: v4 trainer + coin_id support

v4 trainer에서 coin_id만 추가.
Dataset에 coin_id 포함, forward에 전달.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ple.model_v7b import PLEv7b
from ple.loss_v4 import PLEv4Loss
from ple.model_v3 import partition_features


class TradingDatasetV7b(Dataset):
    """v4 dataset + coin_id."""

    def __init__(self, features, tbm, mae, mfe, rar, account=None, wgt=None, coin_id=0):
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
        self.coin_id = torch.full((len(features),), coin_id, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx], "account": self.account[idx],
            "coin_id": self.coin_id[idx],
            "tbm": self.tbm[idx], "tbm_mask": self.tbm_mask[idx],
            "mae": self.mae[idx], "mfe": self.mfe[idx],
            "rar": self.rar[idx], "reg_mask": self.reg_mask[idx],
            "rar_mask": self.rar_mask[idx], "wgt": self.wgt[idx],
        }


def _kl_binary(p1, p2, mask):
    eps = 1e-7
    p1, p2 = p1.clamp(eps, 1 - eps), p2.clamp(eps, 1 - eps)
    kl_1 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
    kl_2 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())
    return ((kl_1 + kl_2) / 2 * mask).sum() / mask.sum().clamp(1)


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2**32))


def train_ple_v7b(model, train_ds, val_ds, epochs=50, batch_size=2048,
                   lr=5e-4, device="cuda", patience=7, rdrop_alpha=1.0, seed=42):
    """v4 training loop with coin_id passed to model."""
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * (len(train_ds) // batch_size + 1))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=12,
                              pin_memory=True, persistent_workers=True,
                              generator=g, worker_init_fn=_worker_init)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True,
                            worker_init_fn=_worker_init)

    best_val = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Mixup (float tensors only)
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]

            # R-Drop with coin_id
            if rdrop_alpha > 0:
                out1 = model(batch["features"], batch["account"], batch["coin_id"])
                out2 = model(batch["features"], batch["account"], batch["coin_id"])
                l1, l2 = loss_fn(out1, batch), loss_fn(out2, batch)
                task_loss = (l1["total"] + l2["total"]) / 2
                rdrop_loss = _kl_binary(out1["label_probs"], out2["label_probs"], batch["rar_mask"])
                total = task_loss + rdrop_alpha * rdrop_loss
            else:
                out1 = model(batch["features"], batch["account"], batch["coin_id"])
                total = loss_fn(out1, batch)["total"]

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch["features"], batch["account"], batch["coin_id"])
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

            sb = next(iter(val_loader))
            sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sb.items()}
            so = model(sb["features"], sb["account"], sb["coin_id"])
            gw = so["gate_weights"].cpu().numpy().mean(0)

        v = {k: np.mean([m[k] for m in vm]) for k in vm[0]}
        gate_str = " ".join(f"{w:.2f}" for w in gw)

        print(f"  E{epoch+1:02d}  loss={v['total']:.3f}  "
              f"bce={v['L_label']:.3f}  cal={v['L_cal']:.4f}  "
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
