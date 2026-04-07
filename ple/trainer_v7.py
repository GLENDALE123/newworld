"""
PLE v7 Trainer: 1D CNN temporal + adaptive experts

Key changes from v4 trainer:
  - Temporal sequence (recent 32 bars OHLCV) as additional input
  - Coin ID for universal training
  - No confidence head loss
  - lag features 불필요 (CNN이 대체)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ple.model_v7 import PLEv7
from ple.loss_v4 import PLEv4Loss


def partition_features_v7(feature_names: list[str]) -> dict[str, list[int]]:
    """Partition features for v7. Updated for polars factory naming.

    Partitions:
      price: 5m/15m/1h/4h price/ATR/vol/position features
      flow: order flow (CVD, buy_ratio, delta, intensity)
      metrics: derivatives (OI, LS ratio, funding)
      cross: cross-timeframe features
    """
    partitions = {"price": [], "flow": [], "metrics": [], "cross": []}

    for i, name in enumerate(feature_names):
        nl = name.lower()
        if nl.startswith("xtf_"):
            partitions["cross"].append(i)
        elif nl.startswith("flow_"):
            partitions["flow"].append(i)
        elif nl.startswith(("deriv_", "fund_")):
            partitions["metrics"].append(i)
        else:
            partitions["price"].append(i)

    # Remove empty partitions
    return {k: v for k, v in partitions.items() if len(v) > 0}


class TradingDatasetV7(Dataset):
    """Dataset with multi-scale temporal sequences for 1D CNN."""

    def __init__(
        self,
        features: np.ndarray,       # (N, F) point-in-time features
        seq_5m: np.ndarray,          # (N, 12, 6) recent 5m bars
        seq_15m: np.ndarray,         # (N, 32, 6) recent 15m bars
        seq_1h: np.ndarray,          # (N, 24, 6) recent 1h bars
        tbm: np.ndarray,
        mae: np.ndarray,
        mfe: np.ndarray,
        rar: np.ndarray,
        coin_ids: np.ndarray | None = None,
        account: np.ndarray | None = None,
        wgt: np.ndarray | None = None,
    ):
        self.features = torch.tensor(np.nan_to_num(features, 0.0), dtype=torch.float32)
        self.seq_5m = torch.tensor(np.nan_to_num(seq_5m, 0.0), dtype=torch.float32)
        self.seq_15m = torch.tensor(np.nan_to_num(seq_15m, 0.0), dtype=torch.float32)
        self.seq_1h = torch.tensor(np.nan_to_num(seq_1h, 0.0), dtype=torch.float32)
        self.tbm = torch.tensor(np.nan_to_num((tbm + 1) / 2, nan=0.0), dtype=torch.float32)
        self.tbm_mask = torch.tensor(~np.isnan(tbm), dtype=torch.float32)
        self.mae = torch.tensor(np.nan_to_num(mae, 0.0), dtype=torch.float32)
        self.mfe = torch.tensor(np.nan_to_num(mfe, 0.0), dtype=torch.float32)
        self.rar = torch.tensor(np.nan_to_num(rar, 0.0), dtype=torch.float32)
        self.reg_mask = torch.tensor(~np.isnan(mae), dtype=torch.float32)
        self.rar_mask = torch.tensor(~np.isnan(rar), dtype=torch.float32)
        self.wgt = torch.tensor(np.nan_to_num(wgt, nan=1.0), dtype=torch.float32) if wgt is not None else torch.ones_like(self.rar)
        self.coin_ids = torch.tensor(coin_ids, dtype=torch.long) if coin_ids is not None else torch.zeros(len(features), dtype=torch.long)
        self.account = torch.tensor(account, dtype=torch.float32) if account is not None else torch.zeros(len(features), 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "seq_5m": self.seq_5m[idx],
            "seq_15m": self.seq_15m[idx],
            "seq_1h": self.seq_1h[idx],
            "coin_id": self.coin_ids[idx],
            "account": self.account[idx],
            "tbm": self.tbm[idx],
            "tbm_mask": self.tbm_mask[idx],
            "mae": self.mae[idx],
            "mfe": self.mfe[idx],
            "rar": self.rar[idx],
            "reg_mask": self.reg_mask[idx],
            "rar_mask": self.rar_mask[idx],
            "wgt": self.wgt[idx],
        }


def _build_one_tf_sequence(
    kline_df: pd.DataFrame,
    seq_len: int,
) -> np.ndarray:
    """Build temporal sequences for one timeframe.

    Returns: (N, seq_len, 6) normalized array
    """
    close = kline_df["close"].values.astype(np.float64)
    high = kline_df["high"].values.astype(np.float64)
    low = kline_df["low"].values.astype(np.float64)
    opn = kline_df["open"].values.astype(np.float64)
    vol = kline_df["volume"].values.astype(np.float64)
    n = len(close)

    sequences = np.zeros((n, seq_len, 6), dtype=np.float32)

    for i in range(seq_len, n):
        start = i - seq_len
        wc = close[start:i]
        bp = wc[0]
        if bp <= 0:
            continue

        sequences[i, :, 0] = (opn[start:i] - bp) / bp
        sequences[i, :, 1] = (high[start:i] - bp) / bp
        sequences[i, :, 2] = (low[start:i] - bp) / bp
        sequences[i, :, 3] = (wc - bp) / bp

        vm = vol[start:i].mean()
        if vm > 0:
            sequences[i, :, 4] = vol[start:i] / vm

        rets = np.diff(wc, prepend=wc[0]) / np.maximum(wc, 1e-8)
        sequences[i, :, 5] = rets

    return sequences


def build_multiscale_temporal(
    kline_5m: pd.DataFrame,
    kline_15m: pd.DataFrame,
    kline_1h: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build multi-scale temporal sequences aligned to target index.

    Returns: (seq_5m, seq_15m, seq_1h) each aligned to target_index
      seq_5m:  (N, 12, 6) — 1시간
      seq_15m: (N, 32, 6) — 8시간
      seq_1h:  (N, 24, 6) — 24시간
    """
    raw_5m = _build_one_tf_sequence(kline_5m, seq_len=12)
    raw_15m = _build_one_tf_sequence(kline_15m, seq_len=32)
    raw_1h = _build_one_tf_sequence(kline_1h, seq_len=24)

    N = len(target_index)

    def align(raw, kline_idx, seq_len):
        aligned = np.zeros((N, seq_len, 6), dtype=np.float32)
        for i, ts in enumerate(target_index):
            loc = kline_idx.searchsorted(ts)
            if 0 <= loc < len(raw):
                aligned[i] = raw[loc]
        return aligned

    s5 = align(raw_5m, kline_5m.index, 12)
    s15 = align(raw_15m, kline_15m.index, 32)
    s1h = align(raw_1h, kline_1h.index, 24)

    return s5, s15, s1h


def prepare_data_v7(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    kline_dict: dict[str, pd.DataFrame],
    coin_id: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Prepare v7 datasets with multi-scale temporal sequences."""
    common = features_df.index.intersection(labels_df.index)
    X = np.nan_to_num(features_df.loc[common].values, 0.0, 0.0, 0.0)
    L = labels_df.loc[common]

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])

    n = len(X)
    s1, s2 = int(n * train_ratio), int(n * (train_ratio + val_ratio))

    # Build multi-scale temporal sequences
    kline_5m = kline_dict.get("5m", kline_dict.get("15m"))
    kline_15m = kline_dict.get("15m", list(kline_dict.values())[0])
    kline_1h = kline_dict.get("1h", kline_15m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna())

    seq_5m, seq_15m, seq_1h = build_multiscale_temporal(
        kline_5m, kline_15m, kline_1h, common,
    )

    coin_ids = np.full(n, coin_id, dtype=np.int64)
    account = np.zeros((n, 4), dtype=np.float32)
    account[:, 0] = 1.0

    def ds(a, b):
        return TradingDatasetV7(
            X[a:b], seq_5m[a:b], seq_15m[a:b], seq_1h[a:b],
            L[tbm_cols].values[a:b], L[mae_cols].values[a:b],
            L[mfe_cols].values[a:b], L[rar_cols].values[a:b],
            coin_ids[a:b], account[a:b],
        )

    partitions = partition_features_v7(list(features_df.columns))
    return ds(0, s1), ds(s1, s2), ds(s2, n), partitions


def _kl_binary(p1, p2, mask):
    eps = 1e-7
    p1, p2 = p1.clamp(eps, 1 - eps), p2.clamp(eps, 1 - eps)
    kl_1 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
    kl_2 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())
    return ((kl_1 + kl_2) / 2 * mask).sum() / mask.sum().clamp(1)


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2**32))


def train_ple_v7(
    model: PLEv7,
    train_ds: TradingDatasetV7,
    val_ds: TradingDatasetV7,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 5e-4,
    device: str = "cuda",
    patience: int = 7,
    rdrop_alpha: float = 1.0,
    seed: int = 42,
):
    model = model.to(device)
    loss_fn = PLEv4Loss(n_losses=4).to(device)

    # Remove confidence loss weight since v7 has no confidence head
    # PLEv4Loss has 4 losses: label, cal, equity (disabled), conf
    # We still use it but conf loss will be 0 since there's no confidence output

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
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Mixup (float tensors only)
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(batch["features"].size(0), device=device)
                for k in batch:
                    if isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.float32:
                        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]

            def fwd(b):
                out = model(b["features"], b["seq_5m"], b["seq_15m"], b["seq_1h"],
                            b["coin_id"], b["account"])
                out["confidence"] = out["label_probs"].max(dim=1).values
                return out

            # R-Drop
            if rdrop_alpha > 0:
                out1 = fwd(batch)
                out2 = fwd(batch)
                l1, l2 = loss_fn(out1, batch), loss_fn(out2, batch)
                task_loss = (l1["total"] + l2["total"]) / 2
                rdrop = _kl_binary(out1["label_probs"], out2["label_probs"], batch["rar_mask"])
                total = task_loss + rdrop_alpha * rdrop
            else:
                out1 = fwd(batch)
                total = loss_fn(out1, batch)["total"]

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch["features"], batch["seq_5m"], batch["seq_15m"],
                            batch["seq_1h"], batch["coin_id"], batch["account"])
                out["confidence"] = out["label_probs"].max(dim=1).values
                losses = loss_fn(out, batch)
                vm.append({k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in losses.items() if k != "task_weights"})

            # Gate weights
            sb = next(iter(val_loader))
            sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sb.items()}
            so = model(sb["features"], sb["seq_5m"], sb["seq_15m"],
                       sb["seq_1h"], sb["coin_id"], sb["account"])
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
