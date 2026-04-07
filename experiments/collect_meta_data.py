"""
Meta-Model Training Data Collection

PLE 범용모델로 모든 코인에서 시그널 생성 → 실제 결과와 매칭.
이 데이터로 메타모델이 "어떤 시그널을 신뢰할지" 학습.

출력 데이터:
  - coin: 코인명
  - coin_id: 코인 ID
  - timestamp: 시그널 시점
  - best_prob: 최고 확률
  - best_strategy: 최고 전략 인덱스
  - direction: 방향 (1=long, -1=short)
  - ev: 예상 EV
  - mfe_pred: 예상 MFE
  - mae_pred: 예상 MAE
  - actual_return_1h: 실제 1시간 후 수익률
  - actual_return_4h: 실제 4시간 후 수익률
  - actual_return_1d: 실제 24시간 후 수익률
  - was_profitable: 실제로 수익이었는가
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultrathink.pipeline import UltraThink
from ple.model_v7b import PLEv7b
from ple.trainer_v7b import TradingDatasetV7b, train_ple_v7b
from ple.model_v3 import partition_features
from torch.utils.data import ConcatDataset


DATA_DIR = "data/merged"


def collect(coins: list[str], device: str = "cuda"):
    ut = UltraThink()

    # 1. Prepare data for all coins
    print(f"[1] Loading {len(coins)} coins...")
    all_data = {}
    common_cols = None

    for coin in coins:
        try:
            X, labels, kline, _ = ut.prepare(coin, "2020-01-01", "2026-12-31")
            all_data[coin] = (X, labels, kline)
            common_cols = set(X.columns) if common_cols is None else common_cols & set(X.columns)
        except Exception as e:
            print(f"  {coin}: skip ({e})")

    common_cols = sorted(common_cols)
    partitions = partition_features(common_cols)
    print(f"  {len(all_data)} coins loaded, {len(common_cols)} common features")

    # 2. Train universal PLE
    print(f"\n[2] Training universal PLE...")
    train_sets, val_sets = [], []
    coin_list = list(all_data.keys())

    for cid, coin in enumerate(coin_list):
        X, labels, kline = all_data[coin]
        X = X[common_cols]
        common = X.index.intersection(labels.index)
        Xv = np.nan_to_num(X.loc[common].values, 0.0)
        L = labels.loc[common]
        tc = sorted([c for c in L.columns if c.startswith("tbm_")])
        mc = sorted([c for c in L.columns if c.startswith("mae_")])
        fc = sorted([c for c in L.columns if c.startswith("mfe_")])
        rc = sorted([c for c in L.columns if c.startswith("rar_")])
        n = len(Xv)
        s1 = int(n * 0.6)
        s2 = int(n * 0.8)
        acc = np.zeros((n, 4), dtype=np.float32)
        acc[:, 0] = 1.0

        train_sets.append(TradingDatasetV7b(
            Xv[:s1], L[tc].values[:s1], L[mc].values[:s1],
            L[fc].values[:s1], L[rc].values[:s1], acc[:s1], coin_id=cid,
        ))
        val_sets.append(TradingDatasetV7b(
            Xv[s1:s2], L[tc].values[s1:s2], L[mc].values[s1:s2],
            L[fc].values[s1:s2], L[rc].values[s1:s2], acc[s1:s2], coin_id=cid,
        ))

    model = PLEv7b(
        feature_partitions=partitions,
        n_coins=len(coin_list),
        n_strategies=32,
        expert_hidden=128,
        expert_output=96,
        fusion_dim=192,
    )
    train_ple_v7b(
        model, ConcatDataset(train_sets), ConcatDataset(val_sets),
        epochs=30, device=device, patience=7,
    )

    # 3. Collect signals on TEST set for all coins
    print(f"\n[3] Collecting meta-model training data...")
    model.eval()
    meta_rows = []

    for cid, coin in enumerate(coin_list):
        X, labels, kline = all_data[coin]
        X = X[common_cols]
        common = X.index.intersection(labels.index)
        n = len(common)
        test_start = int(n * 0.8)
        test_idx = common[test_start:]

        Xv = np.nan_to_num(X.loc[test_idx].values, 0.0).astype(np.float32)
        kline_15m = kline["15m"]
        close = kline_15m["close"].reindex(test_idx, method="ffill").values

        with torch.no_grad():
            for i in range(0, len(Xv) - 96, 4):  # every 1h, need 24h forward
                x = torch.tensor(Xv[i : i + 1], dtype=torch.float32).to(device)
                a = torch.zeros(1, 4).to(device)
                c = torch.tensor([cid], dtype=torch.long).to(device)
                o = model(x, a, c)

                probs = o["label_probs"].cpu().numpy()[0]
                mfe_pred = o["mfe_pred"].cpu().numpy()[0]
                mae_pred = o["mae_pred"].cpu().numpy()[0]

                best = probs.argmax()
                p = probs[best]
                direction = 1 if best % 2 == 0 else -1
                ev = p * max(abs(mfe_pred[best]), 0.001) - (1 - p) * max(abs(mae_pred[best]), 0.001) - 0.0008

                # Actual returns
                ret_1h = direction * (close[min(i + 4, len(close) - 1)] - close[i]) / close[i]
                ret_4h = direction * (close[min(i + 16, len(close) - 1)] - close[i]) / close[i]
                ret_1d = direction * (close[min(i + 96, len(close) - 1)] - close[i]) / close[i]

                meta_rows.append({
                    "coin": coin,
                    "coin_id": cid,
                    "timestamp": str(test_idx[i]),
                    "best_prob": float(p),
                    "best_strategy": int(best),
                    "direction": direction,
                    "ev": float(ev),
                    "mfe_pred": float(mfe_pred[best]),
                    "mae_pred": float(mae_pred[best]),
                    "n_active": int((probs > 0.5).sum()),
                    "prob_std": float(probs.std()),
                    "actual_return_1h": float(ret_1h),
                    "actual_return_4h": float(ret_4h),
                    "actual_return_1d": float(ret_1d),
                    "was_profitable_1h": ret_1h > 0.0008,
                    "was_profitable_4h": ret_4h > 0.0008,
                })

        print(f"  {coin}: {len([r for r in meta_rows if r['coin'] == coin])} signals")

    # 4. Save
    df = pd.DataFrame(meta_rows)
    os.makedirs("data/meta", exist_ok=True)
    df.to_parquet("data/meta/signal_outcomes.parquet", index=False)
    print(f"\n[4] Saved: {len(df)} rows → data/meta/signal_outcomes.parquet")

    # Quick stats
    print(f"\n=== Signal Quality by Coin ===")
    for coin in coin_list:
        cd = df[df["coin"] == coin]
        wr_1h = cd["was_profitable_1h"].mean() * 100
        wr_4h = cd["was_profitable_4h"].mean() * 100
        avg_ev = cd["ev"].mean()
        print(f"  {coin:12s}: {len(cd):5d} signals, WR_1h={wr_1h:.1f}%, WR_4h={wr_4h:.1f}%, avg_EV={avg_ev:.4f}")

    return df


if __name__ == "__main__":
    coins = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT",
        "BCHUSDT", "UNIUSDT", "DUSKUSDT",
        "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT",
    ]
    collect(coins)
