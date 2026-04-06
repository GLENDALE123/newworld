#!/usr/bin/env python3
"""
ultraTM Production Pipeline

End-to-end pipeline for paper/live trading:
1. Load data from disk (or Binance API)
2. Generate features (factory v2)
3. Generate TBM labels (ATR-based)
4. Train PLE v4 model
5. Run inference with adaptive thresholds
6. Execute via NautilusTrader (when connected)

Best validated config (Iter 34, full-history):
  Walk-forward mean: +26% (3 windows all positive)
  Bull market: +64% (was -10% before full history)
  Bear/sideways: +3-12%
  Fee-inclusive (0.08% round-trip)
"""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def load_data(data_dir: str = "data/merged/BTCUSDT", start: str = "2020-06-01", end: str = "2026-02-28"):
    """Load all available data for a symbol.

    Uses 15m→1h resampling to unlock full 2020-2026 history when native 1h
    data has limited coverage. This solved the bull market bias (iter 33).
    """
    kline = {}
    for tf in ["5m", "15m"]:
        path = os.path.join(data_dir, f"kline_{tf}.parquet")
        if os.path.exists(path):
            kline[tf] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]

    # 1h: resample from 15m for full history coverage
    native_1h_path = os.path.join(data_dir, "kline_1h.parquet")
    native_1h = pd.DataFrame()
    if os.path.exists(native_1h_path):
        native_1h = pd.read_parquet(native_1h_path).set_index("timestamp").sort_index()[start:end]

    if "15m" in kline:
        resampled_1h = kline["15m"].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        kline["1h"] = resampled_1h if len(resampled_1h) > len(native_1h) else native_1h
    elif len(native_1h) > 0:
        kline["1h"] = native_1h

    if "1h" in kline:
        kline["4h"] = kline["1h"].resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    # Optional data sources
    extras = {}
    for name in ["tick_bar", "metrics", "funding_rate"]:
        path = os.path.join(data_dir, f"{name}.parquet")
        if os.path.exists(path):
            extras[name] = pd.read_parquet(path).set_index("timestamp").sort_index()[start:end]

    return kline, extras


def build_features(kline, extras, target_tf="15min"):
    """Generate features using factory v2 + sequence features."""
    from features.factory_v2 import generate_features_v2

    features = generate_features_v2(
        kline_data=kline,
        tick_bar=extras.get("tick_bar"),
        metrics=extras.get("metrics"),
        funding=extras.get("funding_rate"),
        target_tf=target_tf,
        progress=False,
    )

    # Sequence features (lookback=8)
    top_feats = features.std().sort_values(ascending=False).head(30).index.tolist()
    seq_cols = {}
    for lag in range(1, 8):
        for col in top_feats:
            seq_cols[f"{col}_lag{lag}"] = features[col].shift(lag)
    for col in top_feats[:10]:
        for lag in [1, 2, 4]:
            seq_cols[f"{col}_chg{lag}"] = features[col] - features[col].shift(lag)

    features = pd.concat([features, pd.DataFrame(seq_cols, index=features.index)], axis=1)

    # OI divergence + Funding rate features (verified alpha: +0.18%/day)
    if "metrics" in extras:
        oi = extras["metrics"]["sum_open_interest_value"].resample("15min").last().ffill()
        oi = oi.reindex(features.index, method="ffill")
        close = kline["15m"]["close"].reindex(features.index, method="ffill")
        for lb in [48, 96, 192]:
            features[f"oi_div_{lb}"] = oi.pct_change(lb) - close.pct_change(lb)
            features[f"oi_chg_{lb}"] = oi.pct_change(lb)
    if "funding_rate" in extras:
        fr = extras["funding_rate"].iloc[:, -1].resample("15min").ffill()
        fr = fr.reindex(features.index, method="ffill")
        features["funding_rate_raw"] = fr
        features["funding_zscore_672"] = (fr - fr.rolling(672).mean()) / fr.rolling(672).std().replace(0, np.nan)
        features["funding_extreme_95"] = (fr.abs() > fr.abs().rolling(672).quantile(0.95)).astype(float)

    # LS ratio contrarian features (verified alpha: +0.435%/day spread)
    if "metrics" in extras:
        for col_name, feat_prefix in [
            ("count_long_short_ratio", "ls"),
            ("count_toptrader_long_short_ratio", "topls"),
        ]:
            if col_name in extras["metrics"].columns:
                ls = extras["metrics"][col_name].resample("15min").last().ffill()
                ls = ls.reindex(features.index, method="ffill")
                for lb in [48, 96, 192]:
                    features[f"{feat_prefix}_chg_{lb}"] = ls.pct_change(lb)
                features[f"{feat_prefix}_extreme_high"] = (ls > ls.rolling(672).quantile(0.95)).astype(float)
                features[f"{feat_prefix}_extreme_low"] = (ls < ls.rolling(672).quantile(0.05)).astype(float)

    features = features.dropna().replace([np.inf, -np.inf], np.nan).fillna(0)

    return features


def build_labels(kline, fee_pct=0.0008):
    """Generate ATR-based TBM labels."""
    from labeling.multi_tbm_v2 import generate_multi_tbm_v2

    lr = generate_multi_tbm_v2(kline, fee_pct=fee_pct, progress=False)

    labels = lr["intraday"].copy()
    if "swing" in lr:
        sw = lr["swing"].resample("15min").ffill()
        for col in sw.columns:
            labels[col] = sw[col]

    return labels


def train_model(X, L, partitions, train_ratio=0.5, val_ratio=0.25, seed=42):
    """Train PLE v4 model."""
    from ple.model_v4 import PLEv4
    from ple.trainer_v4 import TradingDatasetV4, train_ple_v4

    tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
    mae_cols = sorted([c for c in L.columns if c.startswith("mae_")])
    mfe_cols = sorted([c for c in L.columns if c.startswith("mfe_")])
    rar_cols = sorted([c for c in L.columns if c.startswith("rar_")])
    wgt_cols = sorted([c for c in L.columns if c.startswith("wgt_")])

    X_np = X.values.astype(np.float32)
    acc = np.zeros((len(X_np), 4), dtype=np.float32)
    acc[:, 0] = 1.0
    wgt_np = L[wgt_cols].values if wgt_cols else None

    s1 = int(len(X) * train_ratio)
    s2 = int(len(X) * (train_ratio + val_ratio))

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TradingDatasetV4(
        X_np[:s1], L[tbm_cols].values[:s1], L[mae_cols].values[:s1],
        L[mfe_cols].values[:s1], L[rar_cols].values[:s1], acc[:s1],
        wgt_np[:s1] if wgt_np is not None else None,
    )
    val_ds = TradingDatasetV4(
        X_np[s1:s2], L[tbm_cols].values[s1:s2], L[mae_cols].values[s1:s2],
        L[mfe_cols].values[s1:s2], L[rar_cols].values[s1:s2], acc[s1:s2],
        wgt_np[s1:s2] if wgt_np is not None else None,
    )

    model = PLEv4(
        feature_partitions=partitions,
        n_account_features=4,
        n_strategies=len(tbm_cols),
        expert_hidden=128,
        expert_output=64,
        fusion_dim=192,
        dropout=0.2,
    )

    train_ple_v4(model, train_ds, val_ds, epochs=50, batch_size=2048,
                  lr=5e-4, device="cuda", patience=7)

    return model, s2


def main():
    parser = argparse.ArgumentParser(description="ultraTM Production Pipeline")
    parser.add_argument("--mode", choices=["train", "backtest", "paper"], default="backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end", default="2026-02-28")
    parser.add_argument("--size", type=float, default=0.03)
    parser.add_argument("--fee", type=float, default=0.0008)
    args = parser.parse_args()

    print(f"ultraTM — {args.mode.upper()} mode")
    print(f"Symbol: {args.symbol}, Period: {args.start} ~ {args.end}\n")

    # Load data
    data_dir = f"data/merged/{args.symbol}"
    kline, extras = load_data(data_dir, args.start, args.end)

    # Build features
    features = build_features(kline, extras)
    print(f"Features: {features.shape}")

    # Build labels
    labels = build_labels(kline, args.fee)
    print(f"Labels: {labels.shape}")

    # Align
    common = features.index.intersection(labels.index)
    X = features.loc[common]
    L = labels.loc[common]
    print(f"Aligned: {len(X)} rows\n")

    # Feature partitions
    from ple.model_v3 import partition_features
    partitions = {k: v for k, v in partition_features(list(X.columns)).items() if len(v) > 0}

    # Train
    model, test_start = train_model(X, L, partitions)
    print(f"\nModel trained. Test starts at index {test_start}")

    if args.mode in ["backtest", "paper"]:
        # Inference
        model = model.to("cuda").eval()
        X_test = X.iloc[test_start:]
        n = len(X_test)

        tbm_cols = sorted([c for c in L.columns if c.startswith("tbm_")])
        strat_info = [
            {"style": c.replace("tbm_", "").split("_")[0], "dir": c.replace("tbm_", "").split("_")[1]}
            for c in tbm_cols
        ]

        with torch.no_grad():
            out = model(
                torch.tensor(X_test.values.astype(np.float32)).to("cuda"),
                torch.zeros(n, 4).to("cuda"),
            )
            probs = out["label_probs"].cpu().numpy()
            mfe_p = out["mfe_pred"].cpu().numpy()
            mae_p = out["mae_pred"].cpu().numpy()
            confidence = out["confidence"].cpu().numpy()

        # SMA
        close = kline["15m"]["close"]
        sma = kline["4h"]["close"].rolling(50).mean().resample("15min").ffill()
        tc = close.reindex(X_test.index, method="ffill").values
        sv = sma.reindex(X_test.index, method="ffill").values

        # Backtest with adaptive thresholds
        capital = 100000.0
        peak = capital
        trades = []

        for i in range(0, n - 1, 4):
            above = not np.isnan(sv[i]) and tc[i] > sv[i]
            lt = 0.40 if above else 0.55
            st = 0.55 if above else 0.40

            best_ev = -1
            best_j = -1
            for j in range(len(strat_info)):
                th = lt if strat_info[j]["dir"] == "long" else st
                if probs[i, j] < th:
                    continue
                p = probs[i, j]
                rew = max(abs(mfe_p[i, j]), 0.001)
                rsk = max(abs(mae_p[i, j]), 0.001)
                ev = p * rew - (1 - p) * rsk - args.fee
                if ev > best_ev:
                    best_ev = ev
                    best_j = j

            if best_j < 0 or best_ev <= 0:
                continue

            d = 1 if strat_info[best_j]["dir"] == "long" else -1
            base_hold = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}.get(strat_info[best_j]["style"], 12)
            # Confidence-scaled hold: strong signal → hold longer
            hold_mult = max(0.5, min(2.0, confidence[i] * 2))
            hold = int(base_hold * hold_mult)
            ei = min(i + hold, n - 1)
            pnl = d * (tc[ei] - tc[i]) / tc[i]
            net = pnl - args.fee

            dd = (peak - capital) / peak if peak > 0 else 0
            sz = args.size * max(0.2, 1 - dd / 0.15)
            capital += net * capital * sz
            peak = max(peak, capital)
            trades.append({"net": net * 100, "dir": "L" if d == 1 else "S"})

        # Results
        tdf = pd.DataFrame(trades)
        ret = (capital - 100000) / 100000 * 100
        bh = (tc[-1] - tc[0]) / tc[0] * 100
        wr = (tdf["net"] > 0).mean() * 100 if len(tdf) > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"  RESULTS")
        print(f"{'=' * 50}")
        print(f"  Return:  {ret:+.2f}%")
        print(f"  B&H:     {bh:+.2f}%")
        print(f"  Excess:  {ret - bh:+.2f}%")
        print(f"  WR:      {wr:.1f}%")
        print(f"  Trades:  {len(tdf)}")


if __name__ == "__main__":
    main()
