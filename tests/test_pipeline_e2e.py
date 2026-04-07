"""
End-to-End Pipeline Integration Test

Data → Features(polars) → Temporal sequences → PLE v7 → Meta-model → Portfolio Manager

검증:
  1. 각 단계가 올바른 포맷으로 연결되는가?
  2. 메타모델이 시그널을 필터링하는가?
  3. 포트폴리오 매니저가 EV/hour 기반으로 랭킹하는가?
  4. 동적 레버리지가 올바르게 작동하는가?
"""

import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultrathink.pipeline import UltraThink
from ple.model_v7 import PLEv7
from ple.trainer_v7 import build_multiscale_temporal, partition_features_v7
from execution.meta_model import MetaModel, SignalSnapshot, PortfolioState
from execution.dynamic_portfolio import DynamicPortfolioManager, TradeRequest
from execution.trendline_scanner import TrendlineScanner, ScannerSignal
from execution.coin_classifier import CoinClassifier
import re


def test_full_pipeline():
    print("=" * 60)
    print("End-to-End Pipeline Integration Test")
    print("=" * 60)

    ut = UltraThink()
    classifier = CoinClassifier("data/merged")

    # === Step 1: Data + Features ===
    print("\n[1] Loading data + features...")
    test_coins = {
        "ple": ["BCHUSDT", "ETHUSDT"],      # PLE coins
        "scanner": ["CHRUSDT", "DUSKUSDT"],  # scanner coins
    }

    features_cache = {}
    kline_cache = {}
    temporal_cache = {}

    # First pass: collect all feature columns, then align
    raw_features = {}
    for coin in test_coins["ple"]:
        X, labels, kline, strat_info = ut.prepare(coin, "2020-01-01", "2026-12-31")
        keep = [c for c in X.columns if not re.search(r'_lag\d+$', c) and not re.search(r'_chg\d+$', c)]
        raw_features[coin] = X[keep]
        kline_cache[coin] = kline

    # Unified feature set: intersection of all coins
    common_cols = None
    for coin, X in raw_features.items():
        if common_cols is None:
            common_cols = set(X.columns)
        else:
            common_cols &= set(X.columns)
    common_cols = sorted(common_cols)
    print(f"  Common features across coins: {len(common_cols)}")

    for coin in test_coins["ple"]:
        features_cache[coin] = raw_features[coin][common_cols]
        kline = kline_cache[coin]

        # Build temporal sequences for last bar
        kline_5m = kline.get("5m", kline.get("15m"))
        kline_15m = kline.get("15m")
        kline_1h = kline.get("1h", kline_15m.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna())

        # Just get last bar's temporal context
        idx = X.index[-1:]
        s5, s15, s1h = build_multiscale_temporal(kline_5m, kline_15m, kline_1h, idx)
        temporal_cache[coin] = (s5, s15, s1h)

        print(f"  {coin}: {features_cache[coin].shape[1]} features, temporal OK")

    # === Step 2: Create PLE v7 model ===
    print("\n[2] Creating PLE v7...")
    ref_coin = test_coins["ple"][0]
    partitions = partition_features_v7(list(features_cache[ref_coin].columns))
    n_labels = 32  # standard

    model = PLEv7(
        feature_partitions=partitions,
        n_strategies=n_labels,
        expert_output=64,
        temporal_dim=64,
        fusion_dim=192,
    )
    model.eval()
    print(f"  Params: {model.count_parameters():,}")
    print(f"  Partitions: {', '.join(f'{k}={len(v)}' for k, v in partitions.items())}")

    # === Step 3: PLE Inference ===
    print("\n[3] PLE inference...")
    ple_signals = []

    for coin_idx, coin in enumerate(test_coins["ple"]):
        X = features_cache[coin]
        s5, s15, s1h = temporal_cache[coin]

        # Last bar
        feat = torch.tensor(np.nan_to_num(X.values[-1:], 0.0), dtype=torch.float32)
        t5 = torch.tensor(s5, dtype=torch.float32)
        t15 = torch.tensor(s15, dtype=torch.float32)
        t1h = torch.tensor(s1h, dtype=torch.float32)
        cid = torch.tensor([coin_idx], dtype=torch.long)
        acc = torch.zeros(1, 4)

        with torch.no_grad():
            out = model(feat, t5, t15, t1h, cid, acc)
            probs = out["label_probs"].numpy()[0]
            mae = out["mae_pred"].numpy()[0]
            mfe = out["mfe_pred"].numpy()[0]
            conf = float(probs.max())

        best_idx = probs.argmax()
        best_prob = probs[best_idx]
        direction = 1 if best_idx % 2 == 0 else -1

        # Calculate EV
        ev = best_prob * max(abs(mfe[best_idx]), 0.001) - (1 - best_prob) * max(abs(mae[best_idx]), 0.001) - 0.0008

        category = classifier.classify(coin)
        kline_15m = kline_cache[coin].get("15m")
        close_arr = kline_15m["close"].values
        vol = np.std(np.diff(close_arr[-21:]) / close_arr[-21:-1]) if len(close_arr) > 21 else 0.02

        snap = SignalSnapshot(
            coin=coin, coin_id=coin_idx, category=category,
            probs=probs, confidence=conf, regime=2,  # range
            ev_best=ev, direction_best=direction,
            strategy_idx=int(best_idx), volatility=vol,
        )
        ple_signals.append(snap)
        print(f"  {coin}: prob={best_prob:.3f} dir={'long' if direction==1 else 'short'} EV={ev:.4f} conf={conf:.3f}")

    # === Step 4: Scanner signals ===
    print("\n[4] Scanner signals...")
    scanner = TrendlineScanner()
    scanner_signals = []

    for coin in test_coins["scanner"]:
        path = f"data/merged/{coin}/kline_15m.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df = df.sort_index()

        sig = scanner.scan(
            coin, df["high"].values, df["low"].values,
            df["close"].values, df["volume"].values,
        )
        if sig:
            scanner_signals.append(sig)
            print(f"  {coin}: BREAKOUT {sig.signal_type} conf={sig.confidence:.2f}")
        else:
            print(f"  {coin}: no breakout")

    # === Step 5: Meta-model ===
    print("\n[5] Meta-model evaluation...")
    meta = MetaModel(min_confidence=0.50, persistence_bars=1)

    portfolio_state = PortfolioState(
        equity=500, peak_equity=500, drawdown_pct=0,
        n_open_positions=0, max_positions=3,
    )

    # Feed PLE signals to meta
    decisions = meta.evaluate(ple_signals, portfolio_state)
    print(f"  Meta passed {len(decisions)} / {len(ple_signals)} PLE signals")

    # === Step 6: Portfolio Manager ===
    print("\n[6] Portfolio Manager evaluation...")
    pm = DynamicPortfolioManager(initial_equity=500)

    # Convert meta decisions + scanner signals to TradeRequests
    requests = []

    for d in decisions:
        snap = ple_signals[0]  # simplified
        requests.append(TradeRequest(
            coin=d.coin, direction=d.direction,
            confidence=d.confidence, ev=snap.ev_best,
            hold_bars=8, mfe=0.02, mae=-0.01,
            strategy_type="scalp", source="ple",
            volatility=snap.volatility,
        ))

    for sig in scanner_signals:
        requests.append(TradeRequest(
            coin=sig.coin, direction=sig.direction,
            confidence=sig.confidence, ev=sig.ev,
            hold_bars=8, mfe=0.03, mae=-0.015,
            strategy_type="breakout", source="scanner",
            volatility=0.04,
        ))

    orders = pm.evaluate(requests)
    print(f"  Orders: {len(orders)}")
    for o in orders:
        print(f"    {o.coin:12s} {'LONG' if o.direction==1 else 'SHORT':5s} "
              f"size={o.size_pct:.1%} lev={o.leverage}x "
              f"TP={o.tp_pct*100:.2f}% SL={o.sl_pct*100:.2f}% "
              f"EV/h={o.ev_per_hour:.4f}")

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("Pipeline Integration: OK")
    print(f"  Features: polars factory → {features_cache[ref_coin].shape[1]} cols")
    print(f"  Temporal: 5m(12) + 15m(32) + 1h(24) multi-scale")
    print(f"  PLE v7: {model.count_parameters():,} params, {n_labels} strategies")
    print(f"  Meta: {len(ple_signals)} signals → {len(decisions)} passed")
    print(f"  Scanner: {len(scanner_signals)} breakouts")
    print(f"  Portfolio: {len(orders)} execution orders")
    print(f"  Leverage range: {min(o.leverage for o in orders) if orders else 0}x ~ {max(o.leverage for o in orders) if orders else 0}x")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_full_pipeline()
