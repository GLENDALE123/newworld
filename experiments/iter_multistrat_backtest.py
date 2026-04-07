"""
Multi-Strategy Backtest — Validate StrategyRouter vs single-strategy baseline

Compare:
  A) Old approach: single threshold, trend-following only, top 20% EV filter
  B) New approach: StrategyRouter with regime detection, multi-strategy selection

Uses existing saved models (production_v5) on alpha coins.
Measures: WR, per-trade alpha, drawdown, number of trades.

"다중전략이 단일전략보다 실제로 나은가?"
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ple.model_v4 import PLEv4
from ple.model_v3 import partition_features
from ultrathink.pipeline import UltraThink
from execution.regime_detector import RegimeDetector, Regime
from execution.strategy_router import StrategyRouter, LABEL_INDEX, N_LABELS
from execution.coin_classifier import CoinClassifier
from execution.signal_filter import SignalFilter
from execution.capital_strategy import CapitalStrategy


MODEL_DIR = "models/production_v5"
DATA_DIR = "data/merged"

_ultrathink = UltraThink(data_dir=DATA_DIR)


def load_model(coin: str, device: str = "cpu"):
    """Load saved production model."""
    path = Path(MODEL_DIR) / f"{coin.lower()}.pt"
    if not path.exists():
        return None, None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    pt = checkpoint["partitions"]
    ns = checkpoint["n_strategies"]

    model = PLEv4(
        feature_partitions=pt,
        n_account_features=4,
        n_strategies=ns,
        expert_hidden=256,
        expert_output=128,
        fusion_dim=256,
        dropout=0.2,
        use_vsn=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    return model, checkpoint


def load_data(coin: str):
    """Load ALL available data and generate full features via UltraThink pipeline."""
    kline_path = Path(DATA_DIR) / coin / "kline_15m.parquet"
    if not kline_path.exists():
        return None, None

    # Use UltraThink for proper feature generation (includes lag features)
    features = _ultrathink.features(coin, "2020-01-01", "2026-12-31", target_tf="15min")

    # Load 15m kline for price data
    df_15m = pd.read_parquet(kline_path)
    if "timestamp" in df_15m.columns:
        df_15m = df_15m.set_index("timestamp")
    if df_15m.index.tz is not None:
        df_15m.index = df_15m.index.tz_localize(None)
    df_15m = df_15m.sort_index()

    return df_15m, features


def backtest_single_strategy(model, features_df, kline_df, checkpoint, coin: str):
    """Baseline: single threshold, trend-following approach (old method)."""
    device = next(model.parameters()).device
    feature_cols = checkpoint["feature_cols"]
    strat_info = checkpoint["strat_info"]

    # Align features to model's expected columns
    available = [c for c in feature_cols if c in features_df.columns]
    missing = [c for c in feature_cols if c not in features_df.columns]
    if len(available) < len(feature_cols) * 0.5:
        return None

    # Prepare feature matrix (fill NaN/inf to prevent model NaN output)
    aligned = features_df.reindex(columns=feature_cols, fill_value=0.0)
    X = np.nan_to_num(aligned.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Test set: last 20%
    n = len(X)
    test_start = int(n * 0.8)
    X_test = X[test_start:]
    test_idx = features_df.index[test_start:]
    close_test = kline_df["close"].reindex(test_idx, method="ffill").values

    trades = []
    signal_filter = SignalFilter(top_pct=0.20)
    check_interval = 4  # check every 1h (4 × 15m bars)

    for i in range(0, len(X_test), check_interval):
        x = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
        acc = torch.zeros(1, 4, device=device)

        with torch.no_grad():
            out = model(x, acc)
            probs = out["label_probs"].cpu().numpy()[0]
            mfe_pred = out["mfe_pred"].cpu().numpy()[0]
            mae_pred = out["mae_pred"].cpu().numpy()[0]

        # Old approach: pick highest probability strategy
        best_idx = probs.argmax()
        best_prob = probs[best_idx]
        if best_prob < 0.5:
            continue

        # Calculate EV
        rew = max(abs(mfe_pred[best_idx]), 0.001)
        rsk = max(abs(mae_pred[best_idx]), 0.001)
        ev = best_prob * rew - (1 - best_prob) * rsk - 0.0008

        # Record EV for filter then check
        signal_filter.ev_history.setdefault(coin, __import__('collections').deque(maxlen=500))
        signal_filter.ev_history[coin].append(ev)

        if not signal_filter.should_trade(coin, ev):
            continue

        if ev <= 0:
            continue

        # Simple forward return (4 bars = 1h)
        if i + 4 < len(close_test):
            entry_price = close_test[i]
            exit_price = close_test[i + 4]
            if best_idx < len(strat_info):
                direction = 1 if strat_info[best_idx]["dir"] == "long" else -1
            else:
                direction = 1
            ret = direction * (exit_price - entry_price) / entry_price - 0.0008

            trades.append({
                "timestamp": test_idx[i],
                "direction": direction,
                "probability": float(best_prob),
                "ev": float(ev),
                "return": float(ret),
                "strategy": "single_best",
            })

    return trades


def backtest_multi_strategy(model, features_df, kline_df, checkpoint, coin: str):
    """New approach: StrategyRouter with regime detection."""
    device = next(model.parameters()).device
    feature_cols = checkpoint["feature_cols"]
    strat_info = checkpoint["strat_info"]

    # Align features (fill NaN/inf)
    aligned = features_df.reindex(columns=feature_cols, fill_value=0.0)
    X = np.nan_to_num(aligned.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Test set
    n = len(X)
    test_start = int(n * 0.8)
    X_test = X[test_start:]
    test_idx = features_df.index[test_start:]

    close_test = kline_df["close"].reindex(test_idx, method="ffill").values
    high_test = kline_df["high"].reindex(test_idx, method="ffill").values
    low_test = kline_df["low"].reindex(test_idx, method="ffill").values
    vol_test = kline_df["volume"].reindex(test_idx, method="ffill").values

    classifier = CoinClassifier(DATA_DIR)
    category = classifier.classify(coin)
    regime_detector = RegimeDetector()
    router = StrategyRouter(min_ev=0.0001)
    signal_filter = SignalFilter(top_pct=0.20)

    trades = []
    lookback = 200
    check_interval = classifier.get_strategy(coin).get("check_interval", 4)

    for i in range(lookback, len(X_test), check_interval):
        # Regime detection
        regime_state = regime_detector.detect_current(
            coin,
            high_test[max(0, i-lookback):i+1],
            low_test[max(0, i-lookback):i+1],
            close_test[max(0, i-lookback):i+1],
            vol_test[max(0, i-lookback):i+1],
        )

        # Model inference
        x = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(device)
        acc = torch.zeros(1, 4, device=device)

        with torch.no_grad():
            out = model(x, acc)
            probs = out["label_probs"].cpu().numpy()[0]
            mfe_pred = out["mfe_pred"].cpu().numpy()[0]
            mae_pred = out["mae_pred"].cpu().numpy()[0]
            conf = out["confidence"].cpu().numpy()[0]

        # Route through StrategyRouter
        # Map model's label indices to router's label count
        n_model_labels = len(probs)
        n_router_labels = N_LABELS
        if n_model_labels < n_router_labels:
            probs_padded = np.zeros(n_router_labels)
            mae_padded = np.zeros(n_router_labels)
            mfe_padded = np.zeros(n_router_labels)
            probs_padded[:n_model_labels] = probs
            mae_padded[:n_model_labels] = mae_pred
            mfe_padded[:n_model_labels] = mfe_pred
        else:
            probs_padded = probs[:n_router_labels]
            mae_padded = mae_pred[:n_router_labels]
            mfe_padded = mfe_pred[:n_router_labels]

        signals = router.route(
            coin=coin,
            category=category,
            label_probs=probs_padded,
            mae_pred=mae_padded,
            mfe_pred=mfe_padded,
            regime_state=regime_state,
            confidence=float(conf),
        )

        if not signals:
            continue

        best = signals[0]  # highest score

        # Record EV for filter then check
        signal_filter.ev_history.setdefault(coin, __import__('collections').deque(maxlen=500))
        signal_filter.ev_history[coin].append(best.ev)

        # EV filter — only trade top 20% EV
        if not signal_filter.should_trade(coin, best.ev):
            continue

        # Calculate actual return
        hold_bars = min(best.hold_bars, len(close_test) - i - 1)
        if hold_bars <= 0:
            continue

        entry_price = close_test[i]
        exit_price = close_test[i + hold_bars]
        direction = 1 if best.direction == "long" else -1
        ret = direction * (exit_price - entry_price) / entry_price - 0.0008

        trades.append({
            "timestamp": test_idx[i],
            "direction": direction,
            "probability": float(best.probability),
            "ev": float(best.ev),
            "return": float(ret),
            "strategy": best.strategy_type.value,
            "strategy_name": best.strategy_name,
            "regime": best.regime.value,
            "hold_bars": hold_bars,
        })

    return trades


def analyze_trades(trades: list[dict], label: str) -> dict:
    """Analyze trade results."""
    if not trades:
        return {"label": label, "n_trades": 0}

    rets = [t["return"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    wr = len(wins) / len(rets) * 100
    avg_ret = np.mean(rets) * 100
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0
    total_ret = np.sum(rets) * 100
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252 * 24 / 4) if np.std(rets) > 0 else 0

    # Max drawdown
    equity = np.cumsum(rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak)
    max_dd = np.min(dd) * 100

    return {
        "label": label,
        "n_trades": len(rets),
        "wr": wr,
        "avg_ret_pct": avg_ret,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_ret_pct": total_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
    }


def run_comparison():
    """Compare single-strategy vs multi-strategy on all available coins."""
    print("=" * 70)
    print("Multi-Strategy vs Single-Strategy Backtest Comparison")
    print("=" * 70)

    # Load config to get available coins
    import json
    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    coins = config["coins"]
    # Also try BCH if available
    all_coins = coins + ["BCHUSDT"]

    results = {"single": [], "multi": []}

    for coin in all_coins:
        print(f"\n{'─'*60}")
        print(f"Coin: {coin}")
        print(f"{'─'*60}")

        # Load model
        model, checkpoint = load_model(coin)
        if model is None:
            print(f"  No model for {coin}")
            continue

        # Load data
        kline, features = load_data(coin)
        if features is None:
            print(f"  No data for {coin}")
            continue

        print(f"  Data: {len(kline)} bars, {features.shape[1]} features")

        # Run both backtests
        print(f"  Running single-strategy backtest...")
        trades_single = backtest_single_strategy(model, features, kline, checkpoint, coin)

        print(f"  Running multi-strategy backtest...")
        trades_multi = backtest_multi_strategy(model, features, kline, checkpoint, coin)

        # Analyze
        stats_single = analyze_trades(trades_single or [], f"{coin} Single")
        stats_multi = analyze_trades(trades_multi or [], f"{coin} Multi")

        results["single"].append(stats_single)
        results["multi"].append(stats_multi)

        # Print comparison
        for stats in [stats_single, stats_multi]:
            if stats["n_trades"] == 0:
                print(f"  {stats['label']:20s}: No trades")
            else:
                print(f"  {stats['label']:20s}: "
                      f"trades={stats['n_trades']:4d}  "
                      f"WR={stats['wr']:.1f}%  "
                      f"avg={stats['avg_ret_pct']:.3f}%  "
                      f"total={stats['total_ret_pct']:.1f}%  "
                      f"DD={stats['max_dd_pct']:.1f}%  "
                      f"sharpe={stats['sharpe']:.2f}")

        # Strategy type distribution for multi
        if trades_multi:
            strat_counts = defaultdict(int)
            regime_counts = defaultdict(int)
            for t in trades_multi:
                strat_counts[t["strategy"]] += 1
                regime_counts[t["regime"]] += 1
            print(f"  Multi strategies: {dict(strat_counts)}")
            print(f"  Multi regimes: {dict(regime_counts)}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for approach in ["single", "multi"]:
        valid = [r for r in results[approach] if r["n_trades"] > 0]
        if not valid:
            print(f"\n  {approach.upper()}: No trades")
            continue

        total_trades = sum(r["n_trades"] for r in valid)
        avg_wr = np.mean([r["wr"] for r in valid])
        avg_ret = np.mean([r["avg_ret_pct"] for r in valid])
        avg_sharpe = np.mean([r["sharpe"] for r in valid])

        print(f"\n  {approach.upper()} ({len(valid)} coins):")
        print(f"    Total trades: {total_trades}")
        print(f"    Avg WR: {avg_wr:.1f}%")
        print(f"    Avg per-trade return: {avg_ret:.3f}%")
        print(f"    Avg Sharpe: {avg_sharpe:.2f}")


if __name__ == "__main__":
    run_comparison()
