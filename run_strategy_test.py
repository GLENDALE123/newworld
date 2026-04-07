"""
Strategy Router Integration Test — Validate multi-dimensional pipeline

Pipeline: Data → Features → Model inference (simulated) → Regime Detection
→ Strategy Router → Trade Signals

Tests the new execution layer with all 244 coins:
  1. Load kline data for each coin
  2. Detect regime per coin
  3. Simulate model output (random or from saved models)
  4. Route through StrategyRouter
  5. Select best portfolio-level trades
  6. Print statistics

"전체 파이프라인이 제대로 연결되는지 확인."
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

from execution.coin_classifier import CoinClassifier, ALPHA_COINS
from execution.regime_detector import RegimeDetector, Regime
from execution.strategy_router import (
    StrategyRouter, LABEL_INDEX, N_LABELS, TradeType,
)
from execution.signal_filter import SignalFilter
from execution.capital_strategy import CapitalStrategy


def load_kline(data_dir: str, symbol: str) -> pd.DataFrame | None:
    """Load 15m kline data for a symbol."""
    path = os.path.join(data_dir, symbol, "kline_15m.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df = df.sort_index()
        # Need at least 200 bars for regime detection
        if len(df) < 200:
            return None
        return df
    except Exception:
        return None


def simulate_model_output(n_labels: int = 32, quality: str = "random") -> dict:
    """Simulate PLE model output for testing.

    quality:
      "random" — uniform random probabilities
      "realistic" — centered around 0.45 with some high-conviction signals
      "alpha" — slightly shifted positive (simulates coin with alpha)
    """
    if quality == "random":
        probs = np.random.uniform(0.3, 0.7, n_labels)
    elif quality == "realistic":
        probs = np.random.normal(0.45, 0.08, n_labels).clip(0.2, 0.8)
    elif quality == "alpha":
        probs = np.random.normal(0.50, 0.10, n_labels).clip(0.2, 0.85)
        # Boost a few strategies to simulate real alpha
        top_idx = np.random.choice(n_labels, 3, replace=False)
        probs[top_idx] = np.random.uniform(0.60, 0.75, 3)
    else:
        probs = np.full(n_labels, 0.5)

    mae = np.random.uniform(-0.03, 0, n_labels)   # negative = loss
    mfe = np.random.uniform(0, 0.05, n_labels)     # positive = gain

    return {
        "label_probs": probs,
        "mae_pred": mae,
        "mfe_pred": mfe,
        "confidence": float(np.random.uniform(0.3, 0.7)),
    }


def run_test(data_dir: str = "data/merged", n_coins: int = 0):
    """Run full pipeline test."""
    print("=" * 70)
    print("Strategy Router Integration Test")
    print("=" * 70)

    classifier = CoinClassifier(data_dir)
    regime_detector = RegimeDetector()
    router = StrategyRouter(min_ev=0.0005)
    signal_filter = SignalFilter(top_pct=0.20)
    capital = CapitalStrategy(initial_equity=500)

    # Scan all coins
    coins = classifier.scan_all()
    total = sum(len(v) for v in coins.values())
    print(f"\n{classifier.scan_summary()}")

    # Process each coin
    all_signals = []
    regime_counts = Counter()
    strategy_counts = Counter()
    category_signal_counts = {"major": 0, "large_alt": 0, "small_alt": 0}

    coin_list = []
    for cat_coins in coins.values():
        coin_list.extend(cat_coins)
    if n_coins > 0:
        coin_list = coin_list[:n_coins]

    print(f"\nProcessing {len(coin_list)} coins...")

    for i, symbol in enumerate(coin_list):
        df = load_kline(data_dir, symbol)
        if df is None:
            continue

        # 1. Detect regime
        regime_state = regime_detector.detect_current(
            symbol,
            df["high"].values[-200:],
            df["low"].values[-200:],
            df["close"].values[-200:],
            df["volume"].values[-200:],
        )
        regime_counts[regime_state.regime.value] += 1

        # 2. Classify coin
        category = classifier.classify(symbol)
        is_alpha = classifier.has_proven_alpha(symbol)

        # 3. Simulate model output
        quality = "alpha" if is_alpha else "realistic"
        model_out = simulate_model_output(N_LABELS, quality)

        # 4. Route through strategy router
        signals = router.route(
            coin=symbol,
            category=category,
            label_probs=model_out["label_probs"],
            mae_pred=model_out["mae_pred"],
            mfe_pred=model_out["mfe_pred"],
            regime_state=regime_state,
            confidence=model_out["confidence"],
        )

        if signals:
            category_signal_counts[category] += len(signals)
            for sig in signals:
                strategy_counts[sig.strategy_type.value] += 1
            all_signals.extend(signals)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(coin_list)}] signals so far: {len(all_signals)}")

    # 5. Select best portfolio-level trades
    max_slots = capital.max_slots
    best = router.select_best(all_signals, max_concurrent=max_slots)

    # Results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\nRegime Distribution:")
    for regime, count in sorted(regime_counts.items()):
        pct = count / sum(regime_counts.values()) * 100
        print(f"  {regime:12s}: {count:4d} ({pct:.1f}%)")

    print(f"\nSignals Generated: {len(all_signals)}")
    print(f"  By category:")
    for cat, count in sorted(category_signal_counts.items()):
        print(f"    {cat:12s}: {count:4d}")

    print(f"\n  By strategy type:")
    for stype, count in sorted(strategy_counts.items()):
        print(f"    {stype:20s}: {count:4d}")

    print(f"\nPortfolio Selection (max {max_slots} slots):")
    if best:
        for sig in best:
            alpha_tag = " [ALPHA]" if sig.coin in ALPHA_COINS else ""
            print(f"  {sig.coin:12s} {sig.direction:5s} {sig.strategy_type.value:18s} "
                  f"P={sig.probability:.2f} EV={sig.ev:.4f} "
                  f"regime={sig.regime.value} score={sig.score:.6f}{alpha_tag}")
    else:
        print("  No trades selected (all filtered)")

    print(f"\nCapital: {capital.summary()}")

    # Strategy distribution analysis
    print(f"\n{'=' * 70}")
    print("STRATEGY DIVERSITY ANALYSIS")
    print(f"{'=' * 70}")

    unique_types = set(s.strategy_type for s in all_signals)
    print(f"\nUnique strategy types used: {len(unique_types)}")
    for t in unique_types:
        print(f"  - {t.value}")

    unique_regimes = set(s.regime for s in all_signals)
    print(f"\nRegimes with active signals: {len(unique_regimes)}")
    for r in unique_regimes:
        print(f"  - {r.value}")

    # Direction balance
    long_count = sum(1 for s in all_signals if s.direction == "long")
    short_count = sum(1 for s in all_signals if s.direction == "short")
    print(f"\nDirection balance: {long_count} long / {short_count} short")

    # Top signals by EV
    if all_signals:
        print(f"\nTop 10 signals by score:")
        top10 = sorted(all_signals, key=lambda s: s.score, reverse=True)[:10]
        for sig in top10:
            alpha_tag = " *" if sig.coin in ALPHA_COINS else ""
            print(f"  {sig.coin:12s} {sig.direction:5s} {sig.strategy_name:30s} "
                  f"P={sig.probability:.2f} EV={sig.ev:.4f} "
                  f"score={sig.score:.6f}{alpha_tag}")

    return all_signals


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_test(n_coins=n)
