"""
ultraTM Scanner — 244-coin continuous monitoring with meta-model

Architecture:
  Every 15m bar:
    1. Load latest kline for all 244 coins
    2. Generate features (polars, fast)
    3. Run PLE base model → signal matrix (244 × 32)
    4. Detect regime per coin
    5. Meta-model evaluates ALL signals → trade decisions
    6. Execute selected trades

"하위모델이 시그널을 보내면, 메타모델이 거래할지 결정한다."
"""

import os
import sys
import time
import numpy as np
from collections import deque

from execution.coin_classifier import CoinClassifier
from execution.regime_detector import RegimeDetector, Regime
from execution.meta_model import MetaModel, SignalSnapshot, PortfolioState, TradeDecision
from execution.capital_strategy import CapitalStrategy


DATA_DIR = "data/merged"


class UltraTMScanner:
    """Continuous 244-coin scanner with meta-model trade decisions."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        initial_equity: float = 500,
    ):
        self.data_dir = data_dir
        self.classifier = CoinClassifier(data_dir)
        self.regime_detector = RegimeDetector()
        self.meta = MetaModel(
            persistence_bars=2,
            min_confidence=0.55,
            min_ev=0.001,
        )
        self.capital = CapitalStrategy(initial_equity)

        # Coin registry
        self.coins = self.classifier.scan_all()
        self.all_coins = []
        for cat_coins in self.coins.values():
            self.all_coins.extend(cat_coins)

        self.n_coins = len(self.all_coins)
        print(f"Scanner initialized: {self.n_coins} coins")
        print(f"  Major: {len(self.coins['major'])}")
        print(f"  Large alt: {len(self.coins['large_alt'])}")
        print(f"  Small alt: {len(self.coins['small_alt'])}")

    def scan_once(
        self,
        features_by_coin: dict[str, np.ndarray],
        probs_by_coin: dict[str, np.ndarray],
        confidence_by_coin: dict[str, float],
        kline_by_coin: dict[str, dict],  # {coin: {"high": arr, "low": arr, "close": arr, "volume": arr}}
        portfolio: PortfolioState,
    ) -> list[TradeDecision]:
        """Run one scan cycle across all coins.

        Args:
            features_by_coin: {coin: feature_array} from polars pipeline
            probs_by_coin: {coin: (N_strategies,) probs} from PLE model
            confidence_by_coin: {coin: float} model confidence
            kline_by_coin: {coin: {"high", "low", "close", "volume"}} recent bars
            portfolio: current portfolio state

        Returns:
            List of trade decisions from meta-model.
        """
        signals = []

        for coin in self.all_coins:
            if coin not in probs_by_coin:
                continue

            probs = probs_by_coin[coin]
            conf = confidence_by_coin.get(coin, 0.5)

            # Regime detection
            kd = kline_by_coin.get(coin)
            if kd is None:
                continue

            regime_state = self.regime_detector.detect_current(
                coin, kd["high"], kd["low"], kd["close"], kd["volume"],
            )

            # Best strategy
            best_idx = probs.argmax()
            best_prob = probs[best_idx]

            # EV calculation (simplified — assumes MAE/MFE from model)
            ev = best_prob * 0.02 - (1 - best_prob) * 0.01 - 0.0008
            direction = 1 if best_idx % 2 == 0 else -1  # even=long, odd=short

            # Volatility
            close = kd["close"]
            if len(close) > 20:
                rets = np.diff(close[-21:]) / close[-21:-1]
                vol = np.std(rets) if len(rets) > 0 else 0.02
            else:
                vol = 0.02

            category = self.classifier.classify(coin)

            snap = SignalSnapshot(
                coin=coin,
                coin_id=self.all_coins.index(coin),
                category=category,
                probs=probs,
                confidence=conf,
                regime=regime_state.regime.value if hasattr(regime_state.regime, 'value') else 2,
                ev_best=ev,
                direction_best=direction,
                strategy_idx=int(best_idx),
                volatility=vol,
            )
            # Convert regime string to int for snapshot
            regime_map = {"surge": 0, "dump": 1, "range": 2, "volatile": 3}
            snap.regime = regime_map.get(str(regime_state.regime.value), 2)

            signals.append(snap)

        # Meta-model decides
        decisions = self.meta.evaluate(signals, portfolio)
        return decisions

    def simulate(
        self,
        n_bars: int = 100,
        model_fn=None,  # function(coin, features) → (probs, confidence)
    ):
        """Simulate scanning with mock or real model."""
        print(f"\nSimulating {n_bars} bars...")

        portfolio = PortfolioState(
            equity=self.capital.equity,
            peak_equity=self.capital.peak,
            drawdown_pct=0,
            n_open_positions=0,
            max_positions=self.capital.max_slots,
        )

        total_decisions = 0
        decision_coins = set()

        for bar in range(n_bars):
            # Generate mock signals for all coins
            probs_by_coin = {}
            conf_by_coin = {}
            kline_by_coin = {}

            for coin in self.all_coins[:50]:  # test with 50 coins
                probs = np.random.normal(0.45, 0.08, 32).clip(0.2, 0.8).astype(np.float32)
                if coin in {"CRVUSDT", "UNIUSDT", "CHRUSDT", "DUSKUSDT"}:
                    # Alpha coins get slightly better signals
                    boost = np.random.choice(32, 3, replace=False)
                    probs[boost] = np.random.uniform(0.58, 0.72, 3)

                probs_by_coin[coin] = probs
                conf_by_coin[coin] = float(np.random.uniform(0.4, 0.7))
                kline_by_coin[coin] = {
                    "high": np.random.uniform(100, 110, 200),
                    "low": np.random.uniform(90, 100, 200),
                    "close": np.cumsum(np.random.normal(0, 0.5, 200)) + 100,
                    "volume": np.random.uniform(1000, 5000, 200),
                }

            decisions = self.scan_once(
                features_by_coin={},
                probs_by_coin=probs_by_coin,
                confidence_by_coin=conf_by_coin,
                kline_by_coin=kline_by_coin,
                portfolio=portfolio,
            )

            if decisions:
                total_decisions += len(decisions)
                for d in decisions:
                    decision_coins.add(d.coin)

                if bar % 20 == 0 or len(decisions) > 0:
                    for d in decisions:
                        print(f"  Bar {bar:3d}: {d.coin:12s} {'LONG' if d.direction==1 else 'SHORT':5s} "
                              f"size={d.size_pct:.1%} lev={d.leverage:.0f}x "
                              f"conf={d.confidence:.4f} [{d.reason}]")

        print(f"\n  Total decisions: {total_decisions} over {n_bars} bars")
        print(f"  Unique coins traded: {len(decision_coins)}")
        print(f"  Avg decisions/bar: {total_decisions/n_bars:.2f}")
        trade_rate = total_decisions / n_bars
        print(f"  Trade selectivity: {(1-trade_rate)*100:.0f}% of bars = no trade")


if __name__ == "__main__":
    scanner = UltraTMScanner(initial_equity=500)
    scanner.simulate(n_bars=200)
