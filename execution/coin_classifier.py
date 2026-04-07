"""
Coin Classifier — Categorize 244 coins by trading characteristics

Categories:
  MAJOR:     BTC, ETH, SOL — high liquidity, multi-strategy (not just trend)
  LARGE_ALT: BNB, ADA, XRP, LINK, DOT... — medium liquidity, selective
  SMALL_ALT: 200+ coins — low liquidity, breakout + regime-conditional

Each category gets:
  - Multi-dimensional strategy permissions (per regime)
  - Regime-specific entry rules
  - Leverage limits (category + volatility adjusted)
  - Position sizing and hold periods

"추세추종만하는게 맞아???" → NO. Multi-dimensional per category.
"""

import os
import pandas as pd
import numpy as np


MAJOR = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

LARGE_ALT = {
    "BNBUSDT", "ADAUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT",
    "AVAXUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "NEARUSDT",
    "UNIUSDT", "AAVEUSDT", "INJUSDT", "OPUSDT", "ARBUSDT",
    "APTUSDT", "SUIUSDT", "FILUSDT", "ATOMUSDT", "ICPUSDT",
    "MATICUSDT", "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT",
}

# Coins with proven per-trade alpha (from iter 80-102 analysis)
ALPHA_COINS = {
    "CRVUSDT", "UNIUSDT", "CHRUSDT", "DUSKUSDT", "BCHUSDT",
    "XRPUSDT", "AVAXUSDT", "DOGEUSDT", "ONTUSDT",
}


class CoinClassifier:
    """Classify coins and determine multi-dimensional strategy per category."""

    def __init__(self, data_dir: str = "data/merged"):
        self.data_dir = data_dir

    def classify(self, symbol: str) -> str:
        if symbol in MAJOR:
            return "major"
        if symbol in LARGE_ALT:
            return "large_alt"
        return "small_alt"

    def has_proven_alpha(self, symbol: str) -> bool:
        return symbol in ALPHA_COINS

    def get_strategy(self, symbol: str) -> dict:
        """Get multi-dimensional strategy config for a coin.

        Returns regime-aware strategy rules, not a single fixed mode.
        """
        cat = self.classify(symbol)
        is_alpha = self.has_proven_alpha(symbol)

        if cat == "major":
            return {
                "category": "major",
                "mode": "multi_strategy",  # NOT just trend_following
                "strategies": {
                    # Regime → allowed trade approaches
                    "surge": ["trend_follow", "momentum"],
                    "dump": ["trend_follow", "momentum"],
                    "range": ["mean_reversion", "scalp"],
                    "volatile": ["breakout", "momentum_scalp"],
                },
                "min_probability": {
                    "surge": 0.55, "dump": 0.55,
                    "range": 0.60, "volatile": 0.60,
                },
                "max_leverage": 10,
                "leverage_by_regime": {
                    "surge": 1.0, "dump": 1.0,
                    "range": 0.7, "volatile": 0.5,
                },
                "base_size_pct": 0.10,
                "check_interval": 4,  # every 1h
                "min_ev_percentile": 0.30,  # top 70% (liquid, more trades OK)
            }
        elif cat == "large_alt":
            return {
                "category": "large_alt",
                "mode": "selective_multi",
                "strategies": {
                    "surge": ["trend_follow", "momentum"],
                    "dump": ["trend_follow"],
                    "range": ["mean_reversion", "scalp"],
                    "volatile": ["breakout"],
                },
                "min_probability": {
                    "surge": 0.55, "dump": 0.58,
                    "range": 0.60, "volatile": 0.65,
                },
                "max_leverage": 15,
                "leverage_by_regime": {
                    "surge": 1.0, "dump": 0.8,
                    "range": 0.6, "volatile": 0.4,
                },
                "base_size_pct": 0.15 if is_alpha else 0.10,
                "check_interval": 4,
                "min_ev_percentile": 0.20,  # top 20%
            }
        else:  # small_alt
            return {
                "category": "small_alt",
                "mode": "breakout_conditional",
                "strategies": {
                    "surge": ["breakout", "momentum"],
                    "dump": ["breakout"],
                    "range": [],  # no trading in range for small alts
                    "volatile": ["breakout"],
                },
                "min_probability": {
                    "surge": 0.60, "dump": 0.65,
                    "range": 1.0,  # disabled
                    "volatile": 0.65,
                },
                "max_leverage": 20,
                "leverage_by_regime": {
                    "surge": 1.0, "dump": 0.7,
                    "range": 0.0, "volatile": 0.5,
                },
                "base_size_pct": 0.20 if is_alpha else 0.12,
                "check_interval": 2,  # every 30min (catch breakouts)
                "min_ev_percentile": 0.20,
                "breakout_required": True,
            }

    def scan_all(self) -> dict[str, list[str]]:
        """Scan data dir and classify all available coins."""
        result = {"major": [], "large_alt": [], "small_alt": []}
        for d in sorted(os.listdir(self.data_dir)):
            if not d.endswith("USDT"):
                continue
            p = os.path.join(self.data_dir, d, "kline_15m.parquet")
            if not os.path.exists(p):
                continue
            cat = self.classify(d)
            result[cat].append(d)
        return result

    def scan_summary(self) -> str:
        """Human-readable summary of coin distribution."""
        coins = self.scan_all()
        alpha_count = sum(
            1 for cat_coins in coins.values()
            for c in cat_coins if c in ALPHA_COINS
        )
        lines = [
            f"Total: {sum(len(v) for v in coins.values())} coins",
            f"  Major: {len(coins['major'])} ({', '.join(coins['major'])})",
            f"  Large alt: {len(coins['large_alt'])}",
            f"  Small alt: {len(coins['small_alt'])}",
            f"  Proven alpha: {alpha_count}",
        ]
        return "\n".join(lines)
