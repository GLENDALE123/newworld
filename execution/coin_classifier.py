"""
Coin Classifier — Categorize 220 coins by trading characteristics

Categories:
  MAJOR:     BTC, ETH, SOL — high liquidity, low vol, trend-following
  LARGE_ALT: BNB, ADA, XRP, LINK, DOT, AVAX... — medium liquidity
  SMALL_ALT: 200+ coins — low liquidity, breakout-only trading

Each category gets its own:
  - Entry conditions (always vs breakout-only)
  - Model type or threshold
  - Leverage limits
  - Position sizing
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
}


class CoinClassifier:
    """Classify coins and determine trading strategy per category."""

    def __init__(self, data_dir: str = "data/merged"):
        self.data_dir = data_dir

    def classify(self, symbol: str) -> str:
        if symbol in MAJOR:
            return "major"
        if symbol in LARGE_ALT:
            return "large_alt"
        return "small_alt"

    def get_strategy(self, symbol: str) -> dict:
        cat = self.classify(symbol)

        if cat == "major":
            return {
                "category": "major",
                "mode": "trend_following",
                "entry": "model_signal",  # trade on model signal
                "min_ev_percentile": 0.30,  # top 70% (more trades, liquid)
                "max_leverage": 10,
                "base_size_pct": 0.10,
                "check_interval": 4,  # every 1h
            }
        elif cat == "large_alt":
            return {
                "category": "large_alt",
                "mode": "selective",
                "entry": "model_signal + high_ev",
                "min_ev_percentile": 0.20,  # top 20%
                "max_leverage": 15,
                "base_size_pct": 0.15,
                "check_interval": 4,
            }
        else:  # small_alt
            return {
                "category": "small_alt",
                "mode": "breakout_only",
                "entry": "breakout_detected + model_signal",
                "min_ev_percentile": 0.20,  # top 20% of breakout signals
                "max_leverage": 20,
                "base_size_pct": 0.20,
                "check_interval": 2,  # every 30min (catch breakouts faster)
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
