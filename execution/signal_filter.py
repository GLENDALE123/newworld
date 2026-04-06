"""
Signal Filter — Production-ready selective trading

Maintains rolling EV percentile per coin for real-time filtering.
Only passes high-conviction signals (configurable percentile).

"확신 높은 것만 매매. 나머지는 관망."
"""

import numpy as np
from collections import deque


class SignalFilter:
    """Per-coin rolling EV filter for selective trading.

    Maintains a window of recent EV values per coin.
    Only passes signals in the top percentile.
    """

    def __init__(self, top_pct: float = 0.20, window: int = 500):
        self.top_pct = top_pct
        self.window = window
        self.ev_history: dict[str, deque] = {}

    def should_trade(self, coin: str, ev: float) -> bool:
        """Check if this signal's EV is in the top percentile for this coin."""
        if coin not in self.ev_history:
            self.ev_history[coin] = deque(maxlen=self.window)

        history = self.ev_history[coin]
        history.append(ev)

        # Need minimum history before filtering
        if len(history) < 50:
            return ev > 0  # trade all positive EV until we have enough history

        threshold = np.percentile(list(history), (1 - self.top_pct) * 100)
        return ev >= threshold

    def stats(self, coin: str) -> dict:
        if coin not in self.ev_history:
            return {}
        h = list(self.ev_history[coin])
        return {
            "n": len(h),
            "threshold": np.percentile(h, (1 - self.top_pct) * 100) if len(h) > 10 else 0,
            "avg_ev": np.mean(h),
            "max_ev": max(h),
        }
