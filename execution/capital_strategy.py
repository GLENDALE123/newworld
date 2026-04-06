"""
Capital-Adaptive Strategy — Aggressive to Conservative

Stage 1: $500 or less → 20x leverage, high risk, fast growth
Stage 2: $5K          → 10x leverage, moderate risk
Stage 3: $50K         → 3x leverage, conservative
Stage 4: $100K+       → 1-2x leverage, wealth preservation

The Position Manager adjusts automatically based on current equity.
"""

import numpy as np


class CapitalStrategy:
    """Determines leverage and sizing based on current capital."""

    STAGES = [
        {"name": "aggressive",    "max_equity": 500,    "leverage": 20, "size_pct": 0.50, "max_dd": 0.30},
        {"name": "growth",        "max_equity": 5000,   "leverage": 10, "size_pct": 0.20, "max_dd": 0.20},
        {"name": "moderate",      "max_equity": 50000,  "leverage": 5,  "size_pct": 0.10, "max_dd": 0.15},
        {"name": "conservative",  "max_equity": 100000, "leverage": 3,  "size_pct": 0.05, "max_dd": 0.12},
        {"name": "preservation",  "max_equity": float("inf"), "leverage": 1, "size_pct": 0.03, "max_dd": 0.10},
    ]

    def __init__(self, initial_equity: float = 500):
        self.equity = initial_equity
        self.peak = initial_equity
        self.stage_history = []

    @property
    def stage(self) -> dict:
        for s in self.STAGES:
            if self.equity <= s["max_equity"]:
                return s
        return self.STAGES[-1]

    @property
    def leverage(self) -> float:
        return self.stage["leverage"]

    @property
    def size_pct(self) -> float:
        # DD-adjusted sizing within stage
        dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0
        max_dd = self.stage["max_dd"]
        base = self.stage["size_pct"]
        return base * max(0.2, 1 - dd / max_dd)

    @property
    def max_slots(self) -> int:
        if self.equity <= 500: return 1      # focus capital
        if self.equity <= 5000: return 2
        if self.equity <= 50000: return 3
        return 5

    def update(self, equity: float):
        old_stage = self.stage["name"]
        self.equity = equity
        self.peak = max(self.peak, equity)
        new_stage = self.stage["name"]
        if old_stage != new_stage:
            self.stage_history.append({
                "from": old_stage, "to": new_stage,
                "equity": equity,
            })
            return True  # stage changed
        return False

    def summary(self) -> str:
        s = self.stage
        dd = (self.peak - self.equity) / self.peak * 100 if self.peak > 0 else 0
        return (f"${self.equity:,.0f} [{s['name']}] "
                f"lever={s['leverage']}x size={self.size_pct*100:.1f}% "
                f"slots={self.max_slots} DD={dd:.1f}%")
