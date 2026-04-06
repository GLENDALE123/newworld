"""
Position Manager — Multi-Slot Portfolio Management

Manages N concurrent positions across potentially multiple assets.
This is the infrastructure layer between the PLE model (predictions)
and the execution layer (NautilusTrader orders).

Design:
  PLE model outputs: {strategy: probability, MFE, MAE} for each asset
  Position Manager decides:
    - Which slots to open/close
    - How much capital per slot (sizing)
    - Leverage per slot
    - When to rebalance

This module is RL-ready: the state/action/reward interface is designed
to be wrapped by an RL agent later.

State:  (equity, drawdown, n_open, per_slot_pnl, per_slot_hold_time, market_regime)
Action: (slot_id, asset, direction, size_pct, leverage)
Reward: equity_change / max_drawdown_penalty
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    """Single open position."""
    slot_id: int
    asset: str
    direction: int  # 1=long, -1=short
    entry_price: float
    size_pct: float  # % of equity
    leverage: float
    entry_time: int  # bar index
    strategy: str
    hold_limit: int  # max bars to hold
    # Tracking
    unrealized_pnl: float = 0.0
    max_pnl: float = 0.0
    bars_held: int = 0


@dataclass
class PortfolioState:
    """Current portfolio state (RL observation)."""
    equity: float
    peak_equity: float
    drawdown: float
    n_open: int
    total_exposure: float  # sum of all position sizes
    position_pnls: list[float] = field(default_factory=list)
    position_hold_times: list[int] = field(default_factory=list)


class PositionManager:
    """Multi-slot position manager.

    Args:
        max_slots: Maximum concurrent positions
        initial_equity: Starting capital
        max_total_exposure: Maximum total exposure (sum of all sizes)
        max_drawdown: Hard stop drawdown limit
        fee_pct: Trading fee per trade
    """

    def __init__(
        self,
        max_slots: int = 3,
        initial_equity: float = 100000.0,
        max_total_exposure: float = 1.0,  # 100% of equity
        max_drawdown: float = 0.15,
        fee_pct: float = 0.0008,
    ):
        self.max_slots = max_slots
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.max_total_exposure = max_total_exposure
        self.max_drawdown = max_drawdown
        self.fee_pct = fee_pct

        self.positions: dict[int, Position] = {}
        self.next_slot_id = 0
        self.trade_history: list[dict] = []
        self.equity_curve: list[float] = [initial_equity]

    @property
    def state(self) -> PortfolioState:
        """Current portfolio state for RL observation."""
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        exposure = sum(p.size_pct * p.leverage for p in self.positions.values())
        return PortfolioState(
            equity=self.equity,
            peak_equity=self.peak_equity,
            drawdown=dd,
            n_open=len(self.positions),
            total_exposure=exposure,
            position_pnls=[p.unrealized_pnl for p in self.positions.values()],
            position_hold_times=[p.bars_held for p in self.positions.values()],
        )

    def can_open(self, size_pct: float, leverage: float = 1.0) -> bool:
        """Check if a new position can be opened."""
        if len(self.positions) >= self.max_slots:
            return False
        current_exposure = sum(p.size_pct * p.leverage for p in self.positions.values())
        if current_exposure + size_pct * leverage > self.max_total_exposure:
            return False
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd >= self.max_drawdown:
            return False
        return True

    def open_position(
        self,
        asset: str,
        direction: int,
        entry_price: float,
        size_pct: float,
        leverage: float = 1.0,
        strategy: str = "",
        hold_limit: int = 168,
        bar_idx: int = 0,
    ) -> Optional[int]:
        """Open a new position. Returns slot_id or None if rejected."""
        if not self.can_open(size_pct, leverage):
            return None

        slot_id = self.next_slot_id
        self.next_slot_id += 1

        self.positions[slot_id] = Position(
            slot_id=slot_id,
            asset=asset,
            direction=direction,
            entry_price=entry_price,
            size_pct=size_pct,
            leverage=leverage,
            entry_time=bar_idx,
            strategy=strategy,
            hold_limit=hold_limit,
        )

        # Deduct fee
        self.equity -= self.equity * size_pct * self.fee_pct

        return slot_id

    def update_positions(self, prices: dict[str, float], bar_idx: int = 0):
        """Update all positions with current prices. Close expired ones."""
        to_close = []

        for slot_id, pos in self.positions.items():
            if pos.asset not in prices:
                continue

            current_price = prices[pos.asset]
            pnl_pct = pos.direction * (current_price - pos.entry_price) / pos.entry_price
            pos.unrealized_pnl = pnl_pct * pos.size_pct * pos.leverage
            pos.max_pnl = max(pos.max_pnl, pos.unrealized_pnl)
            pos.bars_held += 1

            # Check hold limit
            if pos.bars_held >= pos.hold_limit:
                to_close.append((slot_id, current_price, "timeout"))

        for slot_id, price, reason in to_close:
            self.close_position(slot_id, price, reason, bar_idx)

    def close_position(
        self, slot_id: int, exit_price: float, reason: str = "manual", bar_idx: int = 0
    ) -> Optional[dict]:
        """Close a position. Returns trade record."""
        if slot_id not in self.positions:
            return None

        pos = self.positions.pop(slot_id)
        pnl_pct = pos.direction * (exit_price - pos.entry_price) / pos.entry_price
        net_pnl = pnl_pct * pos.leverage - self.fee_pct
        equity_change = net_pnl * self.equity * pos.size_pct

        self.equity += equity_change
        self.peak_equity = max(self.peak_equity, self.equity)

        trade = {
            "slot_id": pos.slot_id,
            "asset": pos.asset,
            "direction": pos.direction,
            "strategy": pos.strategy,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(net_pnl * 100, 3),
            "equity_after": round(self.equity, 2),
            "hold_bars": pos.bars_held,
            "leverage": pos.leverage,
            "reason": reason,
        }
        self.trade_history.append(trade)
        self.equity_curve.append(self.equity)
        return trade

    def get_best_signals(
        self,
        signals: list[dict],
        top_n: int = None,
    ) -> list[dict]:
        """Rank signals by EV and return top candidates.

        Each signal: {asset, direction, probability, mfe, mae, strategy, hold_limit}
        """
        if top_n is None:
            top_n = self.max_slots - len(self.positions)

        # Filter out assets we already have positions in
        open_assets = {p.asset for p in self.positions.values()}

        candidates = []
        for sig in signals:
            if sig["asset"] in open_assets:
                continue
            ev = sig["probability"] * sig["mfe"] - (1 - sig["probability"]) * sig["mae"] - self.fee_pct
            if ev > 0:
                candidates.append({**sig, "ev": ev})

        # Sort by EV descending
        candidates.sort(key=lambda x: x["ev"], reverse=True)
        return candidates[:top_n]

    def summary(self) -> dict:
        """Portfolio performance summary."""
        if not self.trade_history:
            return {"trades": 0, "return_pct": 0.0}

        trades = self.trade_history
        pnls = [t["pnl_pct"] for t in trades]
        ret = (self.equity - 100000) / 100000 * 100
        max_dd = 0
        peak = 100000
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        return {
            "trades": len(trades),
            "return_pct": round(ret, 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            "avg_pnl": round(np.mean(pnls), 3),
            "max_drawdown": round(max_dd * 100, 1),
            "final_equity": round(self.equity, 2),
        }
