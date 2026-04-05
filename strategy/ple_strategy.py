"""
PLE v4 NautilusTrader Strategy

Real-time execution of the PLE multi-label trading system.
Connects trained model to live/paper trading via NautilusTrader.
"""

import numpy as np
import pandas as pd
import torch
from collections import deque

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType, InstrumentId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy


class PLEStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    model_path: str = ""
    sma_period: int = 50
    long_threshold: float = 0.40
    short_threshold: float = 0.55
    base_size_pct: float = 0.03
    max_dd_pct: float = 0.15
    fee_pct: float = 0.0008


class PLEStrategy(Strategy):
    """Execute PLE v4 model predictions via NautilusTrader.

    - Receives 15m bars
    - Computes features from price history
    - Runs model inference
    - Applies adaptive thresholds + SMA filter
    - Submits orders with DD-based sizing
    """

    def __init__(self, config: PLEStrategyConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.sma_period = config.sma_period
        self.long_thresh_base = config.long_threshold
        self.short_thresh_base = config.short_threshold
        self.base_size = config.base_size_pct
        self.max_dd = config.max_dd_pct
        self.fee_pct = config.fee_pct

        # Price history for feature computation
        self.close_history = deque(maxlen=500)
        self.high_history = deque(maxlen=500)
        self.low_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)

        # Tracking
        self.peak_equity = 0.0
        self.bars_in_position = 0
        self.model = None
        self.strat_info = []

    def set_model(self, model, strat_info: list[dict]):
        """Set the trained PLE model and strategy metadata."""
        self.model = model
        self.strat_info = strat_info

    def on_start(self):
        self.subscribe_bars(self.bar_type)
        self.log.info(f"PLEStrategy started for {self.instrument_id}")

    def on_bar(self, bar: Bar):
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        volume = float(bar.volume)

        self.close_history.append(close)
        self.high_history.append(high)
        self.low_history.append(low)
        self.volume_history.append(volume)

        # Need enough history
        if len(self.close_history) < 200:
            return

        if self.model is None:
            return

        # Track position holding time
        has_pos = (self.portfolio.is_net_long(self.instrument_id) or
                   self.portfolio.is_net_short(self.instrument_id))
        if has_pos:
            self.bars_in_position += 1
            # Max holding: 168 bars (7 days at 15m)
            if self.bars_in_position >= 168:
                self.close_all_positions(self.instrument_id)
                self.bars_in_position = 0
                return
        else:
            self.bars_in_position = 0

        # Don't enter new position if already in one
        if has_pos:
            return

        # Compute features (simplified — production should use full factory)
        features = self._compute_features()
        if features is None:
            return

        # Model inference
        signal = self._get_signal(features, close)
        if signal is None:
            return

        direction, size_pct, strategy_name = signal

        # Get instrument and compute quantity
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        account = self.portfolio.account(self.instrument_id.venue)
        if account is None:
            return

        equity = float(account.balance_total(instrument.quote_currency))
        self.peak_equity = max(self.peak_equity, equity)

        notional = equity * size_pct
        quantity = instrument.make_qty(notional / close)

        if float(quantity) == 0:
            return

        side = OrderSide.BUY if direction == 1 else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.bars_in_position = 0
        self.log.info(f"PLE signal: {strategy_name} {'LONG' if direction==1 else 'SHORT'} size={size_pct:.1%}")

    def _compute_features(self) -> np.ndarray | None:
        """Compute basic features from price history."""
        closes = np.array(self.close_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)
        volumes = np.array(self.volume_history)

        n = len(closes)
        if n < 200:
            return None

        features = []

        # Returns at various windows
        for w in [1, 4, 12, 48, 96]:
            if n > w:
                features.append((closes[-1] - closes[-1-w]) / closes[-1-w])
            else:
                features.append(0.0)

        # Volatility
        rets = np.diff(closes[-100:]) / closes[-100:-1]
        for w in [5, 10, 20, 50]:
            if len(rets) >= w:
                features.append(np.std(rets[-w:]))
            else:
                features.append(0.0)

        # ATR
        tr = np.maximum(highs[-20:] - lows[-20:],
                        np.abs(highs[-20:] - np.roll(closes[-20:], 1)[-20:]))
        features.append(np.mean(tr[-14:]) / closes[-1])

        # Volume
        if len(volumes) > 20:
            features.append(volumes[-1] / np.mean(volumes[-20:]))
        else:
            features.append(1.0)

        # Price position
        if n > 50:
            rmax = np.max(closes[-50:])
            rmin = np.min(closes[-50:])
            features.append((closes[-1] - rmin) / max(rmax - rmin, 1e-8))
        else:
            features.append(0.5)

        # Pad to match expected feature count (simplified)
        while len(features) < self.model.n_features if hasattr(self.model, 'n_features') else 200:
            features.append(0.0)

        return np.array(features[:200], dtype=np.float32)

    def _get_signal(self, features: np.ndarray, current_price: float):
        """Run model inference and return trading signal."""
        # SMA computation
        closes = np.array(self.close_history)
        sma = np.mean(closes[-self.sma_period:]) if len(closes) >= self.sma_period else current_price

        above_sma = current_price > sma
        long_thresh = self.long_thresh_base if above_sma else self.short_thresh_base
        short_thresh = self.short_thresh_base if above_sma else self.long_thresh_base

        # Model inference
        with torch.no_grad():
            x = torch.tensor(features).unsqueeze(0).to(next(self.model.parameters()).device)
            acc = torch.zeros(1, 4).to(x.device)
            out = self.model(x, acc)
            probs = out["label_probs"].cpu().numpy()[0]

        # Find best strategy
        best_ev = -1
        best_j = -1
        for j, info in enumerate(self.strat_info):
            thresh = long_thresh if info["dir"] == "long" else short_thresh
            if probs[j] < thresh:
                continue
            ev = probs[j] * 0.01 - (1 - probs[j]) * 0.01 - self.fee_pct
            if ev > best_ev:
                best_ev = ev
                best_j = j

        if best_j < 0 or best_ev <= 0:
            return None

        info = self.strat_info[best_j]
        direction = 1 if info["dir"] == "long" else -1

        # DD-based sizing
        account = self.portfolio.account(self.instrument_id.venue)
        if account:
            equity = float(account.balance_total(
                self.cache.instrument(self.instrument_id).quote_currency))
            dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
            size = self.base_size * max(0.2, 1 - dd / self.max_dd)
        else:
            size = self.base_size

        return direction, size, f"{info['style']}_{info['dir']}"

    def on_stop(self):
        self.close_all_positions(self.instrument_id)
        self.log.info("PLEStrategy stopped")
