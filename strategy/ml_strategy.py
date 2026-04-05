import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType, InstrumentId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy


class MLStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    pt_multiplier: float = 1.0
    sl_multiplier: float = 1.0
    max_holding_bars: int = 4
    volatility_span: int = 24
    position_size_pct: float = 0.02


class MLStrategy(Strategy):
    def __init__(self, config: MLStrategyConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.pt = config.pt_multiplier
        self.sl = config.sl_multiplier
        self.max_holding_bars = config.max_holding_bars
        self.vol_span = config.volatility_span
        self.position_size_pct = config.position_size_pct

        self.close_prices: list[float] = []
        self.bars_held: int = 0
        self.signals: dict = {}  # timestamp -> signal, pre-loaded before backtest

    def set_signals(self, signals: dict) -> None:
        """Pre-load signals as {timestamp_ns: signal_value} for backtest use."""
        self.signals = signals

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        close = float(bar.close)
        self.close_prices.append(close)

        has_position = self.portfolio.is_net_long(self.instrument_id) or self.portfolio.is_net_short(self.instrument_id)

        # Track holding period
        if has_position:
            self.bars_held += 1
            if self.bars_held >= self.max_holding_bars:
                self.close_all_positions(self.instrument_id)
                self.bars_held = 0
                return
        else:
            self.bars_held = 0

        # Need enough data for volatility
        if len(self.close_prices) < self.vol_span + 1:
            return

        # No new entry if already in position
        if has_position:
            return

        # Get signal from pre-loaded dict
        signal = self.signals.get(bar.ts_event)
        if signal is None:
            return

        sigma_t = self._compute_volatility()
        if sigma_t == 0 or np.isnan(sigma_t):
            return

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        account = self.portfolio.account(self.instrument_id.venue)
        if account is None:
            return

        equity = float(account.balance_total(instrument.quote_currency))
        notional = equity * self.position_size_pct
        quantity = instrument.make_qty(notional / close)

        if float(quantity) == 0:
            return

        if signal == 1.0:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=quantity,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.bars_held = 0

        elif signal == -1.0:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=quantity,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.bars_held = 0

    def _compute_volatility(self) -> float:
        prices = pd.Series(self.close_prices)
        returns = prices.pct_change()
        vol = returns.ewm(span=self.vol_span).std().iloc[-1]
        return float(vol) if not np.isnan(vol) else 0.0

    def on_stop(self) -> None:
        self.close_all_positions(self.instrument_id)
        self.bars_held = 0
