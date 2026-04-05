from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.model import (
    BarType,
    Money,
    TraderId,
    Venue,
)
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from config.settings import Settings
from strategy.ml_strategy import MLStrategy, MLStrategyConfig


def run_backtest(
    settings: Settings,
    ohlcv_df: pd.DataFrame,
    signals: dict,
) -> BacktestEngine:
    """
    Run a NautilusTrader backtest.

    Parameters
    ----------
    settings : Settings
    ohlcv_df : pd.DataFrame - OHLCV with datetime index
    signals : dict - {timestamp_ns: signal} pre-computed predictions
    """
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
    )
    engine = BacktestEngine(config=config)

    # Venue
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(settings.initial_capital, USDT)],
    )

    # Instrument
    instrument = TestInstrumentProvider.btcusdt_binance()
    engine.add_instrument(instrument)

    # Data - ensure writable contiguous arrays (parquet buffers can be read-only)
    bar_type = BarType.from_str(f"{instrument.id}-1-HOUR-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    ohlcv_w = ohlcv_df.copy()
    for col in ohlcv_w.columns:
        ohlcv_w[col] = np.ascontiguousarray(ohlcv_w[col].values, dtype=np.float64)
    bars = wrangler.process(ohlcv_w)
    engine.add_data(bars)

    # Strategy
    strategy_config = MLStrategyConfig(
        instrument_id=str(instrument.id),
        bar_type=str(bar_type),
        pt_multiplier=settings.tbm_pt_multiplier,
        sl_multiplier=settings.tbm_sl_multiplier,
        max_holding_bars=settings.tbm_max_holding_bars,
        volatility_span=settings.volatility_span,
        position_size_pct=settings.position_size_pct,
    )
    strategy = MLStrategy(config=strategy_config)
    strategy.set_signals(signals)
    engine.add_strategy(strategy)

    # Run
    engine.run()
    return engine
