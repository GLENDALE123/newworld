from strategy.ml_strategy import MLStrategyConfig


def test_config_defaults():
    config = MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
    )
    assert config.pt_multiplier == 1.0
    assert config.sl_multiplier == 1.0
    assert config.max_holding_bars == 4
    assert config.volatility_span == 24
    assert config.position_size_pct == 0.02


def test_config_override():
    config = MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
        pt_multiplier=1.5,
        max_holding_bars=6,
    )
    assert config.pt_multiplier == 1.5
    assert config.max_holding_bars == 6
