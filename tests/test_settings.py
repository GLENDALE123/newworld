from config.settings import Settings


def test_default_settings():
    s = Settings()
    assert s.symbol == "BTCUSDT"
    assert s.timeframe == "1h"
    assert s.tbm_pt_multiplier == 1.0
    assert s.tbm_sl_multiplier == 1.0
    assert s.tbm_max_holding_bars == 4
    assert s.volatility_span == 24
    assert s.position_size_pct == 0.02
    assert s.initial_capital == 100_000.0
    assert s.data_dir == "data/merged"


def test_settings_override():
    s = Settings(symbol="ETHUSDT", tbm_pt_multiplier=1.5)
    assert s.symbol == "ETHUSDT"
    assert s.tbm_pt_multiplier == 1.5
