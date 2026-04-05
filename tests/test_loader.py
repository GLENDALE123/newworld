import pandas as pd
import pytest

from data.loader import load_kline, list_symbols

DATA_DIR = "data/merged"


def test_load_kline_btcusdt_1h():
    df = load_kline(DATA_DIR, "BTCUSDT", "1h")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000  # should have lots of data
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert df.index.dtype == "datetime64[ns]"
    # Prices should be reasonable for BTC
    assert df["close"].iloc[0] > 10000


def test_load_kline_not_found():
    with pytest.raises(FileNotFoundError):
        load_kline(DATA_DIR, "NONEXISTENT", "1h")


def test_list_symbols():
    symbols = list_symbols(DATA_DIR)
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols
    assert "SOLUSDT" in symbols
    assert len(symbols) > 50
