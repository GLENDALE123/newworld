import os

import pandas as pd


def load_kline(data_dir: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV kline data for a symbol/timeframe from merged parquet files.

    Returns DataFrame with timestamp index and columns: open, high, low, close, volume
    """
    path = os.path.join(data_dir, symbol, f"kline_{timeframe}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}")

    df = pd.read_parquet(path)
    df = df.set_index("timestamp").sort_index()
    # Keep only OHLCV columns
    return df[["open", "high", "low", "close", "volume"]]


def list_symbols(data_dir: str) -> list[str]:
    """List all available symbols in the data directory."""
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
