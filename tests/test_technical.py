import pandas as pd
import numpy as np
from features.technical import compute_technical_features


def _sample_ohlcv(n: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    close = 30000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close - np.random.rand(n) * 50,
            "high": close + np.abs(np.random.randn(n)) * 100,
            "low": close - np.abs(np.random.randn(n)) * 100,
            "close": close,
            "volume": np.random.rand(n) * 1000 + 500,
        },
        index=pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_compute_features_columns():
    df = _sample_ohlcv()
    features = compute_technical_features(df)
    expected_cols = [
        "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "stoch_k", "stoch_d",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14", "obv", "vol_sma_ratio",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_compute_features_length():
    df = _sample_ohlcv()
    features = compute_technical_features(df)
    assert len(features) == len(df)


def test_compute_features_no_inf():
    df = _sample_ohlcv()
    features = compute_technical_features(df)
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values[~np.isnan(numeric.values)]).any()
