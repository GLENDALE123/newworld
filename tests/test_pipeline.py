import pandas as pd
import numpy as np
from features.pipeline import FeaturePipeline
from config.settings import Settings


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


def test_pipeline_produces_features_and_labels():
    settings = Settings()
    pipeline = FeaturePipeline(settings)
    df = _sample_ohlcv()
    result = pipeline.build(df)
    assert "label" in result.columns
    assert "rsi_14" in result.columns
    assert "macd" in result.columns


def test_pipeline_drops_nan_rows():
    settings = Settings()
    pipeline = FeaturePipeline(settings)
    df = _sample_ohlcv()
    result = pipeline.build(df)
    assert not result.isnull().any().any()
    assert len(result) > 0
    assert len(result) < len(df)


def test_pipeline_label_values():
    settings = Settings()
    pipeline = FeaturePipeline(settings)
    df = _sample_ohlcv()
    result = pipeline.build(df)
    assert set(result["label"].unique()).issubset({1.0, -1.0})
