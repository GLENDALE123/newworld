import pandas as pd
import numpy as np
from models.catboost_model import TradingModel


def _sample_features(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "sma_20": np.random.randn(n),
        "sma_50": np.random.randn(n),
        "ema_12": np.random.randn(n),
        "ema_26": np.random.randn(n),
        "macd": np.random.randn(n),
        "macd_signal": np.random.randn(n),
        "macd_hist": np.random.randn(n),
        "rsi_14": np.random.rand(n) * 100,
        "stoch_k": np.random.rand(n) * 100,
        "stoch_d": np.random.rand(n) * 100,
        "bb_upper": np.random.randn(n),
        "bb_mid": np.random.randn(n),
        "bb_lower": np.random.randn(n),
        "atr_14": np.abs(np.random.randn(n)),
        "obv": np.cumsum(np.random.randn(n)),
        "vol_sma_ratio": np.random.rand(n) + 0.5,
        "label": np.random.choice([1.0, -1.0], size=n),
    }
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(data, index=idx)


def test_train_and_predict():
    df = _sample_features()
    train = df.iloc[:400]
    test = df.iloc[400:]

    model = TradingModel(iterations=50, depth=4, learning_rate=0.1)
    model.train(train.drop(columns=["label"]), train["label"])

    preds = model.predict(test.drop(columns=["label"]))
    assert len(preds) == len(test)
    assert set(preds).issubset({1.0, -1.0})


def test_predict_proba():
    df = _sample_features()
    train = df.iloc[:400]
    test = df.iloc[400:]

    model = TradingModel(iterations=50, depth=4, learning_rate=0.1)
    model.train(train.drop(columns=["label"]), train["label"])

    probas = model.predict_proba(test.drop(columns=["label"]))
    assert len(probas) == len(test)
    assert all(0 <= p <= 1 for p in probas)


def test_walk_forward():
    df = _sample_features(n=800)

    model = TradingModel(iterations=50, depth=4, learning_rate=0.1)
    results = model.walk_forward(df, train_window=400, val_window=100)

    assert "predictions" in results
    assert "actuals" in results
    assert "indices" in results
    assert len(results["predictions"]) == len(results["actuals"])
    assert len(results["predictions"]) > 0


def test_untrained_predict_raises():
    model = TradingModel()
    df = _sample_features(n=10)
    try:
        model.predict(df.drop(columns=["label"]))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
