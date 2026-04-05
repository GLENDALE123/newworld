import pandas as pd
import pandas_ta as ta


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()

    # Trend
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd.iloc[:, 0]
    df["macd_signal"] = macd.iloc[:, 2]
    df["macd_hist"] = macd.iloc[:, 1]

    # Momentum
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch.iloc[:, 0]
    df["stoch_d"] = stoch.iloc[:, 1]

    # Volatility
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bbands.iloc[:, 0]
    df["bb_mid"] = bbands.iloc[:, 1]
    df["bb_upper"] = bbands.iloc[:, 2]

    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume
    df["obv"] = ta.obv(df["close"], df["volume"])

    vol_sma = ta.sma(df["volume"], length=20)
    df["vol_sma_ratio"] = df["volume"] / vol_sma

    feature_cols = [
        "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "stoch_k", "stoch_d",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14", "obv", "vol_sma_ratio",
    ]
    return df[feature_cols]
