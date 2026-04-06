import pandas as pd
import numpy as np
from labeling.tbm import TripleBarrierLabeler


def _make_ohlcv(closes: list[float], base_ts: str = "2021-01-01") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(base_ts, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 50 for c in closes],
            "low": [c - 50 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def test_compute_volatility():
    labeler = TripleBarrierLabeler()
    closes = pd.Series([100.0] * 30 + [110.0])
    vol = labeler.compute_volatility(closes)
    assert len(vol) == len(closes)


def test_label_uptrend():
    closes = [100.0] * 30 + [100, 102, 104, 106, 108, 110, 112, 114]
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler(pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4)
    labels = labeler.label(df)
    assert 1 in labels.values


def test_label_downtrend():
    closes = [100.0] * 30 + [100, 98, 96, 94, 92, 90, 88, 86]
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler(pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4)
    labels = labeler.label(df)
    assert -1 in labels.values


def test_label_length_matches():
    closes = [100.0 + i * 0.1 for i in range(50)]
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler()
    labels = labeler.label(df)
    assert len(labels) == len(df)


def test_vertical_barrier():
    closes = [100.0] * 50
    df = _make_ohlcv(closes)
    labeler = TripleBarrierLabeler(pt_multiplier=5.0, sl_multiplier=5.0, max_holding_bars=4)
    labels = labeler.label(df)
    assert -1 in labels.values
