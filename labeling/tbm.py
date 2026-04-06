import numpy as np
import pandas as pd
from numba import njit


@njit
def _tbm_label(closes, highs, lows, vol, pt_mult, sl_mult, max_hold):
    """Numba-compiled TBM labeling — same logic, ~100-200x faster."""
    n = len(closes)
    labels = np.empty(n)
    labels[:] = np.nan
    for i in range(n - 1):
        sigma = vol[i]
        if np.isnan(sigma):
            continue
        entry = closes[i]
        end = min(i + max_hold, n - 1)
        if sigma == 0.0:
            pnl = closes[end] - entry
            labels[i] = 1.0 if pnl > 0 else -1.0
            continue
        upper = entry * (1.0 + pt_mult * sigma)
        lower = entry * (1.0 - sl_mult * sigma)
        hit = np.nan
        for j in range(i + 1, end + 1):
            if highs[j] >= upper:
                hit = 1.0
                break
            if lows[j] <= lower:
                hit = -1.0
                break
        if np.isnan(hit):
            pnl = closes[end] - entry
            hit = 1.0 if pnl > 0 else -1.0
        labels[i] = hit
    return labels


@njit
def _ewm_vol(closes, span):
    """EWM standard deviation of returns."""
    n = len(closes)
    vol = np.empty(n)
    vol[0] = np.nan
    alpha = 2.0 / (span + 1.0)
    var = 0.0
    prev = closes[0]
    for i in range(1, n):
        ret = (closes[i] - prev) / prev if prev != 0 else 0.0
        var = (1.0 - alpha) * var + alpha * ret * ret
        vol[i] = np.sqrt(var)
        prev = closes[i]
    return vol


class TripleBarrierLabeler:
    def __init__(
        self,
        pt_multiplier: float = 1.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 4,
        volatility_span: int = 24,
    ):
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.volatility_span = volatility_span

    def compute_volatility(self, close_prices: pd.Series) -> np.ndarray:
        return _ewm_vol(close_prices.values.astype(np.float64), self.volatility_span)

    def label(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        closes = ohlcv_df["close"].values.astype(np.float64)
        highs = ohlcv_df["high"].values.astype(np.float64)
        lows = ohlcv_df["low"].values.astype(np.float64)
        vol = self.compute_volatility(ohlcv_df["close"])

        labels = _tbm_label(
            closes, highs, lows, vol,
            self.pt_multiplier, self.sl_multiplier, self.max_holding_bars,
        )
        return pd.Series(labels, index=ohlcv_df.index, dtype=float)

    def label_multi(
        self, ohlcv_df: pd.DataFrame, params: list[tuple[float, float, int]],
    ) -> pd.DataFrame:
        """Generate multi-label matrix for many (pt, sl, max_hold) combos.

        Returns DataFrame with one column per combo, named 'pt{pt}_sl{sl}_h{h}'.
        """
        closes = ohlcv_df["close"].values.astype(np.float64)
        highs = ohlcv_df["high"].values.astype(np.float64)
        lows = ohlcv_df["low"].values.astype(np.float64)
        vol = self.compute_volatility(ohlcv_df["close"])

        result = {}
        for pt, sl, h in params:
            col = f"pt{pt}_sl{sl}_h{h}"
            result[col] = _tbm_label(closes, highs, lows, vol, pt, sl, h)
        return pd.DataFrame(result, index=ohlcv_df.index)
