import numpy as np
import pandas as pd


class TripleBarrierLabeler:
    def __init__(
        self,
        pt_multiplier: float = 1.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 4,
    ):
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars

    def compute_volatility(self, close_prices: pd.Series, span: int = 24) -> pd.Series:
        returns = close_prices.pct_change()
        vol = returns.ewm(span=span).std()
        return vol

    def label(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        closes = ohlcv_df["close"]
        highs = ohlcv_df["high"]
        lows = ohlcv_df["low"]
        vol = self.compute_volatility(closes)

        labels = pd.Series(index=ohlcv_df.index, dtype=float)

        for i in range(len(ohlcv_df)):
            sigma_t = vol.iloc[i]
            if np.isnan(sigma_t):
                labels.iloc[i] = np.nan
                continue

            entry_price = closes.iloc[i]

            end_idx = min(i + self.max_holding_bars, len(ohlcv_df) - 1)
            if i >= len(ohlcv_df) - 1:
                labels.iloc[i] = np.nan
                continue

            # When volatility is zero, barriers collapse to entry price;
            # skip directly to the vertical (time) barrier.
            if sigma_t == 0:
                exit_price = closes.iloc[end_idx]
                pnl = exit_price - entry_price
                labels.iloc[i] = 1.0 if pnl > 0 else -1.0
                continue

            upper = entry_price * (1 + self.pt_multiplier * sigma_t)
            lower = entry_price * (1 - self.sl_multiplier * sigma_t)

            hit_label = np.nan
            for j in range(i + 1, end_idx + 1):
                if highs.iloc[j] >= upper:
                    hit_label = 1.0
                    break
                if lows.iloc[j] <= lower:
                    hit_label = -1.0
                    break

            if np.isnan(hit_label):
                exit_price = closes.iloc[end_idx]
                pnl = exit_price - entry_price
                hit_label = 1.0 if pnl > 0 else -1.0

            labels.iloc[i] = hit_label

        return labels
