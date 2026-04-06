"""
Breakout Filter — Trade only when conditions align

Core concept: 저변동성 → 돌파 시에만 진입
  1. Volatility squeeze 감지 (ATR < rolling median)
  2. Squeeze 해제 + 방향 확인 → 진입
  3. 고확신 시에만 레버리지 (top 10% EV)

"항상 매매하지 않는다. 제대로 된 것만 먹는다."
"""

import numpy as np
import pandas as pd


class BreakoutFilter:
    """Filter signals to only trade on volatility breakouts.

    Stages:
      1. SQUEEZE: ATR below median → accumulating energy
      2. BREAKOUT: ATR crosses above median + price move → TRADE
      3. TREND: riding the move
      4. COOL: after exit, wait before next entry
    """

    def __init__(
        self,
        atr_period: int = 14,
        squeeze_lookback: int = 96,    # 1 day of 15m bars
        squeeze_threshold: float = 0.8, # ATR < 80% of median = squeeze
        breakout_threshold: float = 1.2, # ATR > 120% of median = breakout
        min_ev_percentile: float = 0.90, # only top 10% EV signals
        cooldown_bars: int = 16,         # 4 hours after exit
    ):
        self.atr_period = atr_period
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_threshold = squeeze_threshold
        self.breakout_threshold = breakout_threshold
        self.min_ev_percentile = min_ev_percentile
        self.cooldown_bars = cooldown_bars

    def detect_breakouts(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Detect breakout bars.

        Returns boolean array: True = breakout active.
        """
        n = len(close)

        # ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.concatenate([[0], tr])
        atr = pd.Series(tr).ewm(span=self.atr_period).mean().values

        # Rolling median ATR
        atr_median = pd.Series(atr).rolling(self.squeeze_lookback).median().values

        # Squeeze: ATR < threshold * median
        is_squeeze = atr < self.squeeze_threshold * atr_median

        # Breakout: ATR > threshold * median AND was in squeeze recently
        is_breakout = np.zeros(n, dtype=bool)
        was_squeeze = np.zeros(n, dtype=bool)

        for i in range(self.squeeze_lookback, n):
            # Was in squeeze in last 24 bars?
            if np.any(is_squeeze[max(0, i - 24):i]):
                was_squeeze[i] = True

            # Current ATR exceeds breakout threshold
            if not np.isnan(atr_median[i]) and atr[i] > self.breakout_threshold * atr_median[i]:
                if was_squeeze[i]:
                    is_breakout[i] = True

        return is_breakout

    def filter_signals(
        self,
        signals: list[dict],
        is_breakout: bool,
        all_evs: list[float] = None,
    ) -> list[dict]:
        """Filter signals based on breakout state and EV quality.

        Only passes signals when:
          1. We're in a breakout state
          2. Signal EV is in top percentile

        Args:
            signals: list of signal dicts with 'ev' key
            is_breakout: whether current bar is a breakout
            all_evs: historical EV values for percentile calculation
        """
        if not is_breakout:
            return []  # no trade outside breakouts

        if not signals:
            return []

        # Only keep top EV signals
        if all_evs and len(all_evs) > 100:
            ev_threshold = np.percentile(all_evs, self.min_ev_percentile * 100)
            signals = [s for s in signals if s["ev"] > ev_threshold]

        return signals


def compute_breakout_mask(kline_15m: pd.DataFrame, **kwargs) -> np.ndarray:
    """Convenience: compute breakout mask for a kline DataFrame."""
    bf = BreakoutFilter(**kwargs)
    return bf.detect_breakouts(
        kline_15m["high"].values,
        kline_15m["low"].values,
        kline_15m["close"].values,
    )
