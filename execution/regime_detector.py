"""
Regime Detector — Real-time market regime classification

4 regimes + transitions:
  SURGE:    Strong uptrend (momentum + volume confirmation)
  DUMP:     Strong downtrend (panic + volume spike)
  RANGE:    Sideways consolidation (low directional movement)
  VOLATILE: High volatility without clear direction (whipsaw)

Transition detection enables regime-switch trades:
  RANGE → SURGE  = breakout long opportunity
  RANGE → DUMP   = breakdown short opportunity
  VOLATILE → RANGE = mean reversion opportunity
  etc.

"시장이 뭘 하고있는지 모르면 매매하지않는다."
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class Regime(Enum):
    SURGE = "surge"
    DUMP = "dump"
    RANGE = "range"
    VOLATILE = "volatile"


@dataclass
class RegimeState:
    """Current regime with confidence and transition info."""
    regime: Regime
    confidence: float            # 0-1 how sure we are
    duration_bars: int           # how long in this regime
    prev_regime: Regime | None   # for transition detection
    transition_age: int          # bars since last regime change
    metrics: dict = field(default_factory=dict)

    @property
    def is_transition(self) -> bool:
        """True if regime changed recently (within 4 bars)."""
        return self.transition_age <= 4

    @property
    def transition_type(self) -> str | None:
        if not self.is_transition or self.prev_regime is None:
            return None
        return f"{self.prev_regime.value}_to_{self.regime.value}"


class RegimeDetector:
    """Real-time regime classification from OHLCV data.

    Uses multiple signals for robust regime detection:
    1. Directional movement (ADX-like)
    2. Volatility regime (ATR vs rolling median)
    3. Volume confirmation
    4. Price position relative to recent range
    """

    def __init__(
        self,
        lookback: int = 96,          # 1 day of 15m bars
        atr_period: int = 14,
        adx_period: int = 14,
        vol_squeeze_pct: float = 0.8,  # ATR < 80% median = low vol
        vol_expand_pct: float = 1.3,   # ATR > 130% median = high vol
        trend_threshold: float = 25,    # ADX > 25 = trending
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.vol_squeeze_pct = vol_squeeze_pct
        self.vol_expand_pct = vol_expand_pct
        self.trend_threshold = trend_threshold

        # State tracking per coin
        self._states: dict[str, RegimeState] = {}

    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Detect regime for each bar. Returns array of Regime enum values."""
        n = len(close)
        regimes = np.empty(n, dtype=object)
        regimes[:] = Regime.RANGE  # default

        if n < self.lookback:
            return regimes

        # 1. ATR for volatility
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.concatenate([[high[0] - low[0]], tr])
        atr = pd.Series(tr).ewm(span=self.atr_period).mean().values
        atr_median = pd.Series(atr).rolling(self.lookback, min_periods=20).median().values

        # 2. Directional movement (simplified ADX)
        up_move = np.diff(high, prepend=high[0])
        down_move = np.diff(low, prepend=low[0]) * -1  # negate so positive = down move
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = pd.Series(plus_dm).ewm(span=self.adx_period).mean().values
        minus_di = pd.Series(minus_dm).ewm(span=self.adx_period).mean().values
        di_sum = plus_di + minus_di
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.where(di_sum > 0, np.abs(plus_di - minus_di) / di_sum * 100, 0.0)
            dx = np.nan_to_num(dx, 0.0)
        adx = pd.Series(dx).ewm(span=self.adx_period).mean().values

        # 3. Returns for direction
        ret_window = min(24, self.lookback // 4)
        returns = np.zeros(n)
        returns[ret_window:] = (close[ret_window:] - close[:-ret_window]) / close[:-ret_window]

        # 4. Volume surge
        vol_ma = pd.Series(volume).rolling(self.lookback, min_periods=20).mean().values
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
            vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)

        # Classification logic
        for i in range(self.lookback, n):
            if np.isnan(atr_median[i]) or atr_median[i] <= 0:
                continue

            vol_regime = atr[i] / atr_median[i]  # >1 = expanding, <1 = contracting
            is_trending = adx[i] > self.trend_threshold
            is_vol_high = vol_regime > self.vol_expand_pct
            is_vol_low = vol_regime < self.vol_squeeze_pct
            direction = returns[i]
            has_volume = vol_ratio[i] > 1.2

            if is_trending and direction > 0 and (has_volume or is_vol_high):
                regimes[i] = Regime.SURGE
            elif is_trending and direction < 0 and (has_volume or is_vol_high):
                regimes[i] = Regime.DUMP
            elif is_vol_high and not is_trending:
                regimes[i] = Regime.VOLATILE
            elif is_vol_low or not is_trending:
                regimes[i] = Regime.RANGE
            # else default RANGE

        return regimes

    def detect_current(
        self,
        coin: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> RegimeState:
        """Detect current regime for a specific coin with state tracking."""
        regimes = self.detect(high, low, close, volume)

        current = regimes[-1]
        prev_state = self._states.get(coin)

        # Calculate confidence based on regime consistency
        recent = regimes[-min(8, len(regimes)):]
        same_count = sum(1 for r in recent if r == current)
        confidence = same_count / len(recent)

        # Transition tracking
        if prev_state is None or prev_state.regime != current:
            prev_regime = prev_state.regime if prev_state else None
            state = RegimeState(
                regime=current,
                confidence=confidence,
                duration_bars=1,
                prev_regime=prev_regime,
                transition_age=0,
                metrics=self._compute_metrics(high, low, close, volume),
            )
        else:
            state = RegimeState(
                regime=current,
                confidence=confidence,
                duration_bars=prev_state.duration_bars + 1,
                prev_regime=prev_state.prev_regime,
                transition_age=prev_state.transition_age + 1,
                metrics=self._compute_metrics(high, low, close, volume),
            )

        self._states[coin] = state
        return state

    def _compute_metrics(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> dict:
        """Compute auxiliary metrics for the current bar."""
        n = len(close)
        if n < 20:
            return {}

        ret_1h = (close[-1] - close[-4]) / close[-4] if n >= 4 else 0
        ret_4h = (close[-1] - close[-16]) / close[-16] if n >= 16 else 0
        ret_1d = (close[-1] - close[-96]) / close[-96] if n >= 96 else 0

        vol_20 = np.std(np.diff(close[-21:]) / close[-21:-1]) if n >= 21 else 0

        # Price position in 24h range
        if n >= 96:
            h24 = np.max(high[-96:])
            l24 = np.min(low[-96:])
            pos = (close[-1] - l24) / (h24 - l24) if h24 > l24 else 0.5
        else:
            pos = 0.5

        return {
            "ret_1h": ret_1h,
            "ret_4h": ret_4h,
            "ret_1d": ret_1d,
            "volatility_20": vol_20,
            "price_position_24h": pos,
        }

    def get_regime_for_strategy(self, regime: Regime) -> list[str]:
        """Map regime to compatible strategy types.

        SURGE → trend following (long), momentum
        DUMP  → trend following (short), momentum
        RANGE → mean reversion, scalping, grid
        VOLATILE → breakout, momentum scalp
        """
        mapping = {
            Regime.SURGE: ["trend_long", "momentum_long", "swing_long"],
            Regime.DUMP: ["trend_short", "momentum_short", "swing_short"],
            Regime.RANGE: ["mean_reversion", "scalp", "grid", "range_bound"],
            Regime.VOLATILE: ["breakout", "momentum_scalp", "volatility_capture"],
        }
        return mapping.get(regime, ["scalp"])
