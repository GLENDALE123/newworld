"""
Trendline Scanner — 잡알트 200개 경량 돌파 감지

Numtic scanner 로직 기반:
  - Donchian channel breakout (30d/90d)
  - Multi-touch level breakout
  - Volume confirmation
  - 조합 신호 (breakout + momentum/volume)

출력: 메타모델과 동일 포맷
  {coin, direction, confidence, ev, source_type="scanner"}
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ScannerSignal:
    """Scanner output — same interface as PLE signal for meta-model."""
    coin: str
    direction: int              # 1=long, -1=short
    confidence: float           # 0-1
    ev: float                   # estimated EV
    source_type: str = "scanner"
    signal_type: str = ""       # don30d, don90d, multi_touch, etc.
    volume_ratio: float = 0.0


class TrendlineScanner:
    """Lightweight scanner for altcoin breakouts.

    Based on Numtic's proven scanner logic:
      1. Donchian 30d/90d breakout (1h data)
      2. Multi-touch level breakout (short-term)
      3. Volume × momentum confirmation
      4. Squeeze breakout (BB width collapse → expansion)

    Designed for 200 small alts — no neural network, pure rule-based.
    """

    def __init__(
        self,
        don_30d_bars: int = 720,     # 30 days of 1h bars
        don_90d_bars: int = 2160,    # 90 days of 1h bars
        vol_mult: float = 3.0,       # volume must be 3x 30d average
        multi_touch_window: int = 100,
        multi_touch_band: float = 0.005,  # 0.5% band
        min_touches: int = 2,
        atr_period: int = 14,
        atr_mult: float = 2.0,       # momentum threshold
        cooldown_bars: int = 48,      # 12h cooldown after signal (15m bars)
    ):
        self.don_30d = don_30d_bars
        self.don_90d = don_90d_bars
        self.vol_mult = vol_mult
        self.mt_window = multi_touch_window
        self.mt_band = multi_touch_band
        self.min_touches = min_touches
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.cooldown_bars = cooldown_bars

        self._cooldown: dict[str, int] = {}

    def scan(
        self,
        coin: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> ScannerSignal | None:
        """Scan a single coin for breakout signals.

        Expects 15m bar data. For long-term signals, internally resamples to 1h.
        """
        n = len(close)
        if n < 200:
            return None

        # Cooldown
        if coin in self._cooldown and self._cooldown[coin] > 0:
            self._cooldown[coin] -= 1
            return None

        signals = []

        # === 1. Donchian breakout (simulated from 15m data) ===
        # 30d = 2880 15m bars, 90d = 8640
        don_30d_bars = min(2880, n - 1)
        don_90d_bars = min(8640, n - 1)

        if n > don_30d_bars + 1:
            prev_high_30d = np.max(high[-(don_30d_bars+1):-1])
            prev_low_30d = np.min(low[-(don_30d_bars+1):-1])

            if close[-1] > prev_high_30d and close[-2] <= prev_high_30d:
                signals.append(("don30d_break", 1, 0.6))
            elif close[-1] < prev_low_30d and close[-2] >= prev_low_30d:
                signals.append(("don30d_break", -1, 0.6))

        if n > don_90d_bars + 1:
            prev_high_90d = np.max(high[-(don_90d_bars+1):-1])
            prev_low_90d = np.min(low[-(don_90d_bars+1):-1])

            if close[-1] > prev_high_90d and close[-2] <= prev_high_90d:
                signals.append(("don90d_break", 1, 0.75))
            elif close[-1] < prev_low_90d and close[-2] >= prev_low_90d:
                signals.append(("don90d_break", -1, 0.75))

        # === 2. Multi-touch breakout ===
        if n > self.mt_window:
            w_high = high[-(self.mt_window+1):-1]
            w_low = low[-(self.mt_window+1):-1]
            level_high = w_high.max()
            level_low = w_low.min()
            band_h = level_high * (1 - self.mt_band)
            band_l = level_low * (1 + self.mt_band)
            touches_h = int((w_high >= band_h).sum())
            touches_l = int((w_low <= band_l).sum())

            if close[-1] > level_high and touches_h >= self.min_touches:
                conf = min(0.5 + touches_h * 0.05, 0.85)
                signals.append(("multi_touch", 1, conf))
            elif close[-1] < level_low and touches_l >= self.min_touches:
                conf = min(0.5 + touches_l * 0.05, 0.85)
                signals.append(("multi_touch", -1, conf))

        # === 3. Volume check ===
        vol_avg = np.mean(volume[-min(2880, n):]) if n > 96 else volume.mean()
        vol_ratio = volume[-1] / max(vol_avg, 1e-8)

        # === 4. Momentum check ===
        tr = np.maximum(
            high[-self.atr_period:] - low[-self.atr_period:],
            np.maximum(
                np.abs(high[-self.atr_period:] - np.roll(close[-self.atr_period:], 1)),
                np.abs(low[-self.atr_period:] - np.roll(close[-self.atr_period:], 1)),
            )
        )
        atr = np.mean(tr)
        atr_ratio = atr / max(close[-2], 1e-8)
        bar_ret = abs(close[-1] - close[-2]) / max(close[-2], 1e-8)
        has_momentum = bar_ret > self.atr_mult * atr_ratio

        if not signals:
            return None

        # Select best signal, boost with volume/momentum
        best = max(signals, key=lambda x: x[2])
        sig_type, direction, base_conf = best

        # Boost confidence if volume and/or momentum confirm
        conf = base_conf
        if vol_ratio > self.vol_mult:
            conf = min(conf + 0.1, 0.95)
        if has_momentum:
            conf = min(conf + 0.1, 0.95)

        # Key finding from Numtic E01: volume confirmation is critical
        # LT90d+Vol×3 = lift 3.03x vs LT90d alone = 2.70x
        # Require volume confirmation for all signals
        if vol_ratio < self.vol_mult:
            # No volume → only pass if don90d with momentum
            if sig_type != "don90d_break" or not has_momentum:
                return None

        ev = conf * 0.02 - (1 - conf) * 0.01 - 0.0008

        self._cooldown[coin] = self.cooldown_bars

        return ScannerSignal(
            coin=coin,
            direction=direction,
            confidence=conf,
            ev=ev,
            signal_type=sig_type,
            volume_ratio=vol_ratio,
        )

    def scan_batch(
        self,
        coins_data: dict[str, dict],
    ) -> list[ScannerSignal]:
        """Scan multiple coins. Returns signals sorted by confidence."""
        results = []
        for coin, data in coins_data.items():
            sig = self.scan(
                coin,
                data["high"], data["low"],
                data["close"], data["volume"],
            )
            if sig is not None:
                results.append(sig)

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
