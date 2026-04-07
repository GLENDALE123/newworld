"""
Dynamic Portfolio Manager — EV/hour + Dynamic Leverage

핵심:
  1. EV/hour로 전략 랭킹 (15분 단타 20x > 2시간 스윙 5x)
  2. 동적 레버리지 1x ~ 60x (확신도 × regime × DD 함수)
  3. 포지션 사이즈 = EV 비례
  4. DD circuit breaker

"시간대비 수익률이 가장 높은 전략을 가장 높은 확신도로 실행"
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TradeRequest:
    """Input from meta-model → portfolio manager."""
    coin: str
    direction: int              # 1=long, -1=short
    confidence: float           # 0-1 from meta-model
    ev: float                   # expected value per trade
    hold_bars: int              # expected holding period (15m bars)
    mfe: float                  # predicted max favorable excursion
    mae: float                  # predicted max adverse excursion
    strategy_type: str          # trend/mean_reversion/breakout/scalp
    source: str                 # "ple" or "scanner"
    volatility: float = 0.02   # coin's realized volatility


@dataclass
class ExecutionOrder:
    """Output from portfolio manager → execution layer."""
    coin: str
    direction: int
    size_pct: float             # % of equity
    leverage: float             # 1x ~ 60x
    tp_pct: float               # take profit %
    sl_pct: float               # stop loss %
    hold_limit_bars: int
    ev_per_hour: float
    reason: str


class DynamicPortfolioManager:
    """EV/hour 기반 전략 선택 + 동적 레버리지.

    레버리지 결정 요소:
      1. 확신도 (confidence) — 높을수록 높은 레버리지
      2. DD 상태 — DD 깊으면 레버리지 낮춤
      3. 변동성 — 고변동성이면 레버리지 낮춤
      4. 자본 단계 — 소자본이면 기본 레버리지 높음

    확신 최상 + 시장 좋으면 60x도 가능.
    """

    # Capital stages
    STAGES = [
        {"max_equity": 500,    "base_leverage": 20, "max_leverage": 60, "base_size": 0.30},
        {"max_equity": 5000,   "base_leverage": 10, "max_leverage": 40, "base_size": 0.15},
        {"max_equity": 50000,  "base_leverage": 5,  "max_leverage": 20, "base_size": 0.08},
        {"max_equity": 100000, "base_leverage": 3,  "max_leverage": 10, "base_size": 0.05},
        {"max_equity": float("inf"), "base_leverage": 1, "max_leverage": 5, "base_size": 0.03},
    ]

    def __init__(
        self,
        initial_equity: float = 500,
        max_positions: int = 3,
        max_dd: float = 0.30,
        fee_pct: float = 0.0008,
    ):
        self.equity = initial_equity
        self.peak = initial_equity
        self.max_positions = max_positions
        self.max_dd = max_dd
        self.fee_pct = fee_pct

        self.open_positions: list[dict] = []
        self.trade_count = 0

    @property
    def dd_pct(self) -> float:
        return (self.peak - self.equity) / self.peak if self.peak > 0 else 0

    @property
    def dd_ratio(self) -> float:
        """0 = no DD, 1 = at max DD."""
        return min(self.dd_pct / self.max_dd, 1.0)

    @property
    def stage(self) -> dict:
        for s in self.STAGES:
            if self.equity <= s["max_equity"]:
                return s
        return self.STAGES[-1]

    @property
    def available_slots(self) -> int:
        return self.max_positions - len(self.open_positions)

    def compute_ev_per_hour(self, request: TradeRequest) -> float:
        """EV/hour = EV / (hold_bars × 0.25h per 15m bar)."""
        hours = request.hold_bars * 0.25  # 15m bars to hours
        if hours <= 0:
            return 0
        return request.ev / hours

    def compute_leverage(self, request: TradeRequest) -> float:
        """Dynamic leverage: 1x ~ 60x.

        leverage = base × confidence_mult × dd_adj × vol_adj

        예시:
          conf=0.85, DD=0%, vol=BTC수준 → 60x (최대)
          conf=0.60, DD=10%, vol=높음    → 5x
          conf=0.55, DD=25%              → 1x (최소)
        """
        stage = self.stage

        # Confidence multiplier: 0.55→0.5x, 0.70→1.0x, 0.85→2.0x, 0.95→3.0x
        conf = request.confidence
        if conf < 0.55:
            return 0  # don't trade
        conf_mult = max(0, (conf - 0.55) / 0.15)  # 0 at 0.55, 1 at 0.70, 2 at 0.85
        conf_mult = min(conf_mult, 3.0)

        # DD adjustment: more DD → less leverage
        dd_adj = max(0.1, 1.0 - self.dd_ratio * 0.8)

        # Volatility adjustment: higher vol → less leverage
        # BTC ~2% daily vol → ratio 1.0
        btc_vol = 0.02
        vol_ratio = max(request.volatility / btc_vol, 0.3)
        vol_adj = 1.0 / vol_ratio

        leverage = stage["base_leverage"] * conf_mult * dd_adj * vol_adj

        # Clamp
        leverage = max(1.0, min(leverage, stage["max_leverage"]))

        return round(leverage, 1)

    def compute_size(self, request: TradeRequest, leverage: float) -> float:
        """Position size as % of equity.

        Size = base_size × (EV / baseline_EV) × dd_adj
        EV-proportional sizing: higher EV → larger position.
        """
        stage = self.stage
        base = stage["base_size"]

        # EV proportional: normalize to baseline EV of 0.005 (0.5%)
        ev_mult = min(request.ev / 0.005, 3.0) if request.ev > 0 else 0

        # DD adjustment
        dd_adj = max(0.2, 1.0 - self.dd_ratio * 0.6)

        size = base * max(ev_mult, 0.3) * dd_adj

        # Cap: don't risk more than max per trade
        max_risk = 0.50 if self.equity < 500 else 0.25
        size = min(size, max_risk)

        # Cap total exposure
        current_exposure = sum(p.get("size_pct", 0) * p.get("leverage", 1)
                               for p in self.open_positions)
        max_exposure = 3.0 if self.equity < 5000 else 1.5
        remaining = max_exposure - current_exposure
        if size * leverage > remaining:
            size = remaining / max(leverage, 1)

        return max(0.01, round(size, 4))

    def compute_tp_sl(self, request: TradeRequest, leverage: float) -> tuple[float, float]:
        """TP/SL based on MAE/MFE and leverage.

        Higher leverage → tighter stops (protect capital).
        """
        # Base from model predictions
        tp = abs(request.mfe) if request.mfe > 0 else 0.02
        sl = abs(request.mae) if request.mae < 0 else 0.01

        # Leverage-adjusted: higher leverage needs tighter stops
        # At 60x, a 1.67% adverse move = 100% loss
        max_loss_pct = 0.05  # max 5% equity loss per trade
        sl_cap = max_loss_pct / max(leverage, 1)
        sl = min(sl, sl_cap)

        return round(tp, 5), round(sl, 5)

    def evaluate(self, requests: list[TradeRequest]) -> list[ExecutionOrder]:
        """Evaluate trade requests and produce execution orders.

        Process:
          1. Compute EV/hour for each request
          2. Rank by EV/hour
          3. Compute leverage and size for top candidates
          4. Check portfolio constraints
          5. Return execution orders
        """
        if not requests or self.available_slots <= 0:
            return []

        # DD circuit breaker
        if self.dd_ratio >= 0.95:
            return []

        # Filter: already holding this coin?
        open_coins = {p["coin"] for p in self.open_positions}
        eligible = [r for r in requests if r.coin not in open_coins]

        if not eligible:
            return []

        # Compute EV/hour and rank
        scored = []
        for req in eligible:
            ev_h = self.compute_ev_per_hour(req)
            if ev_h <= 0:
                continue
            scored.append((req, ev_h))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top candidates
        orders = []
        for req, ev_h in scored[:self.available_slots]:
            leverage = self.compute_leverage(req)
            if leverage < 1:
                continue

            size = self.compute_size(req, leverage)
            tp, sl = self.compute_tp_sl(req, leverage)

            order = ExecutionOrder(
                coin=req.coin,
                direction=req.direction,
                size_pct=size,
                leverage=leverage,
                tp_pct=tp,
                sl_pct=sl,
                hold_limit_bars=req.hold_bars,
                ev_per_hour=ev_h,
                reason=f"{req.strategy_type} conf={req.confidence:.2f} "
                       f"EV/h={ev_h:.4f} lev={leverage}x",
            )
            orders.append(order)

        return orders

    def update_equity(self, equity: float):
        """Update after trade execution."""
        self.equity = equity
        self.peak = max(self.peak, equity)

    def summary(self) -> str:
        stage = self.stage
        return (f"${self.equity:,.0f} "
                f"base_lev={stage['base_leverage']}x "
                f"max_lev={stage['max_leverage']}x "
                f"DD={self.dd_pct*100:.1f}% "
                f"slots={self.available_slots}/{self.max_positions}")
