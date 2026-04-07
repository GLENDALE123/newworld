"""
Strategy Router — Multi-dimensional trade selection engine

Takes model's multi-label output (128 strategy probabilities) + regime detection
→ selects the best trade considering category, regime, and conviction.

This is the brain that connects:
  - PLE model output (what strategies are profitable?)
  - Regime detection (what is the market doing?)
  - Coin category (what strategies are allowed?)
  - EV calculation (is this trade worth taking?)

"모든 전략을 동시에 감시하고, 가장 확신 높은 것만 실행한다."
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum

from execution.regime_detector import Regime, RegimeState


# ── Label Index Mapping ────────────────────────────────────────────────────

STRATEGY_TYPES = ["scalp", "intraday", "daytrade", "swing"]
DIRECTIONS = ["long", "short"]
REGIMES = ["surge", "dump", "range", "volatile"]


def build_label_index() -> dict[str, int]:
    """Build mapping from label name → index in model output.

    Order: scalp_long_surge=0, scalp_long_dump=1, ... swing_short_volatile=31
    """
    idx = 0
    mapping = {}
    for strat in STRATEGY_TYPES:
        for direction in DIRECTIONS:
            for regime in REGIMES:
                mapping[f"{strat}_{direction}_{regime}"] = idx
                idx += 1
    return mapping


LABEL_INDEX = build_label_index()
N_LABELS = len(LABEL_INDEX)  # 32


# ── Trade Signal ──────────────────────────────────────────────────────────

class TradeType(Enum):
    TREND_FOLLOW = "trend_follow"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    SCALP = "scalp"


@dataclass
class TradeSignal:
    """A concrete trade recommendation from the router."""
    coin: str
    direction: str             # "long" or "short"
    strategy_type: TradeType
    strategy_name: str         # e.g. "daytrade_long_surge"
    probability: float         # model's P(profitable) for this strategy
    ev: float                  # expected value
    regime: Regime
    regime_confidence: float
    category: str              # "major" / "large_alt" / "small_alt"
    leverage_mult: float       # suggested leverage multiplier (0-1)
    hold_bars: int             # expected hold period
    tp_atr_mult: float
    sl_atr_mult: float
    meta: dict | None = None

    @property
    def score(self) -> float:
        """Overall conviction score for ranking signals."""
        return self.probability * self.ev * self.regime_confidence


# ── Category Strategy Rules ───────────────────────────────────────────────

# What strategies are allowed per category + regime combination
CATEGORY_RULES = {
    "major": {
        # Majors: ALL strategy types, regime-adaptive
        # "추세추종만하는게 맞아???" → NO, multi-dimensional
        Regime.SURGE: {
            "allowed_strategies": ["daytrade", "swing", "intraday"],
            "allowed_directions": ["long"],  # follow the trend
            "trade_types": [TradeType.TREND_FOLLOW, TradeType.MOMENTUM],
            "min_probability": 0.55,
            "leverage_factor": 1.0,
        },
        Regime.DUMP: {
            "allowed_strategies": ["daytrade", "swing", "intraday"],
            "allowed_directions": ["short"],
            "trade_types": [TradeType.TREND_FOLLOW, TradeType.MOMENTUM],
            "min_probability": 0.55,
            "leverage_factor": 1.0,
        },
        Regime.RANGE: {
            "allowed_strategies": ["scalp", "intraday"],
            "allowed_directions": ["long", "short"],  # both directions OK
            "trade_types": [TradeType.MEAN_REVERSION, TradeType.SCALP],
            "min_probability": 0.60,  # higher bar for range trading
            "leverage_factor": 0.7,   # lower leverage in range
        },
        Regime.VOLATILE: {
            "allowed_strategies": ["scalp", "intraday"],
            "allowed_directions": ["long", "short"],
            "trade_types": [TradeType.BREAKOUT, TradeType.MOMENTUM],
            "min_probability": 0.60,
            "leverage_factor": 0.5,  # very cautious in volatile
        },
    },
    "large_alt": {
        # Large alts: selective, medium-term, need high EV
        Regime.SURGE: {
            "allowed_strategies": ["intraday", "daytrade", "swing"],
            "allowed_directions": ["long"],
            "trade_types": [TradeType.TREND_FOLLOW, TradeType.MOMENTUM],
            "min_probability": 0.55,
            "leverage_factor": 1.0,
        },
        Regime.DUMP: {
            "allowed_strategies": ["intraday", "daytrade"],
            "allowed_directions": ["short"],
            "trade_types": [TradeType.TREND_FOLLOW],
            "min_probability": 0.58,
            "leverage_factor": 0.8,
        },
        Regime.RANGE: {
            "allowed_strategies": ["scalp", "intraday"],
            "allowed_directions": ["long", "short"],
            "trade_types": [TradeType.MEAN_REVERSION, TradeType.SCALP],
            "min_probability": 0.60,
            "leverage_factor": 0.6,
        },
        Regime.VOLATILE: {
            "allowed_strategies": ["scalp"],
            "allowed_directions": ["long", "short"],
            "trade_types": [TradeType.BREAKOUT],
            "min_probability": 0.65,
            "leverage_factor": 0.4,
        },
    },
    "small_alt": {
        # Small alts: breakout-focused, high conviction only
        # "저변동성코인은 돌파매매만 감지"
        Regime.SURGE: {
            "allowed_strategies": ["scalp", "intraday"],
            "allowed_directions": ["long"],
            "trade_types": [TradeType.BREAKOUT, TradeType.MOMENTUM],
            "min_probability": 0.60,
            "leverage_factor": 1.0,
        },
        Regime.DUMP: {
            "allowed_strategies": ["scalp", "intraday"],
            "allowed_directions": ["short"],
            "trade_types": [TradeType.BREAKOUT],
            "min_probability": 0.65,  # higher bar for shorting small alts
            "leverage_factor": 0.7,
        },
        Regime.RANGE: {
            # In range, small alts only trade on breakout transition
            "allowed_strategies": [],
            "allowed_directions": [],
            "trade_types": [],
            "min_probability": 1.0,  # effectively disabled
            "leverage_factor": 0.0,
        },
        Regime.VOLATILE: {
            "allowed_strategies": ["scalp"],
            "allowed_directions": ["long", "short"],
            "trade_types": [TradeType.BREAKOUT],
            "min_probability": 0.65,
            "leverage_factor": 0.5,
        },
    },
}

# Strategy type → hold period (15m bars)
HOLD_PERIODS = {
    "scalp": 3,       # ~45min
    "intraday": 8,    # ~2h
    "daytrade": 32,   # ~8h
    "swing": 96,      # ~24h
}

# Strategy type → ATR multipliers
ATR_PARAMS = {
    "scalp":    {"tp": 1.5, "sl": 1.0},
    "intraday": {"tp": 2.0, "sl": 1.0},
    "daytrade": {"tp": 2.5, "sl": 1.2},
    "swing":    {"tp": 3.0, "sl": 1.5},
}


def _classify_trade_type(
    strategy: str,
    direction: str,
    regime: Regime,
) -> TradeType:
    """Classify a strategy+direction+regime combo into a trade type."""
    # Breakout: volatile regime or transition from range
    if regime == Regime.VOLATILE:
        return TradeType.BREAKOUT

    # Mean reversion: range regime
    if regime == Regime.RANGE:
        return TradeType.MEAN_REVERSION

    # Scalp in trending = momentum scalp
    if strategy == "scalp" and regime in (Regime.SURGE, Regime.DUMP):
        return TradeType.MOMENTUM

    # Trend following: trending regime + longer hold
    if regime in (Regime.SURGE, Regime.DUMP):
        return TradeType.TREND_FOLLOW

    return TradeType.SCALP


# ── Strategy Router ───────────────────────────────────────────────────────

class StrategyRouter:
    """Select best trade from model's multi-label output.

    Process:
    1. Map model's 32 label probabilities to strategy names
    2. Filter by category rules (regime-specific)
    3. Calculate EV using P(win) and MAE/MFE
    4. Select highest-scoring signal that passes all filters
    """

    def __init__(
        self,
        min_ev: float = 0.001,          # minimum expected value
        max_signals_per_scan: int = 3,   # top N signals to return
    ):
        self.min_ev = min_ev
        self.max_signals = max_signals_per_scan

    def route(
        self,
        coin: str,
        category: str,
        label_probs: np.ndarray,      # (32,) P(profitable) per strategy
        mae_pred: np.ndarray,         # (32,) predicted MAE
        mfe_pred: np.ndarray,         # (32,) predicted MFE
        regime_state: RegimeState,
        confidence: float = 0.5,      # model's overall confidence
    ) -> list[TradeSignal]:
        """Route model output to concrete trade signals.

        Returns list of TradeSignal sorted by score (best first).
        """
        regime = regime_state.regime
        rules = CATEGORY_RULES.get(category, CATEGORY_RULES["small_alt"])
        regime_rules = rules.get(regime, {})

        if not regime_rules.get("allowed_strategies"):
            # Check if transition enables trading
            if regime_state.is_transition:
                return self._handle_transition(
                    coin, category, label_probs, mae_pred, mfe_pred, regime_state
                )
            return []

        allowed_strats = regime_rules["allowed_strategies"]
        allowed_dirs = regime_rules["allowed_directions"]
        min_prob = regime_rules["min_probability"]
        lev_factor = regime_rules["leverage_factor"]

        signals = []

        for label_name, idx in LABEL_INDEX.items():
            if idx >= len(label_probs):
                continue

            parts = label_name.split("_")
            strat = parts[0]
            direction = parts[1]
            label_regime = parts[2]

            # Filter: strategy allowed for this category+regime?
            if strat not in allowed_strats:
                continue
            if direction not in allowed_dirs:
                continue

            prob = label_probs[idx]
            if prob < min_prob:
                continue

            # EV = P(win) * E[win] - P(loss) * E[loss]
            mfe_val = float(mfe_pred[idx]) if idx < len(mfe_pred) else 0
            mae_val = float(mae_pred[idx]) if idx < len(mae_pred) else 0
            ev = prob * max(mfe_val, 0) - (1 - prob) * max(-mae_val, 0)

            if ev < self.min_ev:
                continue

            trade_type = _classify_trade_type(strat, direction, regime)
            atr_params = ATR_PARAMS[strat]

            signal = TradeSignal(
                coin=coin,
                direction=direction,
                strategy_type=trade_type,
                strategy_name=label_name,
                probability=prob,
                ev=ev,
                regime=regime,
                regime_confidence=regime_state.confidence,
                category=category,
                leverage_mult=lev_factor,
                hold_bars=HOLD_PERIODS[strat],
                tp_atr_mult=atr_params["tp"],
                sl_atr_mult=atr_params["sl"],
                meta={
                    "mfe_pred": mfe_val,
                    "mae_pred": mae_val,
                    "model_confidence": confidence,
                    "regime_duration": regime_state.duration_bars,
                },
            )
            signals.append(signal)

        # Sort by score (probability * EV * regime_confidence)
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals[:self.max_signals]

    def _handle_transition(
        self,
        coin: str,
        category: str,
        label_probs: np.ndarray,
        mae_pred: np.ndarray,
        mfe_pred: np.ndarray,
        regime_state: RegimeState,
    ) -> list[TradeSignal]:
        """Handle regime transitions — breakout opportunities.

        Key transitions:
          RANGE → SURGE/DUMP = breakout
          VOLATILE → RANGE = mean reversion setup
        """
        transition = regime_state.transition_type
        if transition is None:
            return []

        # Define transition opportunities
        transition_rules = {
            "range_to_surge": {"direction": "long", "type": TradeType.BREAKOUT, "strats": ["intraday", "daytrade"]},
            "range_to_dump": {"direction": "short", "type": TradeType.BREAKOUT, "strats": ["intraday", "daytrade"]},
            "range_to_volatile": {"direction": "long", "type": TradeType.BREAKOUT, "strats": ["scalp", "intraday"]},
            "volatile_to_range": {"direction": "long", "type": TradeType.MEAN_REVERSION, "strats": ["scalp"]},
            "dump_to_range": {"direction": "long", "type": TradeType.MEAN_REVERSION, "strats": ["scalp", "intraday"]},
            "surge_to_range": {"direction": "short", "type": TradeType.MEAN_REVERSION, "strats": ["scalp", "intraday"]},
        }

        rule = transition_rules.get(transition)
        if rule is None:
            return []

        signals = []
        for label_name, idx in LABEL_INDEX.items():
            if idx >= len(label_probs):
                continue

            parts = label_name.split("_")
            strat = parts[0]
            direction = parts[1]

            if strat not in rule["strats"]:
                continue
            if direction != rule["direction"]:
                continue

            prob = label_probs[idx]
            if prob < 0.55:  # transition trades need decent conviction
                continue

            mfe_val = float(mfe_pred[idx]) if idx < len(mfe_pred) else 0
            mae_val = float(mae_pred[idx]) if idx < len(mae_pred) else 0
            ev = prob * max(mfe_val, 0) - (1 - prob) * max(-mae_val, 0)

            if ev < self.min_ev:
                continue

            atr_params = ATR_PARAMS[strat]
            signal = TradeSignal(
                coin=coin,
                direction=direction,
                strategy_type=rule["type"],
                strategy_name=label_name,
                probability=prob,
                ev=ev,
                regime=regime_state.regime,
                regime_confidence=regime_state.confidence,
                category=category,
                leverage_mult=0.8,  # slightly cautious on transitions
                hold_bars=HOLD_PERIODS[strat],
                tp_atr_mult=atr_params["tp"],
                sl_atr_mult=atr_params["sl"],
                meta={
                    "transition": transition,
                    "mfe_pred": mfe_val,
                    "mae_pred": mae_val,
                },
            )
            signals.append(signal)

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals[:self.max_signals]

    def select_best(
        self,
        all_signals: list[TradeSignal],
        max_concurrent: int = 3,
    ) -> list[TradeSignal]:
        """From signals across all coins, select the best non-conflicting set.

        Rules:
        1. No duplicate coins
        2. No more than max_concurrent positions
        3. Diversify across categories if possible
        4. Prefer higher scores
        """
        if not all_signals:
            return []

        # Sort by score
        ranked = sorted(all_signals, key=lambda s: s.score, reverse=True)

        selected = []
        used_coins = set()
        category_counts = {"major": 0, "large_alt": 0, "small_alt": 0}

        for signal in ranked:
            if len(selected) >= max_concurrent:
                break
            if signal.coin in used_coins:
                continue

            # Soft limit: don't over-concentrate in one category
            cat = signal.category
            max_per_cat = max(1, max_concurrent // 2 + 1)
            if category_counts[cat] >= max_per_cat and len(selected) > 0:
                continue

            selected.append(signal)
            used_coins.add(signal.coin)
            category_counts[cat] += 1

        return selected
