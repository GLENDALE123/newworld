"""
Meta-Model — Trade Decision Supervisor

Sits above the base PLE models and decides WHEN and WHAT to trade.
Instead of fixed interval checking, it continuously monitors all signals
and makes portfolio-level decisions.

Architecture:
  Input:
    - Signal matrix: (N_coins, N_strategies) probabilities from PLE
    - Confidence vector: (N_coins,) base model confidence
    - Regime vector: (N_coins,) current regime per coin
    - Portfolio state: equity, DD, open positions, PnL
    - Market context: cross-coin signal agreement, BTC direction, volatility

  Output:
    - Trade decisions: list of (coin, direction, size, strategy) to execute
    - No-trade signal when conditions don't warrant action

"하위모델이 시그널을 보내면, 메타모델이 거래할지 결정한다."

Design principles:
  1. 하위모델은 매 bar마다 시그널 생성 (비용 낮음)
  2. 메타모델은 시그널을 종합하여 거래 결정 (선택적)
  3. 포트폴리오 수준의 리스크 관리
  4. 코인 간 상관관계 고려 (같은 방향 시그널이 많으면 → 시장 이벤트)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SignalSnapshot:
    """One coin's signal at a point in time."""
    coin: str
    coin_id: int
    category: str             # major / large_alt / small_alt
    probs: np.ndarray         # (N_strategies,) strategy probabilities
    confidence: float
    regime: int               # 0=surge, 1=dump, 2=range, 3=volatile
    ev_best: float            # best strategy's expected value
    direction_best: int       # 1=long, -1=short
    strategy_idx: int         # best strategy index
    volatility: float         # recent realized vol


@dataclass
class PortfolioState:
    """Current portfolio status."""
    equity: float
    peak_equity: float
    drawdown_pct: float
    n_open_positions: int
    max_positions: int
    open_coins: list[str] = field(default_factory=list)
    recent_pnl: list[float] = field(default_factory=list)  # last N trade PnLs

    @property
    def dd_ratio(self) -> float:
        """0 = no DD, 1 = at max allowed DD."""
        max_dd = 0.30 if self.equity < 5000 else 0.15
        return min(self.drawdown_pct / max_dd, 1.0)


@dataclass
class TradeDecision:
    """Meta-model's output: a concrete trade to execute."""
    coin: str
    direction: int            # 1=long, -1=short
    strategy_idx: int
    size_pct: float           # % of equity to risk
    leverage: float
    confidence: float         # meta-model's confidence in this trade
    reason: str               # why this trade was selected


class MetaModel:
    """Rule-based meta-model for trade decision making.

    Monitors all coin signals simultaneously and makes portfolio-level decisions.
    Unlike fixed-interval checking, this evaluates signals on every bar
    but applies intelligent filtering.

    Filters:
      1. Signal strength — only top signals pass
      2. Signal persistence — signal must persist for N bars
      3. Cross-coin agreement — detect market-wide moves vs individual alpha
      4. Portfolio constraints — DD limits, position limits, correlation
      5. Regime alignment — signal must align with detected regime
    """

    def __init__(
        self,
        top_k_signals: int = 5,       # consider top K signals
        persistence_bars: int = 2,     # signal must persist this many bars
        max_correlation: float = 0.7,  # reject if too correlated with open positions
        min_confidence: float = 0.55,  # minimum base model confidence
        min_ev: float = 0.001,         # minimum expected value
        market_agreement_limit: float = 0.8,  # if >80% coins agree, it's market move not alpha
    ):
        self.top_k = top_k_signals
        self.persistence_bars = persistence_bars
        self.max_correlation = max_correlation
        self.min_confidence = min_confidence
        self.min_ev = min_ev
        self.market_agreement_limit = market_agreement_limit

        # Signal persistence tracking
        self._signal_history: dict[str, deque] = {}  # coin → recent best signals
        self._ev_history: dict[str, deque] = {}       # coin → recent EVs

    def evaluate(
        self,
        signals: list[SignalSnapshot],
        portfolio: PortfolioState,
    ) -> list[TradeDecision]:
        """Evaluate all coin signals and decide what to trade.

        This is called every bar with fresh signals from all coins.
        """
        if not signals:
            return []

        # 1. Market context — detect market-wide moves
        market_ctx = self._analyze_market(signals)

        # 2. Filter signals
        candidates = self._filter_signals(signals, market_ctx, portfolio)

        # 3. Rank by meta-score
        ranked = self._rank_signals(candidates, market_ctx, portfolio)

        # 4. Select trades respecting portfolio constraints
        decisions = self._select_trades(ranked, portfolio)

        return decisions

    def _analyze_market(self, signals: list[SignalSnapshot]) -> dict:
        """Analyze cross-coin signal patterns."""
        n = len(signals)
        if n == 0:
            return {"agreement": 0, "dominant_direction": 0, "avg_confidence": 0}

        # Direction agreement
        directions = [s.direction_best for s in signals if s.ev_best > 0]
        if not directions:
            return {"agreement": 0, "dominant_direction": 0, "avg_confidence": 0}

        long_pct = sum(1 for d in directions if d == 1) / len(directions)
        agreement = max(long_pct, 1 - long_pct)  # how much coins agree
        dominant = 1 if long_pct > 0.5 else -1

        # Average confidence
        avg_conf = np.mean([s.confidence for s in signals])

        # Regime distribution
        regimes = [s.regime for s in signals]
        regime_counts = {r: regimes.count(r) for r in set(regimes)}

        # Volatility context
        avg_vol = np.mean([s.volatility for s in signals if s.volatility > 0])

        return {
            "agreement": agreement,
            "dominant_direction": dominant,
            "avg_confidence": avg_conf,
            "regime_distribution": regime_counts,
            "avg_volatility": avg_vol,
            "n_positive_ev": sum(1 for s in signals if s.ev_best > 0),
        }

    def _filter_signals(
        self,
        signals: list[SignalSnapshot],
        market_ctx: dict,
        portfolio: PortfolioState,
    ) -> list[SignalSnapshot]:
        """Apply multi-layer filtering."""
        filtered = []

        for sig in signals:
            # F1: Minimum confidence
            if sig.confidence < self.min_confidence:
                continue

            # F2: Minimum EV
            if sig.ev_best < self.min_ev:
                continue

            # F3: Don't add to already open positions
            if sig.coin in portfolio.open_coins:
                continue

            # F4: Signal persistence check
            self._update_history(sig)
            if not self._check_persistence(sig.coin):
                continue

            # F5: Market agreement filter
            # If too many coins agree on direction, it's likely a market move
            # Individual alpha is when a coin signals AGAINST the market
            is_contrarian = (sig.direction_best != market_ctx["dominant_direction"])
            if market_ctx["agreement"] > self.market_agreement_limit and not is_contrarian:
                # This coin is just following the market, not showing individual alpha
                # Still allow if EV is very high
                if sig.ev_best < self.min_ev * 3:
                    continue

            # F6: Drawdown circuit breaker
            if portfolio.dd_ratio > 0.8:
                # Only allow very high confidence signals during deep DD
                if sig.confidence < 0.65:
                    continue

            filtered.append(sig)

        return filtered

    def _rank_signals(
        self,
        candidates: list[SignalSnapshot],
        market_ctx: dict,
        portfolio: PortfolioState,
    ) -> list[tuple[SignalSnapshot, float]]:
        """Rank signals by meta-score considering portfolio context."""
        scored = []

        for sig in candidates:
            # Base score: EV * confidence
            base_score = sig.ev_best * sig.confidence

            # Bonus: contrarian signals (individual alpha, not market following)
            if sig.direction_best != market_ctx["dominant_direction"]:
                base_score *= 1.2

            # Bonus: alpha coins get priority
            # (these have proven per-trade alpha from historical analysis)
            if sig.category == "small_alt" and sig.ev_best > self.min_ev * 2:
                base_score *= 1.15  # small alts with high EV = breakout opportunity

            # Penalty: high volatility reduces score (risk-adjusted)
            if sig.volatility > 0:
                vol_penalty = min(sig.volatility / 0.05, 2.0)  # normalize to BTC-like vol
                base_score /= max(vol_penalty, 0.5)

            # Penalty: during drawdown, reduce willingness to trade
            dd_penalty = 1.0 - portfolio.dd_ratio * 0.5
            base_score *= dd_penalty

            # Persistence bonus: signal that persisted longer is more reliable
            persistence = self._get_persistence_length(sig.coin)
            if persistence > self.persistence_bars:
                base_score *= 1.0 + min(persistence - self.persistence_bars, 5) * 0.05

            scored.append((sig, base_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _select_trades(
        self,
        ranked: list[tuple[SignalSnapshot, float]],
        portfolio: PortfolioState,
    ) -> list[TradeDecision]:
        """Select actual trades from ranked candidates."""
        available_slots = portfolio.max_positions - portfolio.n_open_positions
        if available_slots <= 0:
            return []

        decisions = []
        selected_coins = set(portfolio.open_coins)
        selected_directions = []

        for sig, score in ranked[:self.top_k]:
            if len(decisions) >= available_slots:
                break

            if sig.coin in selected_coins:
                continue

            # Diversification: don't take all positions in same direction
            if len(selected_directions) >= 2:
                same_dir = sum(1 for d in selected_directions if d == sig.direction_best)
                if same_dir >= available_slots - 1:
                    continue  # force diversification

            # Size based on confidence and DD
            base_size = 0.15 if portfolio.equity < 5000 else 0.08
            size = base_size * sig.confidence * (1 - portfolio.dd_ratio * 0.5)

            # Leverage based on category and volatility
            if sig.category == "major":
                max_lev = 10
            elif sig.category == "large_alt":
                max_lev = 15
            else:
                max_lev = 20

            vol_adj = max(0.02 / max(sig.volatility, 0.01), 0.3)
            leverage = min(max_lev * vol_adj, max_lev)

            # Determine reason
            reasons = []
            if sig.direction_best != 0:
                reasons.append("long" if sig.direction_best == 1 else "short")
            reasons.append(f"EV={sig.ev_best:.4f}")
            reasons.append(f"conf={sig.confidence:.2f}")
            if sig.regime == 0:
                reasons.append("surge")
            elif sig.regime == 1:
                reasons.append("dump")
            elif sig.regime == 3:
                reasons.append("volatile")

            decisions.append(TradeDecision(
                coin=sig.coin,
                direction=sig.direction_best,
                strategy_idx=sig.strategy_idx,
                size_pct=size,
                leverage=leverage,
                confidence=score,
                reason=", ".join(reasons),
            ))

            selected_coins.add(sig.coin)
            selected_directions.append(sig.direction_best)

        return decisions

    def _update_history(self, sig: SignalSnapshot):
        """Track signal persistence."""
        if sig.coin not in self._signal_history:
            self._signal_history[sig.coin] = deque(maxlen=20)
            self._ev_history[sig.coin] = deque(maxlen=500)

        self._signal_history[sig.coin].append({
            "direction": sig.direction_best,
            "strategy_idx": sig.strategy_idx,
            "ev": sig.ev_best,
        })
        self._ev_history[sig.coin].append(sig.ev_best)

    def _check_persistence(self, coin: str) -> bool:
        """Check if signal has persisted for required bars."""
        history = self._signal_history.get(coin, [])
        if len(history) < self.persistence_bars:
            return False

        # Check if last N signals agree on direction
        recent = list(history)[-self.persistence_bars:]
        directions = [h["direction"] for h in recent]
        return len(set(directions)) == 1  # all same direction

    def _get_persistence_length(self, coin: str) -> int:
        """How many consecutive bars has the current signal persisted?"""
        history = self._signal_history.get(coin, [])
        if not history:
            return 0

        current_dir = history[-1]["direction"]
        count = 0
        for h in reversed(history):
            if h["direction"] == current_dir:
                count += 1
            else:
                break
        return count

    def get_ev_percentile(self, coin: str, ev: float) -> float:
        """Get percentile of this EV relative to coin's history."""
        history = self._ev_history.get(coin, [])
        if len(history) < 50:
            return 0.5  # unknown
        arr = np.array(history)
        return (arr < ev).mean()


class NeuralMetaModel(nn.Module):
    """Learnable meta-model (future upgrade).

    Trained on historical base-model signals + actual trade outcomes.
    Learns WHEN the base model is reliable vs not.

    Input: signal matrix (N_coins × feature_per_coin) + portfolio state
    Output: trade decision scores per coin
    """

    def __init__(
        self,
        n_signal_features: int = 8,   # prob, ev, confidence, regime, vol, etc.
        max_coins: int = 250,
        hidden_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()

        # Per-coin signal encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(n_signal_features, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Cross-coin attention — learn which coins' signals inform each other
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Portfolio state encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(6, 32),  # equity, dd, n_positions, recent_pnl_mean, recent_pnl_std, vol
            nn.GELU(),
        )

        # Decision head: per-coin trade score
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),  # score: how good is trading this coin now?
        )

    def forward(
        self,
        signal_matrix: torch.Tensor,  # (B, N_coins, N_features)
        coin_mask: torch.Tensor,       # (B, N_coins) — 1 if coin has signal
        portfolio: torch.Tensor,       # (B, 6)
    ) -> torch.Tensor:
        """Returns per-coin trade scores (B, N_coins)."""
        # Encode each coin's signal
        encoded = self.signal_encoder(signal_matrix)  # (B, N_coins, hidden)

        # Cross-coin attention with masking
        key_padding_mask = ~coin_mask.bool()
        attended, _ = self.cross_attention(
            encoded, encoded, encoded,
            key_padding_mask=key_padding_mask,
        )

        # Portfolio context
        port_enc = self.portfolio_encoder(portfolio)  # (B, 32)
        port_expanded = port_enc.unsqueeze(1).expand(-1, signal_matrix.size(1), -1)

        # Decision per coin
        combined = torch.cat([attended, port_expanded], dim=-1)
        scores = self.decision_head(combined).squeeze(-1)  # (B, N_coins)

        # Mask out coins without signals
        scores = scores * coin_mask + (-1e9) * (1 - coin_mask)

        return scores
