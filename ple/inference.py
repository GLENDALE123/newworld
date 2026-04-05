"""
PLE Inference Engine: Real-time strategy selection.

Given current features, outputs the optimal strategy with highest EV.
"""

import numpy as np
import torch
import pandas as pd

from ple.model import PLETradingModel
from labeling.multi_tbm import TIMEFRAMES, RR_RATIOS, REGIMES


# Build label names in consistent order
LABEL_NAMES = []
for tf in sorted(TIMEFRAMES.keys()):
    for rr in sorted(RR_RATIOS.keys()):
        for regime in REGIMES:
            LABEL_NAMES.append(f"{tf}_{rr}_{regime}")


def compute_ev(
    tbm_prob: np.ndarray,   # (100,) P(take profit)
    mae_pred: np.ndarray,   # (100,) predicted max adverse excursion
    mfe_pred: np.ndarray,   # (100,) predicted max favorable excursion
    rar_pred: np.ndarray,   # (100,) predicted risk-adjusted return
) -> pd.DataFrame:
    """Compute Expected Value for each of the 100 label strategies.

    EV = P(win) * estimated_reward - P(loss) * estimated_risk

    Returns DataFrame with columns: label, p_win, reward, risk, ev, rar
    """
    p_win = tbm_prob
    p_loss = 1 - p_win

    # Reward: use MFE as estimated upside
    reward = np.abs(mfe_pred)
    # Risk: use MAE as estimated downside
    risk = np.abs(mae_pred)

    ev = p_win * reward - p_loss * risk

    results = pd.DataFrame({
        "label": LABEL_NAMES[:len(tbm_prob)],
        "p_win": p_win,
        "reward": reward,
        "risk": risk,
        "ev": ev,
        "rar": rar_pred,
    })

    return results.sort_values("ev", ascending=False)


class PLEInferenceEngine:
    """Real-time inference: features in → strategy decision out."""

    def __init__(self, model: PLETradingModel, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Run inference on a single observation or batch.

        Args:
            features: (n_features,) or (batch, n_features)

        Returns:
            dict with tbm_probs, mae_pred, mfe_pred, rar_pred as numpy arrays
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        outputs = self.model(x)

        return {k: v.cpu().numpy() for k, v in outputs.items()}

    def select_strategy(
        self,
        features: np.ndarray,
        min_ev: float = 0.0001,
        min_p_win: float = 0.52,
        max_risk: float = 0.05,
    ) -> dict | None:
        """Select the best strategy for the current market state.

        Args:
            features: (n_features,) current feature vector
            min_ev: minimum EV threshold to enter
            min_p_win: minimum win probability
            max_risk: maximum acceptable risk (MAE)

        Returns:
            dict with strategy details, or None if no good opportunity
        """
        preds = self.predict(features)

        ev_df = compute_ev(
            preds["tbm_probs"][0],
            preds["mae_pred"][0],
            preds["mfe_pred"][0],
            preds["rar_pred"][0],
        )

        # Filter by constraints
        candidates = ev_df[
            (ev_df["ev"] > min_ev) &
            (ev_df["p_win"] > min_p_win) &
            (ev_df["risk"] < max_risk)
        ]

        if candidates.empty:
            return None

        best = candidates.iloc[0]
        label = best["label"]

        # Parse label to get strategy parameters
        parts = label.split("_")
        tf = parts[0]
        rr = parts[1]
        regime = parts[2]

        rr_cfg = RR_RATIOS[rr]
        tf_cfg = TIMEFRAMES[tf]

        return {
            "label": label,
            "timeframe": tf,
            "rr_type": rr,
            "regime": regime,
            "p_win": float(best["p_win"]),
            "ev": float(best["ev"]),
            "expected_reward": float(best["reward"]),
            "expected_risk": float(best["risk"]),
            "rar": float(best["rar"]),
            "holding_minutes": tf_cfg["minutes"],
            "rr_config": rr_cfg,
            # Position sizing: Kelly fraction
            "kelly_fraction": _kelly(best["p_win"], best["reward"], best["risk"]),
        }

    def evaluate_position(
        self,
        features: np.ndarray,
        current_label: str,
        entry_price: float,
        current_price: float,
    ) -> dict:
        """Re-evaluate an open position. Should we hold or exit?

        Returns dict with updated probabilities and exit recommendation.
        """
        preds = self.predict(features)
        ev_df = compute_ev(
            preds["tbm_probs"][0],
            preds["mae_pred"][0],
            preds["mfe_pred"][0],
            preds["rar_pred"][0],
        )

        current_ev = ev_df[ev_df["label"] == current_label]
        if current_ev.empty:
            return {"action": "EXIT", "reason": "label not in predictions"}

        ev = float(current_ev.iloc[0]["ev"])
        p_win = float(current_ev.iloc[0]["p_win"])

        unrealized_pnl = (current_price - entry_price) / entry_price

        # Exit conditions
        if ev < 0:
            return {"action": "EXIT", "reason": f"EV turned negative: {ev:.6f}", "ev": ev}
        if p_win < 0.45:
            return {"action": "EXIT", "reason": f"P(win) dropped: {p_win:.3f}", "p_win": p_win}

        return {
            "action": "HOLD",
            "ev": ev,
            "p_win": p_win,
            "unrealized_pnl": unrealized_pnl,
        }


def _kelly(p_win: float, reward: float, risk: float) -> float:
    """Half-Kelly fraction: f = 0.5 * (p*b - q) / b where b = reward/risk."""
    if risk <= 0 or reward <= 0:
        return 0.0
    b = reward / risk
    q = 1 - p_win
    f = (p_win * b - q) / b
    return max(0.0, min(0.5 * f, 0.25))  # Half-Kelly, capped at 25%
