"""Lightweight scalping MLP — 31 features → probabilities + MFE/MAE + expected bars."""

import torch
import torch.nn as nn


class ScalpingMLP(nn.Module):
    """Multi-output MLP for scalping signals.

    Outputs per direction (long/short):
    - P(win): probability of profitable trade (sigmoid)
    - MFE: expected max favorable excursion (linear)
    - MAE: expected max adverse excursion (linear, positive = bad)
    - bars: expected bars to barrier hit (linear)
    """

    def __init__(self, n_features: int = 31, hidden: int = 64, dropout: float = 0.2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout),
        )

        dim = hidden // 2

        # Probability heads (sigmoid)
        self.prob_head = nn.Linear(dim, 2)     # [P(long_win), P(short_win)]

        # MFE heads (positive, how much we can gain)
        self.mfe_head = nn.Linear(dim, 2)      # [long_mfe, short_mfe]

        # MAE heads (positive, how much we can lose)
        self.mae_head = nn.Linear(dim, 2)      # [long_mae, short_mae]

        # Expected bars to barrier
        self.bars_head = nn.Linear(dim, 2)     # [long_bars, short_bars]

    def forward(self, x: torch.Tensor) -> dict:
        h = self.backbone(x)

        probs = torch.sigmoid(self.prob_head(h))
        mfe = torch.relu(self.mfe_head(h))     # MFE is always positive
        mae = torch.relu(self.mae_head(h))     # MAE is always positive
        bars = torch.relu(self.bars_head(h))   # bars is always positive

        return {
            "prob_long": probs[:, 0],
            "prob_short": probs[:, 1],
            "mfe_long": mfe[:, 0],
            "mfe_short": mfe[:, 1],
            "mae_long": mae[:, 0],
            "mae_short": mae[:, 1],
            "bars_long": bars[:, 0],
            "bars_short": bars[:, 1],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
