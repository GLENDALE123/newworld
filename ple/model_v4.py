"""
PLE v4: Multi-Label with Independent Sigmoid Outputs

Key change from v3: softmax (pick one) → sigmoid (multiple simultaneous)
- 128 independent binary classifiers: "is this strategy profitable NOW?"
- Multiple strategies can fire simultaneously (scalp + swing at same time)
- NO_TRADE emerges naturally when all 128 are below threshold
- Feature-partitioned experts preserved from v3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PLEv4(nn.Module):
    """
    Multi-Label PLE: 128 independent sigmoid outputs.

    Each output = P(RAR_net > 0) for that strategy.
    Multiple can be YES simultaneously.
    """

    def __init__(
        self,
        feature_partitions: dict[str, list[int]],
        n_account_features: int = 4,
        n_strategies: int = 128,
        expert_hidden: int = 128,
        expert_output: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_partitions = feature_partitions
        self.n_strategies = n_strategies

        # Feature-partitioned experts
        self.experts = nn.ModuleDict()
        for name, indices in feature_partitions.items():
            self.experts[name] = Expert(len(indices), expert_hidden, expert_output, dropout)

        # Account encoder
        account_dim = 32
        self.account_encoder = nn.Sequential(
            nn.Linear(n_account_features, account_dim),
            nn.GELU(),
        )

        # Gate: attention over experts
        n_experts = len(feature_partitions)
        gate_input = n_experts * expert_output + account_dim
        self.gate_query = nn.Linear(gate_input, expert_output)
        self.gate_keys = nn.ModuleDict({
            name: nn.Linear(expert_output, expert_output)
            for name in feature_partitions
        })

        # Fusion
        fusion_input = expert_output + account_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

        # Multi-label head: 128 independent sigmoids
        # "Is strategy i profitable (after fees) right now?"
        self.label_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_strategies),
        )

        # MAE/MFE regression heads (shared across strategies, conditioned on fusion)
        self.mae_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_strategies),
        )
        self.mfe_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_strategies),
        )

        # Confidence: overall "is there any good trade right now?"
        self.conf_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, account: torch.Tensor) -> dict:
        # Expert outputs on partitioned features
        expert_outs = {}
        for name, indices in self.feature_partitions.items():
            idx = torch.tensor(indices, device=features.device)
            expert_outs[name] = self.experts[name](features[:, idx])

        account_enc = self.account_encoder(account)

        expert_list = list(expert_outs.values())
        expert_names = list(expert_outs.keys())
        expert_stacked = torch.stack(expert_list, dim=1)  # (B, n_exp, dim)

        # Gated fusion
        all_cat = torch.cat(expert_list + [account_enc], dim=-1)
        q = self.gate_query(all_cat)

        scores = []
        for name in expert_names:
            k = self.gate_keys[name](expert_outs[name])
            s = (q * k).sum(dim=-1, keepdim=True)
            scores.append(s)

        gate_scores = torch.cat(scores, dim=-1)
        gate_weights = F.softmax(gate_scores / (q.shape[-1] ** 0.5), dim=-1)
        gated = torch.bmm(gate_weights.unsqueeze(1), expert_stacked).squeeze(1)

        fused = self.fusion(torch.cat([gated, account_enc], dim=-1))

        # Multi-label output (sigmoid, NOT softmax)
        label_logits = self.label_head(fused)
        label_probs = torch.sigmoid(label_logits)  # (B, 128) each independent

        mae_pred = self.mae_head(fused)
        mfe_pred = self.mfe_head(fused)
        confidence = self.conf_head(fused).squeeze(-1)

        return {
            "label_logits": label_logits,   # (B, 128) raw
            "label_probs": label_probs,     # (B, 128) P(profitable) per strategy
            "mae_pred": mae_pred,           # (B, 128)
            "mfe_pred": mfe_pred,           # (B, 128)
            "confidence": confidence,       # (B,)
            "gate_weights": gate_weights,   # (B, n_experts)
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
