"""
PLE v5: Universal Multi-Coin Model

Key change from v4: ONE model trades ALL coins.
- Coin embedding: learned representation per coin
- Same features → same model → different coins
- Scales to 200+ coins without separate models

For low-capital aggressive strategy:
- $500 start with 20x leverage
- Gradually reduce leverage as capital grows
- Kelly-optimal sizing per signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoinEmbedding(nn.Module):
    """Learned embedding per coin (like word embedding for NLP)."""
    def __init__(self, n_coins: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(n_coins, embed_dim)

    def forward(self, coin_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(coin_id)


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, output_dim), nn.LayerNorm(output_dim),
        )
    def forward(self, x): return self.net(x)


class PLEv5Universal(nn.Module):
    """Universal PLE: one model for all coins.

    Input: features (B, F) + coin_id (B,) + account_state (B, A)
    Output: strategy probabilities + MAE/MFE predictions
    """

    def __init__(
        self,
        feature_partitions: dict[str, list[int]],
        n_coins: int = 200,
        coin_embed_dim: int = 32,
        n_account_features: int = 6,  # equity, dd, leverage, n_positions, pnl, hold_time
        n_strategies: int = 32,
        expert_hidden: int = 256,
        expert_output: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_partitions = feature_partitions
        self.n_strategies = n_strategies

        # Coin embedding
        self.coin_embed = CoinEmbedding(n_coins, coin_embed_dim)

        # Feature experts
        self.experts = nn.ModuleDict()
        for name, indices in feature_partitions.items():
            self.experts[name] = Expert(len(indices), expert_hidden, expert_output, dropout)

        # Account + coin encoder
        account_input = n_account_features + coin_embed_dim
        self.account_encoder = nn.Sequential(
            nn.Linear(account_input, 64), nn.GELU(),
        )

        # Gate
        n_experts = len(feature_partitions)
        gate_input = n_experts * expert_output + 64
        self.gate_query = nn.Linear(gate_input, expert_output)
        self.gate_keys = nn.ModuleDict({
            name: nn.Linear(expert_output, expert_output)
            for name in feature_partitions
        })

        # Fusion
        fusion_input = expert_output + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim), nn.GELU(), nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.LayerNorm(fusion_dim),
        )

        # Heads
        self.label_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_strategies),
        )
        self.mae_head = nn.Sequential(nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, n_strategies))
        self.mfe_head = nn.Sequential(nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, n_strategies))
        self.conf_head = nn.Sequential(nn.Linear(fusion_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

        # Leverage recommendation head (new for v5)
        self.leverage_head = nn.Sequential(
            nn.Linear(fusion_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),  # outputs 0-1, scaled to leverage range
        )

    def forward(self, features, coin_id, account):
        # Coin embedding
        coin_emb = self.coin_embed(coin_id)

        # Expert outputs
        expert_outs = {}
        for name, indices in self.feature_partitions.items():
            idx = torch.tensor(indices, device=features.device)
            expert_outs[name] = self.experts[name](features[:, idx])

        # Account + coin encoding
        account_coin = torch.cat([account, coin_emb], dim=-1)
        account_enc = self.account_encoder(account_coin)

        # Gated fusion
        expert_list = list(expert_outs.values())
        expert_names = list(expert_outs.keys())
        expert_stacked = torch.stack(expert_list, dim=1)

        all_cat = torch.cat(expert_list + [account_enc], dim=-1)
        q = self.gate_query(all_cat)
        scores = []
        for name in expert_names:
            k = self.gate_keys[name](expert_outs[name])
            scores.append((q * k).sum(dim=-1, keepdim=True))

        gate_scores = torch.cat(scores, dim=-1)
        gate_weights = F.softmax(gate_scores / (q.shape[-1] ** 0.5), dim=-1)
        gated = torch.bmm(gate_weights.unsqueeze(1), expert_stacked).squeeze(1)

        fused = self.fusion(torch.cat([gated, account_enc], dim=-1))

        label_logits = self.label_head(fused)
        label_probs = torch.sigmoid(label_logits)

        return {
            "label_logits": label_logits,
            "label_probs": label_probs,
            "mae_pred": self.mae_head(fused),
            "mfe_pred": self.mfe_head(fused),
            "confidence": self.conf_head(fused).squeeze(-1),
            "leverage_rec": self.leverage_head(fused).squeeze(-1),  # 0-1 → scale to 1x-20x
            "gate_weights": gate_weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
