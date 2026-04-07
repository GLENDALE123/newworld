"""
PLE v6: Regime-Aware Mixture of Experts

Key improvements over v4/v5:
  1. Regime embedding — explicit market state input
  2. Sparse MoE routing — different experts for different regimes
  3. Strategy-grouped heads — separate output groups for scalp/intra/day/swing
  4. Coin embedding (from v5) — universal multi-coin capability
  5. Temporal micro-encoder — encodes recent bar statistics

Architecture:
  Input: features(F) + coin_id + regime_id + account_state + temporal_ctx

  Feature experts (partitioned, same as v4)
  → MoE Router (regime-conditioned, selects top-k experts)
  → Attention-gated fusion
  → Strategy group heads (4 groups × 8 labels each = 32 labels)
  → MAE/MFE regression heads
  → Leverage recommendation

"시장국면에 따라 다른 전문가가 판단한다."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegimeEmbedding(nn.Module):
    """Learned embedding for market regime."""
    REGIME_MAP = {"surge": 0, "dump": 1, "range": 2, "volatile": 3}

    def __init__(self, n_regimes: int = 4, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(n_regimes, embed_dim)
        self.n_regimes = n_regimes

    def forward(self, regime_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(regime_id)


class CoinEmbedding(nn.Module):
    """Learned embedding per coin."""
    def __init__(self, n_coins: int = 250, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(n_coins, embed_dim)

    def forward(self, coin_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(coin_id)


class Expert(nn.Module):
    """Single expert network with residual connection."""
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)
        # Residual projection if dimensions differ
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.norm(self.net(x) + self.residual(x))


class MoERouter(nn.Module):
    """Sparse Mixture of Experts router.

    Selects top-k experts per sample, conditioned on regime + features.
    Inspired by Switch Transformer / Mixtral routing.
    """
    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        top_k: int = 2,
        regime_dim: int = 16,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(input_dim + regime_dim, n_experts * 2),
            nn.GELU(),
            nn.Linear(n_experts * 2, n_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        regime_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (routing_weights, expert_indices) both shape (B, top_k)."""
        gate_input = torch.cat([x, regime_emb], dim=-1)
        logits = self.gate(gate_input)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        return topk_weights, topk_idx


class TemporalEncoder(nn.Module):
    """Lightweight encoder for temporal context features.

    Takes recent bar statistics and produces a compact representation.
    """
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyGroupHead(nn.Module):
    """Output head for a group of related strategies.

    Each group covers one strategy type (scalp/intra/day/swing)
    with labels for each direction × regime combination.
    """
    def __init__(self, input_dim: int, n_labels: int = 8):
        super().__init__()
        # n_labels = 2 directions × 4 regimes = 8
        self.label_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, n_labels),
        )
        self.mae_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_labels),
        )
        self.mfe_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_labels),
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {
            "logits": self.label_head(x),
            "mae": self.mae_head(x),
            "mfe": self.mfe_head(x),
        }


class PLEv6(nn.Module):
    """PLE v6: Regime-Aware Mixture of Experts.

    Improvements:
    1. Regime embedding conditions the routing
    2. MoE selects top-k feature experts per sample (sparse activation)
    3. Grouped strategy heads maintain structure
    4. Temporal encoder adds recent-bar awareness
    5. Coin embedding enables universal operation
    """

    def __init__(
        self,
        feature_partitions: dict[str, list[int]],
        n_coins: int = 250,
        coin_embed_dim: int = 32,
        regime_embed_dim: int = 16,
        n_account_features: int = 4,
        n_temporal_features: int = 40,    # from temporal_context.py
        n_strategy_groups: int = 4,       # scalp, intra, day, swing
        n_labels_per_group: int = 8,      # 2 dirs × 4 regimes
        expert_hidden: int = 256,
        expert_output: int = 128,
        n_moe_experts: int = 8,           # separate MoE experts for routing
        moe_top_k: int = 2,
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_partitions = feature_partitions
        self.n_strategy_groups = n_strategy_groups
        self.n_labels_per_group = n_labels_per_group
        self.n_labels = n_strategy_groups * n_labels_per_group

        # Embeddings
        self.coin_embed = CoinEmbedding(n_coins, coin_embed_dim)
        self.regime_embed = RegimeEmbedding(4, regime_embed_dim)

        # Feature-partitioned experts (same as v4)
        self.feature_experts = nn.ModuleDict()
        for name, indices in feature_partitions.items():
            self.feature_experts[name] = Expert(len(indices), expert_hidden, expert_output, dropout)

        # MoE experts — additional experts for regime-conditioned routing
        self.moe_experts = nn.ModuleList([
            Expert(expert_output, expert_hidden, expert_output, dropout)
            for _ in range(n_moe_experts)
        ])
        self.moe_router = MoERouter(
            input_dim=expert_output,
            n_experts=n_moe_experts,
            top_k=moe_top_k,
            regime_dim=regime_embed_dim,
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(n_temporal_features, 32)

        # Account encoder (includes coin + regime embeddings)
        context_dim = n_account_features + coin_embed_dim + regime_embed_dim + 32  # +temporal
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
        )

        # Attention gate for feature experts
        n_feat_experts = len(feature_partitions)
        gate_input = n_feat_experts * expert_output + 64  # feature experts + context
        self.gate_query = nn.Linear(gate_input, expert_output)
        self.gate_keys = nn.ModuleDict({
            name: nn.Linear(expert_output, expert_output)
            for name in feature_partitions
        })

        # Fusion
        fusion_input = expert_output + 64  # gated features + context
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

        # Strategy group heads
        self.strategy_heads = nn.ModuleList([
            StrategyGroupHead(fusion_dim, n_labels_per_group)
            for _ in range(n_strategy_groups)
        ])

        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Leverage recommendation
        self.leverage_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,      # (B, F)
        coin_id: torch.Tensor,        # (B,)
        regime_id: torch.Tensor,      # (B,)
        account: torch.Tensor,        # (B, A)
        temporal: torch.Tensor,       # (B, T)
    ) -> dict:
        B = features.shape[0]

        # Embeddings
        coin_emb = self.coin_embed(coin_id)
        regime_emb = self.regime_embed(regime_id)
        temporal_enc = self.temporal_encoder(temporal)

        # Feature-partitioned experts
        feat_expert_outs = {}
        for name, indices in self.feature_partitions.items():
            idx = torch.tensor(indices, dtype=torch.long, device=features.device)
            feat_expert_outs[name] = self.feature_experts[name](features[:, idx])

        # Context encoding
        context_input = torch.cat([account, coin_emb, regime_emb, temporal_enc], dim=-1)
        context_enc = self.context_encoder(context_input)

        # Attention-gated fusion of feature experts
        expert_list = list(feat_expert_outs.values())
        expert_names = list(feat_expert_outs.keys())
        expert_stacked = torch.stack(expert_list, dim=1)  # (B, n_exp, dim)

        all_cat = torch.cat(expert_list + [context_enc], dim=-1)
        q = self.gate_query(all_cat)

        scores = []
        for name in expert_names:
            k = self.gate_keys[name](feat_expert_outs[name])
            scores.append((q * k).sum(dim=-1, keepdim=True))

        gate_scores = torch.cat(scores, dim=-1)
        gate_weights = F.softmax(gate_scores / (q.shape[-1] ** 0.5), dim=-1)
        gated_features = torch.bmm(gate_weights.unsqueeze(1), expert_stacked).squeeze(1)

        # MoE: route through regime-conditioned experts
        moe_weights, moe_indices = self.moe_router(gated_features, regime_emb)
        # (B, top_k), (B, top_k)

        # Sparse expert computation
        moe_output = torch.zeros_like(gated_features)
        for k in range(moe_weights.shape[1]):
            expert_idx = moe_indices[:, k]  # (B,)
            weight = moe_weights[:, k:k+1]  # (B, 1)

            # Group by expert for efficiency
            for e in range(len(self.moe_experts)):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = gated_features[mask]
                    expert_out = self.moe_experts[e](expert_input)
                    moe_output[mask] += weight[mask] * expert_out

        # Fusion
        fused = self.fusion(torch.cat([moe_output, context_enc], dim=-1))

        # Strategy group outputs
        all_logits = []
        all_mae = []
        all_mfe = []
        for head in self.strategy_heads:
            out = head(fused)
            all_logits.append(out["logits"])
            all_mae.append(out["mae"])
            all_mfe.append(out["mfe"])

        label_logits = torch.cat(all_logits, dim=-1)  # (B, 32)
        label_probs = torch.sigmoid(label_logits)
        mae_pred = torch.cat(all_mae, dim=-1)
        mfe_pred = torch.cat(all_mfe, dim=-1)

        confidence = self.conf_head(fused).squeeze(-1)
        leverage_rec = self.leverage_head(fused).squeeze(-1)

        return {
            "label_logits": label_logits,     # (B, 32)
            "label_probs": label_probs,       # (B, 32)
            "mae_pred": mae_pred,             # (B, 32)
            "mfe_pred": mfe_pred,             # (B, 32)
            "confidence": confidence,         # (B,)
            "leverage_rec": leverage_rec,     # (B,) 0-1
            "gate_weights": gate_weights,     # (B, n_feat_experts)
            "moe_weights": moe_weights,       # (B, top_k)
            "moe_indices": moe_indices,       # (B, top_k)
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def moe_load_balance_loss(self, moe_indices: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss to encourage balanced expert usage.

        Prevents expert collapse where one expert handles everything.
        From Switch Transformer paper.
        """
        B = moe_indices.shape[0]
        n_experts = len(self.moe_experts)

        # Count how many times each expert is selected
        flat_indices = moe_indices.reshape(-1)
        counts = torch.zeros(n_experts, device=moe_indices.device)
        counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))

        # Ideal: uniform distribution
        ideal = B * moe_indices.shape[1] / n_experts
        # Load balance loss: variance of counts
        loss = ((counts - ideal) ** 2).mean() / (ideal ** 2 + 1e-8)
        return loss
