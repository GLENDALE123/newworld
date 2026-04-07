"""
PLE v7: Cleaned Architecture — 1D CNN Temporal + Adaptive Experts

Changes from v4:
  1. Expert 크기를 partition 크기에 비례하여 조정
  2. 1D CNN temporal encoder — 최근 32 bars 시계열 패턴 학습
  3. Coin embedding — 범용모델용
  4. Confidence head 제거 — probs.max()로 대체
  5. Fusion layer 보강 (3층)
  6. MoE 없음, regime embedding 없음 (메타모델이 담당)

v4에서 검증된 것 유지:
  - Feature-partitioned experts
  - Attention-gated fusion
  - Multi-label sigmoid (32 strategies)
  - MAE/MFE regression heads
  - R-Drop 호환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    """Single timeframe 1D CNN branch."""

    def __init__(self, n_channels: int, embed_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_channels)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.pool(x).squeeze(-1)  # (B, embed_dim)


class MultiScaleTemporalCNN(nn.Module):
    """Multi-timeframe 1D CNN encoder.

    Three scales:
      5m  × 12 bars = 1시간 (단타 패턴)
      15m × 32 bars = 8시간 (중기 흐름)
      1h  × 24 bars = 24시간 (일간 맥락)

    Each scale has its own CNN branch → concat → project.
    ONNX compatible.
    """

    def __init__(self, n_channels: int = 6, output_dim: int = 64, branch_dim: int = 32):
        super().__init__()
        self.branch_5m = TemporalBranch(n_channels, branch_dim)
        self.branch_15m = TemporalBranch(n_channels, branch_dim)
        self.branch_1h = TemporalBranch(n_channels, branch_dim)
        self.norm = nn.LayerNorm(branch_dim * 3)
        self.proj = nn.Linear(branch_dim * 3, output_dim)

    def forward(
        self,
        seq_5m: torch.Tensor,   # (B, 12, 6)
        seq_15m: torch.Tensor,  # (B, 32, 6)
        seq_1h: torch.Tensor,   # (B, 24, 6)
    ) -> torch.Tensor:
        e5 = self.branch_5m(seq_5m)
        e15 = self.branch_15m(seq_15m)
        e1h = self.branch_1h(seq_1h)
        combined = torch.cat([e5, e15, e1h], dim=-1)
        return self.proj(self.norm(combined))  # (B, output_dim)


class AdaptiveExpert(nn.Module):
    """Expert with size proportional to input dimension."""

    def __init__(self, input_dim: int, hidden_mult: float = 2.0,
                 output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        hidden = max(32, int(input_dim * hidden_mult))
        hidden = min(hidden, 512)  # cap

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


class CoinEmbedding(nn.Module):
    """Learned embedding per coin for universal model."""
    def __init__(self, n_coins: int = 250, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(n_coins, embed_dim)

    def forward(self, coin_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(coin_id)


class PLEv7(nn.Module):
    """PLE v7: Clean architecture for multi-coin multi-strategy trading.

    Input:
      - features: (B, F) point-in-time features
      - temporal: (B, seq_len, n_channels) recent bar sequence
      - coin_id: (B,) coin identifier
      - account: (B, A) account state

    Output:
      - label_probs: (B, N_strategies) P(profitable) per strategy
      - mae_pred: (B, N_strategies) predicted MAE
      - mfe_pred: (B, N_strategies) predicted MFE
      - gate_weights: (B, n_experts) for interpretability
    """

    def __init__(
        self,
        feature_partitions: dict[str, list[int]],
        n_coins: int = 250,
        coin_embed_dim: int = 16,
        n_account_features: int = 4,
        n_strategies: int = 32,
        expert_output: int = 64,
        temporal_channels: int = 6,  # OHLCV + returns
        temporal_dim: int = 64,
        fusion_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_partitions = feature_partitions
        self.n_strategies = n_strategies

        # Register partition indices as buffers (ONNX compatible)
        self._partition_indices = {}
        for name, indices in feature_partitions.items():
            buf = torch.tensor(indices, dtype=torch.long)
            self.register_buffer(f"_idx_{name}", buf)
            self._partition_indices[name] = buf

        # Adaptive experts — size scales with partition size
        self.experts = nn.ModuleDict()
        for name, indices in feature_partitions.items():
            n_feat = len(indices)
            # Larger partitions get wider hidden layers
            self.experts[name] = AdaptiveExpert(
                input_dim=n_feat,
                hidden_mult=2.0 if n_feat > 50 else 3.0,
                output_dim=expert_output,
                dropout=dropout,
            )

        # Multi-scale temporal CNN encoder (5m/15m/1h)
        self.temporal_cnn = MultiScaleTemporalCNN(
            n_channels=temporal_channels,
            output_dim=temporal_dim,
        )

        # Coin embedding
        self.coin_embed = CoinEmbedding(n_coins, coin_embed_dim)

        # Account encoder
        context_dim = n_account_features + coin_embed_dim + temporal_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
        )

        # Attention gate
        n_experts = len(feature_partitions)
        gate_input = n_experts * expert_output + 64  # experts + context
        self.gate_query = nn.Linear(gate_input, expert_output)
        self.gate_keys = nn.ModuleDict({
            name: nn.Linear(expert_output, expert_output)
            for name in feature_partitions
        })

        # Fusion — deeper than v4 (3 layers instead of 2)
        fusion_input = expert_output + 64  # gated features + context
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

        # Multi-label strategy head
        self.label_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_strategies),
        )

        # MAE/MFE regression heads
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

        # No confidence head — use label_probs.max() instead

    def forward(
        self,
        features: torch.Tensor,     # (B, F) point-in-time
        seq_5m: torch.Tensor,       # (B, 12, 6) recent 5m bars
        seq_15m: torch.Tensor,      # (B, 32, 6) recent 15m bars
        seq_1h: torch.Tensor,       # (B, 24, 6) recent 1h bars
        coin_id: torch.Tensor,      # (B,)
        account: torch.Tensor,      # (B, A)
    ) -> dict:
        # Expert outputs on partitioned features
        expert_outs = {}
        for name in self.feature_partitions:
            idx = getattr(self, f"_idx_{name}")
            expert_outs[name] = self.experts[name](features[:, idx])

        # Multi-scale temporal encoding
        temporal_enc = self.temporal_cnn(seq_5m, seq_15m, seq_1h)

        # Context
        coin_emb = self.coin_embed(coin_id)
        context_input = torch.cat([account, coin_emb, temporal_enc], dim=-1)
        context_enc = self.context_encoder(context_input)

        # Attention-gated fusion
        expert_list = list(expert_outs.values())
        expert_names = list(expert_outs.keys())
        expert_stacked = torch.stack(expert_list, dim=1)  # (B, n_exp, dim)

        all_cat = torch.cat(expert_list + [context_enc], dim=-1)
        q = self.gate_query(all_cat)

        scores = []
        for name in expert_names:
            k = self.gate_keys[name](expert_outs[name])
            scores.append((q * k).sum(dim=-1, keepdim=True))

        gate_scores = torch.cat(scores, dim=-1)
        gate_weights = F.softmax(gate_scores / (q.shape[-1] ** 0.5), dim=-1)
        gated = torch.bmm(gate_weights.unsqueeze(1), expert_stacked).squeeze(1)

        # Fusion
        fused = self.fusion(torch.cat([gated, context_enc], dim=-1))

        # Outputs
        label_logits = self.label_head(fused)
        label_probs = torch.sigmoid(label_logits)

        return {
            "label_logits": label_logits,
            "label_probs": label_probs,
            "mae_pred": self.mae_head(fused),
            "mfe_pred": self.mfe_head(fused),
            "gate_weights": gate_weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
