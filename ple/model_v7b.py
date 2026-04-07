"""
PLE v7b: v4 검증된 구조 + coin embedding만 추가

v7의 실수: adaptive expert, 3층 fusion, temporal CNN 등 동시에 너무 많이 바꿈 → 성능 붕괴
v7b: v4를 그대로 유지하고 coin embedding만 추가 (범용모델용)

검증된 v4 하이퍼파라미터: expert_hidden=128, expert_output=96, fusion_dim=192
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ple.model_v4 import PLEv4


class PLEv7b(PLEv4):
    """v4 + coin embedding. 나머지 전부 동일."""

    def __init__(
        self,
        feature_partitions: dict[str, list[int]],
        n_coins: int = 250,
        coin_embed_dim: int = 16,
        n_account_features: int = 4,
        n_strategies: int = 32,
        expert_hidden: int = 128,
        expert_output: int = 96,
        fusion_dim: int = 192,
        dropout: float = 0.2,
        use_vsn: bool = False,
    ):
        # v4 init with expanded account dim (account + coin_embed)
        super().__init__(
            feature_partitions=feature_partitions,
            n_account_features=n_account_features + coin_embed_dim,
            n_strategies=n_strategies,
            expert_hidden=expert_hidden,
            expert_output=expert_output,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_vsn=use_vsn,
        )

        self.coin_embed = nn.Embedding(n_coins, coin_embed_dim)
        self.coin_embed_dim = coin_embed_dim
        self.n_account_base = n_account_features

    def forward(self, features: torch.Tensor, account: torch.Tensor,
                coin_id: torch.Tensor = None) -> dict:
        # Append coin embedding to account
        if coin_id is not None:
            coin_emb = self.coin_embed(coin_id)
            account_full = torch.cat([account, coin_emb], dim=-1)
        else:
            # No coin_id → pad with zeros (backward compatible with v4)
            pad = torch.zeros(account.shape[0], self.coin_embed_dim,
                              device=account.device)
            account_full = torch.cat([account, pad], dim=-1)

        return super().forward(features, account_full)
