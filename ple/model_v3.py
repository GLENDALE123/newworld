"""
PLE v3: Hierarchical Selection + Pre-trained Experts + Feature Partitioning

Fixes v2's structural problems:
1. Hierarchical selection (4+5+5=14 way) instead of flat 100-way softmax
2. Experts receive different feature subsets (natural differentiation)
3. Pre-trained experts (frozen) with trainable gate
4. Account state injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Feature Partitioning ─────────────────────────────────────────────────────

def partition_features(feature_names: list[str]) -> dict[str, list[int]]:
    """Partition feature indices by data source.

    Returns dict mapping expert name -> list of feature column indices.
    """
    partitions = {
        "price": [],    # price temporal features (includes regime signals)
        "volume": [],   # volume/trade features
        "metrics": [],  # OI, funding, LS ratio
        "cross": [],    # cross-feature combinations
    }

    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(k in name_lower for k in ["_x_", "_corr_", "_div_chg_", "_div_"]):
            partitions["cross"].append(i)
        elif any(k in name_lower for k in ["buy_ratio", "cvd", "vol_surge", "trades_surge", "sell"]):
            partitions["volume"].append(i)
        elif any(k in name_lower for k in ["open_interest", "taker", "long_short", "toptrader", "funding"]):
            partitions["metrics"].append(i)
        else:
            partitions["price"].append(i)

    return partitions


# ── Expert Network ───────────────────────────────────────────────────────────

class SpecializedExpert(nn.Module):
    """Expert that operates on a specific feature subset."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Hierarchical Selector ───────────────────────────────────────────────────

class HierarchicalSelector(nn.Module):
    """Three-stage hierarchical strategy selection.

    Stage 1: Regime (4-way: surge, dump, range, volatile)
    Stage 2: Timeframe (5-way: 3m, 5m, 15m, 1h, 4h) — conditioned on regime
    Stage 3: Risk/Reward (5-way: 1:1, 1:2, 2:1, tight, wide) — conditioned on regime+tf

    Total: 4×5×5 = 100 strategies, but only 14 classification heads.
    """

    DIRECTIONS = ["long", "short"]
    REGIMES = ["dump", "range", "surge", "volatile"]
    TIMEFRAMES = ["15m", "1h", "4h", "5m"]
    RR_TYPES = ["1to1", "1to2", "2to1", "wide"]

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        # Stage 0: Direction (2-way)
        self.dir_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2),
        )

        # Stage 1: Regime classifier (conditioned on direction)
        self.dir_embed = nn.Embedding(2, dim // 4)
        self.regime_head = nn.Sequential(
            nn.Linear(dim + dim // 4, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 4),
        )

        # Stage 2: Timeframe selector
        self.regime_embed = nn.Embedding(4, dim // 4)
        self.tf_head = nn.Sequential(
            nn.Linear(dim + dim // 4 + dim // 4, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 4),
        )

        # Stage 3: RR selector
        self.tf_embed = nn.Embedding(4, dim // 4)
        self.rr_head = nn.Sequential(
            nn.Linear(dim + dim // 4 + dim // 4 + dim // 4, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 4),
        )

    def forward(self, h: torch.Tensor) -> dict:
        """
        Returns logits/probs for each hierarchical stage + composite strategy_idx.
        strategy_idx maps to: dir * 64 + regime * 16 + tf * 4 + rr
        Total: 2 * 4 * 4 * 4 = 128 strategies
        """
        # Stage 0: Direction
        dir_logits = self.dir_head(h)
        dir_probs = F.softmax(dir_logits, dim=-1)
        dir_idx = dir_probs.argmax(dim=-1)

        # Stage 1: Regime (conditioned on direction)
        d_emb = self.dir_embed(dir_idx)
        regime_logits = self.regime_head(torch.cat([h, d_emb], dim=-1))
        regime_probs = F.softmax(regime_logits, dim=-1)
        regime_idx = regime_probs.argmax(dim=-1)

        # Stage 2: Timeframe
        r_emb = self.regime_embed(regime_idx)
        tf_logits = self.tf_head(torch.cat([h, d_emb, r_emb], dim=-1))
        tf_probs = F.softmax(tf_logits, dim=-1)
        tf_idx = tf_probs.argmax(dim=-1)

        # Stage 3: Risk/Reward
        t_emb = self.tf_embed(tf_idx)
        rr_logits = self.rr_head(torch.cat([h, d_emb, r_emb, t_emb], dim=-1))
        rr_probs = F.softmax(rr_logits, dim=-1)
        rr_idx = rr_probs.argmax(dim=-1)

        # Composite: dir * 64 + regime * 16 + tf * 4 + rr
        strategy_idx = dir_idx * 64 + regime_idx * 16 + tf_idx * 4 + rr_idx

        return {
            "dir_logits": dir_logits, "dir_probs": dir_probs,
            "regime_logits": regime_logits, "regime_probs": regime_probs,
            "tf_logits": tf_logits, "tf_probs": tf_probs,
            "rr_logits": rr_logits, "rr_probs": rr_probs,
            "strategy_idx": strategy_idx,
        }


# ── Main Model ───────────────────────────────────────────────────────────────

class PLEv3(nn.Module):
    """
    PLE v3: Hierarchical Selection + Specialized Experts.

    Input: (B, n_features) market features + (B, n_account) account state
    Output: strategy selection + MAE/MFE predictions + confidence
    """

    def __init__(
        self,
        feature_partitions: dict[str, list[int]],
        n_account_features: int = 4,  # equity_pct, drawdown, consecutive_losses, position_count
        n_strategies: int = 100,
        expert_hidden: int = 128,
        expert_output: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_partitions = feature_partitions
        self.n_strategies = n_strategies

        # Specialized experts (different input features)
        self.experts = nn.ModuleDict()
        for name, indices in feature_partitions.items():
            self.experts[name] = SpecializedExpert(
                len(indices), expert_hidden, expert_output, dropout
            )

        # Account state encoder
        self.account_encoder = nn.Sequential(
            nn.Linear(n_account_features, expert_output // 2),
            nn.GELU(),
        )

        # Fusion: gated expert output + account
        account_dim = expert_output // 2
        fusion_input_dim = expert_output + account_dim  # gated + account
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        # Expert attention gate
        n_experts = len(feature_partitions)
        total_expert_dim = n_experts * expert_output + account_dim
        self.gate_query = nn.Linear(total_expert_dim, expert_output)
        self.gate_keys = nn.ModuleDict({
            name: nn.Linear(expert_output, expert_output)
            for name in feature_partitions
        })

        # Hierarchical selector
        self.selector = HierarchicalSelector(fusion_dim, dropout)

        # MAE/MFE towers (predict for selected strategy)
        tower_input = fusion_dim + 2 + 4 + 4 + 4  # dir + regime + tf + rr one-hots
        self.mae_tower = nn.Sequential(
            nn.Linear(tower_input, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.mfe_tower = nn.Sequential(
            nn.Linear(tower_input, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, account_state: torch.Tensor) -> dict:
        """
        Args:
            features: (B, n_features) market features
            account_state: (B, 4) [equity_pct, drawdown, consec_losses, position_count]
        """
        # Run each expert on its feature subset
        expert_outputs = {}
        for name, indices in self.feature_partitions.items():
            idx_tensor = torch.tensor(indices, device=features.device)
            x_subset = features[:, idx_tensor]
            expert_outputs[name] = self.experts[name](x_subset)

        # Account encoding
        account_enc = self.account_encoder(account_state)

        # Stack expert outputs
        expert_list = list(expert_outputs.values())
        expert_names = list(expert_outputs.keys())
        expert_stacked = torch.stack(expert_list, dim=1)  # (B, n_experts, expert_dim)

        # Gated fusion with attention
        all_expert_concat = torch.cat(expert_list, dim=-1)  # (B, n_experts*expert_dim)
        q = self.gate_query(torch.cat([all_expert_concat, account_enc], dim=-1))  # needs resize

        gate_scores = []
        for name in expert_names:
            k = self.gate_keys[name](expert_outputs[name])
            score = (q * k).sum(dim=-1, keepdim=True)
            gate_scores.append(score)

        gate_scores = torch.cat(gate_scores, dim=-1)
        gate_weights = F.softmax(gate_scores / (q.shape[-1] ** 0.5), dim=-1)

        gated = torch.bmm(gate_weights.unsqueeze(1), expert_stacked).squeeze(1)

        fused = self.fusion(torch.cat([gated, account_enc], dim=-1))

        # Hierarchical strategy selection
        selection = self.selector(fused)

        # MAE/MFE prediction (conditioned on selection)
        dir_oh = F.one_hot(selection["dir_probs"].argmax(-1), 2).float()
        regime_oh = F.one_hot(selection["regime_probs"].argmax(-1), 4).float()
        tf_oh = F.one_hot(selection["tf_probs"].argmax(-1), 4).float()
        rr_oh = F.one_hot(selection["rr_probs"].argmax(-1), 4).float()

        pred_input = torch.cat([fused, dir_oh, regime_oh, tf_oh, rr_oh], dim=-1)
        mae_pred = self.mae_tower(pred_input).squeeze(-1)
        mfe_pred = self.mfe_tower(pred_input).squeeze(-1)

        confidence = self.confidence_head(fused).squeeze(-1)

        return {
            **selection,
            "mae_pred": mae_pred,       # (B,) MAE for selected strategy
            "mfe_pred": mfe_pred,       # (B,) MFE for selected strategy
            "confidence": confidence,   # (B,)
            "gate_weights": gate_weights,  # (B, n_experts)
            "expert_names": expert_names,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
