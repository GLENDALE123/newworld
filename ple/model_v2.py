"""
PLE v2: Corrected Progressive Layered Extraction

Fixes from v1:
1. Task-specific paths preserved between levels (no mean collapse)
2. Strategy selection via softmax (competitive, not independent)
3. Expert diversity enforced via cosine similarity loss
4. Gate attention considers expert outputs
5. Account state features injected at input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class AttentionGate(nn.Module):
    """Gate that considers both input AND expert outputs for routing."""

    def __init__(self, input_dim: int, expert_dim: int, n_experts: int):
        super().__init__()
        self.query = nn.Linear(input_dim, expert_dim)
        self.key = nn.Linear(expert_dim, expert_dim)
        self.scale = expert_dim ** 0.5

    def forward(self, x, expert_outputs):
        # x: (B, input_dim), expert_outputs: list of (B, expert_dim)
        q = self.query(x)                              # (B, expert_dim)
        stacked = torch.stack(expert_outputs, dim=1)   # (B, n_experts, expert_dim)
        k = self.key(stacked)                          # (B, n_experts, expert_dim)

        scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2) / self.scale  # (B, n_experts)
        weights = F.softmax(scores, dim=-1)            # (B, n_experts)

        out = torch.bmm(weights.unsqueeze(1), stacked).squeeze(1)  # (B, expert_dim)
        return out, weights


class PLELevel(nn.Module):
    """One PLE level with proper task-path preservation."""

    def __init__(self, dim: int, n_shared: int, n_task_experts: int,
                 n_tasks: int, dropout: float = 0.1):
        super().__init__()
        self.n_tasks = n_tasks
        n_total = n_shared + n_task_experts

        self.shared = nn.ModuleList([Expert(dim, dim * 2, dropout) for _ in range(n_shared)])

        self.task_experts = nn.ModuleList([
            nn.ModuleList([Expert(dim, dim * 2, dropout) for _ in range(n_task_experts)])
            for _ in range(n_tasks)
        ])

        self.gates = nn.ModuleList([
            AttentionGate(dim, dim, n_total) for _ in range(n_tasks)
        ])

    def forward(self, task_inputs: list[torch.Tensor]):
        """
        Args:
            task_inputs: list of (B, dim) — one per task
                         First call: all same (projected input)
                         Subsequent: each task's output from previous level

        Returns:
            task_outputs: list of (B, dim)
            all_gate_weights: list of (B, n_experts) for diversity monitoring
            all_expert_outputs: list of list of (B, dim) for diversity loss
        """
        # Shared experts: same input for all (mean of task inputs)
        shared_input = torch.stack(task_inputs).mean(dim=0)
        shared_outs = [e(shared_input) for e in self.shared]

        task_outputs = []
        all_weights = []
        all_expert_outs = []

        for t in range(self.n_tasks):
            task_outs = [e(task_inputs[t]) for e in self.task_experts[t]]
            all_outs = shared_outs + task_outs

            gated, weights = self.gates[t](task_inputs[t], all_outs)
            task_outputs.append(gated)
            all_weights.append(weights)
            all_expert_outs.append(all_outs)

        return task_outputs, all_weights, all_expert_outs


class StrategySelector(nn.Module):
    """Selects best strategy via softmax (competitive selection)."""

    def __init__(self, dim: int, n_strategies: int):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, n_strategies),
        )

    def forward(self, x):
        logits = self.tower(x)        # (B, n_strategies)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class RegressionTower(nn.Module):
    """Predicts continuous value per strategy (MAE/MFE/confidence)."""

    def __init__(self, dim: int, n_strategies: int):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, n_strategies),
        )

    def forward(self, x):
        return self.tower(x)


class PLEv2(nn.Module):
    """
    PLE v2 Trading Model.

    Input: (B, n_features) — market features + account state
    Output: dict with strategy selection, MAE/MFE predictions, confidence
    """

    def __init__(
        self,
        n_features: int = 420,
        n_strategies: int = 100,
        dim: int = 128,
        n_shared: int = 3,
        n_task_experts: int = 2,
        n_tasks: int = 4,    # strategy_select, mae, mfe, confidence
        n_levels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_levels = n_levels

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

        # PLE levels
        self.levels = nn.ModuleList([
            PLELevel(dim, n_shared, n_task_experts, n_tasks, dropout)
            for _ in range(n_levels)
        ])

        # Task towers
        self.strategy_selector = StrategySelector(dim, n_strategies)
        self.mae_tower = RegressionTower(dim, n_strategies)
        self.mfe_tower = RegressionTower(dim, n_strategies)
        self.confidence_tower = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.input_proj(x)  # (B, dim)

        # Initialize all task inputs to the same projection
        task_inputs = [h] * self.n_tasks

        # Progressive extraction
        all_gate_weights = []
        all_expert_outputs = []

        for level in self.levels:
            task_inputs, gate_w, expert_outs = level(task_inputs)
            all_gate_weights.extend(gate_w)
            all_expert_outputs.extend(expert_outs)

        # Task towers (each uses its own path's output)
        strategy_logits, strategy_probs = self.strategy_selector(task_inputs[0])
        mae_pred = self.mae_tower(task_inputs[1])
        mfe_pred = self.mfe_tower(task_inputs[2])
        confidence = self.confidence_tower(task_inputs[3]).squeeze(-1)

        return {
            "strategy_logits": strategy_logits,    # (B, 100) raw logits
            "strategy_probs": strategy_probs,      # (B, 100) softmax probs
            "mae_pred": mae_pred,                  # (B, 100)
            "mfe_pred": mfe_pred,                  # (B, 100)
            "confidence": confidence,              # (B,) overall confidence
            "gate_weights": all_gate_weights,      # for monitoring
            "expert_outputs": all_expert_outputs,  # for diversity loss
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
