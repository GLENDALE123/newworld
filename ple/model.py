"""
PLE (Progressive Layered Extraction) Multi-Task Trading Model

Architecture:
  - N Expert networks per task + M Shared experts
  - Gating networks select expert mix per task
  - 2 extraction levels (progressive)
  - 3 task towers: TBM classification (100), MAE/MFE regression (200), RAR regression (100)

Input: 416 auto-generated features
Output: 400 simultaneous predictions (100 TBM + 100 MAE + 100 MFE + 100 RAR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertNetwork(nn.Module):
    """Single expert: 2-layer MLP with residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

        # Residual projection if dims don't match
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = F.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.norm(out + residual)


class GatingNetwork(nn.Module):
    """Soft gating: learns which experts to attend to for this task."""

    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        # x: (batch, input_dim)
        # expert_outputs: list of (batch, expert_output_dim)
        weights = self.gate(x)  # (batch, n_experts)
        stacked = torch.stack(expert_outputs, dim=1)  # (batch, n_experts, expert_dim)
        # Weighted sum
        out = torch.bmm(weights.unsqueeze(1), stacked).squeeze(1)  # (batch, expert_dim)
        return out


class PLELayer(nn.Module):
    """One PLE extraction level: shared experts + task-specific experts + gating."""

    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        n_shared_experts: int,
        n_task_experts: int,
        n_tasks: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        total_experts = n_shared_experts + n_task_experts

        # Shared experts
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_dim * 2, expert_dim, dropout)
            for _ in range(n_shared_experts)
        ])

        # Task-specific experts
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertNetwork(input_dim, expert_dim * 2, expert_dim, dropout)
                for _ in range(n_task_experts)
            ])
            for _ in range(n_tasks)
        ])

        # Gating networks (one per task)
        self.gates = nn.ModuleList([
            GatingNetwork(input_dim, total_experts)
            for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Compute shared expert outputs
        shared_outs = [expert(x) for expert in self.shared_experts]

        # Per-task: compute task experts + gate
        task_outputs = []
        for t in range(self.n_tasks):
            task_outs = [expert(x) for expert in self.task_experts[t]]
            all_outs = shared_outs + task_outs  # combine shared + task
            gated = self.gates[t](x, all_outs)
            task_outputs.append(gated)

        return task_outputs


class TaskTower(nn.Module):
    """Task-specific output tower."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tower(x)
        if self.task_type == "classification":
            return torch.sigmoid(out)  # probability per label
        return out  # regression: raw values


class PLETradingModel(nn.Module):
    """
    Full PLE Trading Model.

    Input: (batch, n_features) — 416 auto-generated features
    Output: dict with:
      - tbm_probs: (batch, 100) — P(take profit hit) per TBM label
      - mae_pred:  (batch, 100) — predicted MAE per label
      - mfe_pred:  (batch, 100) — predicted MFE per label
      - rar_pred:  (batch, 100) — predicted risk-adjusted return per label
    """

    def __init__(
        self,
        n_features: int = 416,
        n_labels: int = 100,
        expert_dim: int = 128,
        n_shared_experts: int = 3,
        n_task_experts: int = 2,
        n_tasks: int = 4,  # TBM, MAE, MFE, RAR
        n_levels: int = 2,
        tower_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_labels = n_labels
        self.n_tasks = n_tasks

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, expert_dim),
            nn.GELU(),
            nn.LayerNorm(expert_dim),
        )

        # PLE levels
        self.ple_levels = nn.ModuleList([
            PLELayer(
                input_dim=expert_dim,
                expert_dim=expert_dim,
                n_shared_experts=n_shared_experts,
                n_task_experts=n_task_experts,
                n_tasks=n_tasks,
                dropout=dropout,
            )
            for _ in range(n_levels)
        ])

        # Task towers
        self.tbm_tower = TaskTower(expert_dim, tower_hidden, n_labels, "classification")
        self.mae_tower = TaskTower(expert_dim, tower_hidden, n_labels, "regression")
        self.mfe_tower = TaskTower(expert_dim, tower_hidden, n_labels, "regression")
        self.rar_tower = TaskTower(expert_dim, tower_hidden, n_labels, "regression")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Input projection
        h = self.input_proj(x)  # (batch, expert_dim)

        # Progressive extraction
        for level in self.ple_levels:
            task_outputs = level(h)
            # For next level, use mean of task outputs as shared input
            h = torch.stack(task_outputs).mean(dim=0)

        # Task towers (using final level's task-specific outputs)
        task_outputs = self.ple_levels[-1](h)

        return {
            "tbm_probs": self.tbm_tower(task_outputs[0]),   # (batch, 100)
            "mae_pred": self.mae_tower(task_outputs[1]),     # (batch, 100)
            "mfe_pred": self.mfe_tower(task_outputs[2]),     # (batch, 100)
            "rar_pred": self.rar_tower(task_outputs[3]),     # (batch, 100)
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
