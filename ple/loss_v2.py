"""
PLE v2 Loss System

5 components with uncertainty weighting:
1. L_select:    Strategy selection (CE with oracle)
2. L_calibrate: MAE/MFE quantile regression
3. L_diversity: Expert differentiation + gate entropy
4. L_equity:    Differentiable Sharpe on batch
5. L_uncertainty: Auto-balancing (Kendall 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target, mask):
        """
        pred: (B, n_labels)
        target: (B, n_labels)
        mask: (B, n_labels) — 1 where valid
        """
        losses = []
        for q in self.quantiles:
            error = target - pred
            loss = torch.where(error >= 0, q * error, (q - 1) * error)
            masked = (loss * mask).sum() / mask.sum().clamp(min=1)
            losses.append(masked)
        return torch.stack(losses).mean()


class DifferentiableSharpe(nn.Module):
    """Differentiable Sharpe ratio for batch of returns."""

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        returns: (B,) — sequence of trade returns in batch
        Returns: negative Sharpe (to minimize)
        """
        if len(returns) < 2:
            return torch.tensor(0.0, device=returns.device)
        mean_r = returns.mean()
        std_r = returns.std().clamp(min=1e-8)
        sharpe = mean_r / std_r
        return -sharpe  # negative because we minimize


class PLEv2Loss(nn.Module):
    """
    Complete loss system for PLE v2.

    L_total = Σ (1/2σi²) * Li + log(σi)  (uncertainty weighted)
    """

    def __init__(self, n_losses: int = 5, diversity_weight: float = 0.1):
        super().__init__()
        # Learnable uncertainty parameters
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_losses))
        self.quantile_loss = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
        self.diff_sharpe = DifferentiableSharpe()
        self.diversity_weight = diversity_weight

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["strategy_logits"].device

        # ─── 1. Strategy Selection Loss ───
        # Oracle: which label had the best actual RAR?
        rar_actual = batch["rar"]           # (B, 100)
        rar_mask = batch["rar_mask"]        # (B, 100)

        # Replace NaN/masked with -inf so they're never selected as oracle
        rar_for_oracle = rar_actual.clone()
        rar_for_oracle[rar_mask == 0] = -1e9

        oracle_idx = rar_for_oracle.argmax(dim=1)  # (B,) best strategy per sample
        L_select = F.cross_entropy(outputs["strategy_logits"], oracle_idx)

        # ─── 2. Calibration Loss (Quantile) ───
        mae_mask = batch["reg_mask"]
        mfe_mask = batch["reg_mask"]

        L_mae = self.quantile_loss(outputs["mae_pred"], batch["mae"], mae_mask)
        L_mfe = self.quantile_loss(outputs["mfe_pred"], batch["mfe"], mfe_mask)
        L_calibrate = L_mae + L_mfe

        # ─── 3. Expert Diversity Loss ───
        L_diversity = torch.tensor(0.0, device=device)

        # Expert output cosine similarity (should be low)
        expert_outs = outputs.get("expert_outputs", [])
        n_pairs = 0
        for expert_group in expert_outs:
            if len(expert_group) < 2:
                continue
            for i in range(len(expert_group)):
                for j in range(i + 1, len(expert_group)):
                    cos_sim = F.cosine_similarity(
                        expert_group[i].detach(), expert_group[j].detach(), dim=-1
                    ).mean()
                    L_diversity = L_diversity + cos_sim
                    n_pairs += 1

        if n_pairs > 0:
            L_diversity = L_diversity / n_pairs

        # Gate entropy (should be high = using multiple experts)
        gate_weights = outputs.get("gate_weights", [])
        L_gate_entropy = torch.tensor(0.0, device=device)
        for gw in gate_weights:
            entropy = -(gw * (gw + 1e-8).log()).sum(dim=-1).mean()
            L_gate_entropy = L_gate_entropy - entropy  # negative because we minimize

        L_diversity = self.diversity_weight * (L_diversity + L_gate_entropy)

        # ─── 4. Equity Curve Loss ───
        # Simulated returns from model's strategy selections
        selected_idx = outputs["strategy_probs"].argmax(dim=1)  # (B,)
        B = selected_idx.shape[0]

        # Get RAR of selected strategy for each sample
        selected_rar = rar_actual[torch.arange(B, device=device), selected_idx]
        # Mask invalid selections
        selected_valid = rar_mask[torch.arange(B, device=device), selected_idx]
        selected_returns = selected_rar * selected_valid

        # Confidence-weighted returns
        conf = outputs["confidence"]
        weighted_returns = selected_returns * conf

        L_equity = self.diff_sharpe(weighted_returns)

        # ─── 5. Uncertainty Weighting ───
        losses = [L_select, L_calibrate, L_diversity, L_equity]

        # Add confidence calibration: confidence should predict if selection was correct
        oracle_match = (selected_idx == oracle_idx).float()
        L_conf = F.binary_cross_entropy(conf, oracle_match.detach())
        losses.append(L_conf)

        total = torch.tensor(0.0, device=device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma_sq[i])
            total = total + precision * loss + self.log_sigma_sq[i]

        # Task weights for monitoring
        with torch.no_grad():
            task_weights = torch.exp(-self.log_sigma_sq)

        return {
            "total": total,
            "L_select": L_select.detach(),
            "L_calibrate": L_calibrate.detach(),
            "L_diversity": L_diversity.detach(),
            "L_equity": L_equity.detach(),
            "L_conf": L_conf.detach(),
            "task_weights": task_weights,
            "mean_confidence": conf.mean().detach(),
            "oracle_acc": oracle_match.mean().detach(),
        }
