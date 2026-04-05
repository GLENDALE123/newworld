"""
PLE v3 Loss: Hierarchical CE + Calibration + Gate Balance + Equity Sharpe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLEv3Loss(nn.Module):
    def __init__(self, n_losses: int = 5):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_losses))

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["regime_logits"].device
        B = outputs["regime_logits"].shape[0]

        # ── Oracle: decompose best strategy into regime/tf/rr ──
        rar = batch["rar"]          # (B, 100)
        rar_mask = batch["rar_mask"]  # (B, 100)

        rar_oracle = rar.clone()
        rar_oracle[rar_mask == 0] = -1e9
        best_flat = rar_oracle.argmax(dim=1)  # (B,)

        # Decompose flat index → regime, tf, rr
        oracle_regime = best_flat // 25        # 0-3
        oracle_tf = (best_flat % 25) // 5      # 0-4
        oracle_rr = best_flat % 5              # 0-4

        # ── L1: Hierarchical Selection Loss ──
        L_regime = F.cross_entropy(outputs["regime_logits"], oracle_regime)
        L_tf = F.cross_entropy(outputs["tf_logits"], oracle_tf)
        L_rr = F.cross_entropy(outputs["rr_logits"], oracle_rr)
        L_select = L_regime + L_tf + L_rr

        # ── L2: MAE/MFE Calibration ──
        # Get actual MAE/MFE for the selected strategy
        selected_idx = outputs["strategy_idx"]  # (B,)
        selected_idx_clamped = selected_idx.clamp(0, 99)

        actual_mae = batch["mae"][torch.arange(B, device=device), selected_idx_clamped]
        actual_mfe = batch["mfe"][torch.arange(B, device=device), selected_idx_clamped]
        valid = batch["reg_mask"][torch.arange(B, device=device), selected_idx_clamped]

        L_mae = ((outputs["mae_pred"] - actual_mae) ** 2 * valid).sum() / valid.sum().clamp(1)
        L_mfe = ((outputs["mfe_pred"] - actual_mfe) ** 2 * valid).sum() / valid.sum().clamp(1)
        L_calibrate = L_mae + L_mfe

        # ── L3: Gate Balance ──
        # Each expert should be used ~25% of the time
        gate_w = outputs["gate_weights"]  # (B, n_experts)
        mean_usage = gate_w.mean(dim=0)  # (n_experts,)
        target_usage = torch.ones_like(mean_usage) / mean_usage.shape[0]
        L_balance = F.mse_loss(mean_usage, target_usage)

        # Gate entropy (should be moderate, not max and not collapsed)
        entropy = -(gate_w * (gate_w + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(gate_w.shape[1]), device=device))
        # Target: 60-80% of max entropy (some specialization but not collapse)
        target_entropy = 0.7 * max_entropy
        L_entropy = (entropy - target_entropy) ** 2
        L_gate = L_balance + L_entropy

        # ── L4: Equity (differentiable Sharpe on selected returns) ──
        selected_rar = rar[torch.arange(B, device=device), selected_idx_clamped]
        selected_valid = rar_mask[torch.arange(B, device=device), selected_idx_clamped]
        returns = selected_rar * selected_valid * outputs["confidence"]

        mean_r = returns.mean()
        std_r = returns.std().clamp(min=1e-8)
        L_equity = -(mean_r / std_r)

        # ── L5: Confidence calibration ──
        oracle_match = (selected_idx == best_flat).float()
        L_conf = F.binary_cross_entropy(outputs["confidence"], oracle_match.detach())

        # ── Uncertainty weighting ──
        losses = [L_select, L_calibrate, L_gate, L_equity, L_conf]
        total = torch.tensor(0.0, device=device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma_sq[i])
            total = total + precision * loss + self.log_sigma_sq[i]

        with torch.no_grad():
            weights = torch.exp(-self.log_sigma_sq)

        return {
            "total": total,
            "L_select": L_select.detach(),
            "L_regime": L_regime.detach(),
            "L_tf": L_tf.detach(),
            "L_rr": L_rr.detach(),
            "L_calibrate": L_calibrate.detach(),
            "L_gate": L_gate.detach(),
            "L_equity": L_equity.detach(),
            "L_conf": L_conf.detach(),
            "gate_entropy": entropy.detach(),
            "oracle_regime_acc": (outputs["regime_probs"].argmax(-1) == oracle_regime).float().mean().detach(),
            "oracle_tf_acc": (outputs["tf_probs"].argmax(-1) == oracle_tf).float().mean().detach(),
            "oracle_rr_acc": (outputs["rr_probs"].argmax(-1) == oracle_rr).float().mean().detach(),
            "confidence": outputs["confidence"].mean().detach(),
            "task_weights": weights,
        }
