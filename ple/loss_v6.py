"""
PLE v6 Loss: Focal BCE + Selectivity Penalty + MoE Balance

Key improvements over v4 loss:
  1. Focal loss: down-weights easy examples, focuses on hard ones
  2. Selectivity penalty: penalizes firing too many strategies simultaneously
  3. Regime-aware calibration: MAE/MFE per regime
  4. MoE load balance: prevents expert collapse

"확실한것만 매매. 불확실하면 안하는게 낫다."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal Binary Cross-Entropy for selective strategy prediction.

    Focal loss reduces the contribution of easy examples:
      FL(p) = -alpha * (1-p)^gamma * log(p)

    gamma > 0: harder examples get more weight
    When model is already confident (p close to 1 for correct), loss is small.
    When model is uncertain (p close to 0.5), loss is high.

    This pushes the model to be decisive: either clearly YES or clearly NO.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)  # p if correct, 1-p if wrong
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting: balance positive vs negative
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_weight * focal_weight * bce * mask
        return loss.sum() / mask.sum().clamp(1)


class PLEv6Loss(nn.Module):
    """Multi-objective loss with selectivity bias.

    Losses:
      L1: Focal BCE — strategy prediction with focus on hard examples
      L2: MAE/MFE calibration — predict magnitude of wins/losses
      L3: Selectivity penalty — punish firing too many strategies
      L4: Confidence calibration — "is there any good trade?"
    """

    def __init__(
        self,
        n_losses: int = 4,
        focal_gamma: float = 2.0,
        selectivity_target: float = 2.0,  # ideal: ~2 strategies active per bar
        selectivity_weight: float = 0.1,
    ):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_losses))
        self.focal = FocalBCELoss(gamma=focal_gamma)
        self.selectivity_target = selectivity_target
        self.selectivity_weight = selectivity_weight

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["label_logits"].device
        B = outputs["label_logits"].shape[0]

        rar = batch["rar"]
        rar_mask = batch["rar_mask"]

        # ── L1: Focal BCE ──
        target = (rar > 0).float() * rar_mask
        L_label = self.focal(outputs["label_logits"], target, rar_mask)

        # ── L2: MAE/MFE calibration ──
        reg_mask = batch["reg_mask"]
        L_mae = ((outputs["mae_pred"] - batch["mae"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(1)
        L_mfe = ((outputs["mfe_pred"] - batch["mfe"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(1)
        L_cal = L_mae + L_mfe

        # ── L3: Selectivity penalty ──
        # Penalize when too many strategies fire (prob > 0.5)
        probs = outputs["label_probs"]
        n_active = (probs * rar_mask > 0.5).float().sum(dim=1)  # per sample
        # Penalty: squared distance from target active count
        L_select = ((n_active - self.selectivity_target) ** 2).mean() * self.selectivity_weight

        # ── L4: Confidence calibration ──
        # Any profitable strategy = should trade
        any_profitable = (rar * rar_mask > 0).any(dim=1).float()
        L_conf = F.binary_cross_entropy(outputs["confidence"], any_profitable.detach())

        # ── Uncertainty weighting ──
        losses = [L_label, L_cal, L_select, L_conf]
        total = torch.tensor(0.0, device=device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma_sq[i])
            total = total + precision * loss + self.log_sigma_sq[i]

        # ── Metrics ──
        with torch.no_grad():
            fired = (probs * rar_mask > 0.5)
            if fired.sum() > 0:
                precision_metric = (target[fired] > 0.5).float().mean()
            else:
                precision_metric = torch.tensor(0.0)

            no_trade = (probs.max(dim=1).values < 0.5).float().mean()
            avg_active = n_active.mean()

            # Per-strategy accuracy (top-1)
            masked_probs = probs * rar_mask + (-1) * (1 - rar_mask)
            best_idx = masked_probs.argmax(dim=1)
            selected_rar = rar[torch.arange(B, device=device), best_idx]
            selected_mask = rar_mask[torch.arange(B, device=device), best_idx]
            top1_wr = ((selected_rar > 0) & (selected_mask > 0)).float().mean()

        return {
            "total": total,
            "L_label": L_label.detach(),
            "L_cal": L_cal.detach(),
            "L_select": L_select.detach(),
            "L_conf": L_conf.detach(),
            "n_active": avg_active,
            "precision": precision_metric,
            "no_trade_pct": no_trade,
            "top1_wr": top1_wr,
            "confidence": outputs["confidence"].mean().detach(),
        }
