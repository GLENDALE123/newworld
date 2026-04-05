"""
PLE v4 Loss: Multi-Label BCE + MAE/MFE Calibration + Equity Sharpe

Key: BCE per label independently, NOT cross-entropy.
Target: 1 if RAR_net > 0 (strategy is profitable after fees), else 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLEv4Loss(nn.Module):
    def __init__(self, n_losses: int = 4):
        super().__init__()
        self.log_sigma_sq = nn.Parameter(torch.zeros(n_losses))

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["label_logits"].device
        B = outputs["label_logits"].shape[0]

        rar = batch["rar"]          # (B, 128)
        rar_mask = batch["rar_mask"]  # (B, 128)

        # ── L1: Multi-label BCE with class balance + sample weights ──
        target = (rar > 0).float() * rar_mask  # (B, n_strats)

        # Class balance: upweight positives (minority class ~36%)
        n_pos = (target * rar_mask).sum().clamp(1)
        n_neg = ((1 - target) * rar_mask).sum().clamp(1)
        pos_weight = (n_neg / n_pos).clamp(1.0, 5.0)  # cap at 5x

        bce = F.binary_cross_entropy_with_logits(
            outputs["label_logits"],
            target,
            pos_weight=pos_weight.expand_as(target),
            reduction="none",
        )

        # Apply sample weights if available (wgt from TBM: faster+bigger wins = higher weight)
        sample_wgt = batch.get("wgt", None)
        if sample_wgt is not None:
            wgt_mask = sample_wgt * rar_mask
            wgt_mask = wgt_mask / wgt_mask.sum().clamp(1) * rar_mask.sum().clamp(1)  # normalize
            L_label = (bce * wgt_mask).sum() / rar_mask.sum().clamp(1)
        else:
            L_label = (bce * rar_mask).sum() / rar_mask.sum().clamp(1)

        # ── L2: MAE/MFE calibration ──
        reg_mask = batch["reg_mask"]
        L_mae = ((outputs["mae_pred"] - batch["mae"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(1)
        L_mfe = ((outputs["mfe_pred"] - batch["mfe"]) ** 2 * reg_mask).sum() / reg_mask.sum().clamp(1)
        L_cal = L_mae + L_mfe

        # ── L3: Equity Sharpe ──
        # Simulate: for each sample, take the strategy with highest predicted prob
        # If any strategy fires, use that strategy's actual RAR
        probs = outputs["label_probs"]  # (B, 128)
        # Mask invalid
        masked_probs = probs * rar_mask + (-1) * (1 - rar_mask)
        best_idx = masked_probs.argmax(dim=1)  # (B,)

        selected_rar = rar[torch.arange(B, device=device), best_idx]
        selected_mask = rar_mask[torch.arange(B, device=device), best_idx]
        selected_prob = probs[torch.arange(B, device=device), best_idx]

        # Only count trades where model is confident (prob > 0.5)
        trade_mask = (selected_prob > 0.5) & (selected_mask > 0)
        trade_returns = selected_rar * trade_mask.float()

        if trade_mask.sum() > 1:
            mean_r = trade_returns[trade_mask].mean()
            std_r = trade_returns[trade_mask].std().clamp(min=1e-8)
            L_equity = -(mean_r / std_r)
        else:
            L_equity = torch.tensor(0.0, device=device)

        # ── L4: Confidence calibration ──
        # Confidence should predict "is there ANY profitable strategy right now?"
        any_profitable = (rar * rar_mask > 0).any(dim=1).float()  # (B,)
        L_conf = F.binary_cross_entropy(outputs["confidence"], any_profitable.detach())

        # ── Uncertainty weighting ──
        losses = [L_label, L_cal, L_equity, L_conf]
        total = torch.tensor(0.0, device=device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma_sq[i])
            total = total + precision * loss + self.log_sigma_sq[i]

        # Metrics
        with torch.no_grad():
            weights = torch.exp(-self.log_sigma_sq)
            # How many strategies fire per sample (prob > 0.5)?
            n_active = (probs * rar_mask > 0.5).float().sum(dim=1).mean()
            # Accuracy: of those that fire, how many are actually profitable?
            fired = (probs * rar_mask > 0.5)
            if fired.sum() > 0:
                precision_metric = (target[fired] > 0.5).float().mean()
            else:
                precision_metric = torch.tensor(0.0)
            # No-trade ratio
            no_trade = (probs.max(dim=1).values < 0.5).float().mean()

        return {
            "total": total,
            "L_label": L_label.detach(),
            "L_cal": L_cal.detach(),
            "L_equity": L_equity.detach(),
            "L_conf": L_conf.detach(),
            "n_active": n_active,
            "precision": precision_metric,
            "no_trade_pct": no_trade,
            "confidence": outputs["confidence"].mean().detach(),
            "task_weights": weights,
        }
