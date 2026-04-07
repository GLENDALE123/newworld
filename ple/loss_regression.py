"""
Regression Loss — 직접 수익률 예측

Binary classification (win/lose) 대신 연속값 (RAR) 직접 예측.
출력이 곧 expected return → 높을수록 좋은 시그널.

Classification 문제점: prob이 0.7에 수렴, discriminate 안됨
Regression 해결: output이 수익률 자체, spread가 클수록 좋음
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """Direct return prediction loss.

    Targets: RAR per strategy (continuous)
    Output: predicted RAR per strategy

    Loss components:
      1. Huber loss on RAR prediction (robust to outliers)
      2. Ranking loss: correct ordering of strategies
      3. Sign accuracy: at least get the direction right
    """

    def __init__(self, huber_delta: float = 1.0, rank_weight: float = 0.1):
        super().__init__()
        self.huber_delta = huber_delta
        self.rank_weight = rank_weight

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["label_logits"].device

        rar = batch["rar"]            # (B, n_strategies) actual RAR
        rar_mask = batch["rar_mask"]  # (B, n_strategies) valid entries

        # Use label_logits as RAR prediction (no sigmoid)
        pred_rar = outputs["label_logits"]  # raw output, not sigmoid

        # L1: Huber loss on RAR
        huber = F.huber_loss(pred_rar, rar, reduction="none", delta=self.huber_delta)
        L_rar = (huber * rar_mask).sum() / rar_mask.sum().clamp(1)

        # L2: Ranking loss — if strategy A has higher actual RAR than B,
        # predicted RAR for A should also be higher
        # Use pairwise hinge loss on random pairs
        B, S = rar.shape
        if B > 1 and S > 1:
            # Sample pairs
            idx1 = torch.randint(S, (B, 4), device=device)
            idx2 = torch.randint(S, (B, 4), device=device)

            rar1 = rar.gather(1, idx1)
            rar2 = rar.gather(1, idx2)
            pred1 = pred_rar.gather(1, idx1)
            pred2 = pred_rar.gather(1, idx2)
            mask1 = rar_mask.gather(1, idx1)
            mask2 = rar_mask.gather(1, idx2)

            # Target: sign of actual difference
            diff_target = torch.sign(rar1 - rar2)
            diff_pred = pred1 - pred2
            pair_mask = mask1 * mask2

            # Hinge ranking loss
            L_rank = (F.relu(1.0 - diff_target * diff_pred) * pair_mask).sum() / pair_mask.sum().clamp(1)
        else:
            L_rank = torch.tensor(0.0, device=device)

        total = L_rar + self.rank_weight * L_rank

        # Metrics
        with torch.no_grad():
            # Best strategy per sample
            masked_pred = pred_rar * rar_mask + (-1e9) * (1 - rar_mask)
            best_idx = masked_pred.argmax(dim=1)
            selected_rar = rar[torch.arange(B, device=device), best_idx]
            selected_mask = rar_mask[torch.arange(B, device=device), best_idx]

            # Direction accuracy: did we pick a positive-RAR strategy?
            valid = selected_mask > 0
            if valid.sum() > 0:
                direction_acc = (selected_rar[valid] > 0).float().mean()
            else:
                direction_acc = torch.tensor(0.0)

            # Predicted RAR spread
            pred_spread = pred_rar[rar_mask > 0].std() if rar_mask.sum() > 0 else torch.tensor(0.0)

            # Selected avg RAR
            avg_selected = selected_rar[valid].mean() if valid.sum() > 0 else torch.tensor(0.0)

        return {
            "total": total,
            "L_rar": L_rar.detach(),
            "L_rank": L_rank.detach(),
            "direction_acc": direction_acc,
            "pred_spread": pred_spread,
            "avg_selected_rar": avg_selected,
            # Compatibility with v4 trainer logging
            "L_label": L_rar.detach(),
            "L_cal": L_rank.detach(),
            "L_equity": torch.tensor(0.0),
            "L_conf": torch.tensor(0.0),
            "n_active": torch.tensor(0.0),
            "precision": direction_acc,
            "no_trade_pct": torch.tensor(0.0),
            "confidence": torch.tensor(0.0),
        }
