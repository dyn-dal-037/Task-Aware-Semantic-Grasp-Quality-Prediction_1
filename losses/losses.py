"""
Loss functions for both models.

Baseline   : BCE + MSE
Physics    : BCE + MSE + consistency + ranking + region-margin

All losses return scalar tensors. Physics loss also returns a breakdown dict
for TensorBoard logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Loss
# ─────────────────────────────────────────────────────────────────────────────

class BaselineLoss(nn.Module):
    """
    L = L_cls + λ_reg * L_reg
    """

    def __init__(self, lambda_reg: float = 0.5):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, logit_cls: torch.Tensor,
                pred_stab: torch.Tensor,
                label: torch.Tensor,
                stab_target: torch.Tensor) -> tuple:
        """
        Args:
            logit_cls   : [B, 1]  raw logit
            pred_stab   : [B, 1]  raw (pre-sigmoid)
            label       : [B]     float {0, 1}
            stab_target : [B]     float [0, 1]
        Returns:
            total_loss  : scalar
            breakdown   : dict for logging
        """
        label = label.view(-1, 1)
        stab_target = stab_target.view(-1, 1)

        L_cls = F.binary_cross_entropy_with_logits(logit_cls, label)
        L_reg = F.mse_loss(torch.sigmoid(pred_stab), stab_target)

        total = L_cls + self.lambda_reg * L_reg

        return total, {
            "L_total": total.item(),
            "L_cls":   L_cls.item(),
            "L_reg":   L_reg.item()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Physics Loss Components
# ─────────────────────────────────────────────────────────────────────────────

def consistency_loss(pred_physics: torch.Tensor,
                     stab_target: torch.Tensor) -> torch.Tensor:
    """
    Physics head should match dataset stability score.
    L_consist = MSE(σ(pred_physics), stab_target)
    """
    return F.mse_loss(torch.sigmoid(pred_physics), stab_target.view(-1, 1))


def ranking_loss(logit_cls: torch.Tensor,
                 label: torch.Tensor,
                 margin: float = 0.10) -> torch.Tensor:
    """
    For all pairs (i,j) where label_i > label_j:
    enforce pred_i > pred_j by at least `margin`.

    Vectorized pairwise over batch — O(B²) but B is small (32).
    """
    p = torch.sigmoid(logit_cls).squeeze()   # [B]
    l = label.squeeze().float()              # [B]

    # Pairwise label difference
    diff_label = l.unsqueeze(1) - l.unsqueeze(0)    # [B, B]
    diff_pred  = p.unsqueeze(1) - p.unsqueeze(0)    # [B, B]

    # Only penalize pairs where label_i > label_j
    mask = (diff_label > 0).float()
    loss = mask * F.relu(margin - diff_pred)
    return loss.mean()


def region_margin_loss(logit_cls: torch.Tensor,
                       label: torch.Tensor,
                       pos_margin: float = 0.60,
                       neg_margin: float = 0.40) -> torch.Tensor:
    """
    Positive samples (label=1) must score above pos_margin.
    Negative samples (label=0) must score below neg_margin.
    Encourages well-separated predictions regardless of ranking.
    """
    p = torch.sigmoid(logit_cls).squeeze()   # [B]
    l = label.squeeze().float()              # [B]

    pos_mask = (l == 1).float()
    neg_mask = (l == 0).float()

    L_pos = pos_mask * F.relu(pos_margin - p)
    L_neg = neg_mask * F.relu(p - neg_margin)

    return (L_pos + L_neg).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Physics Model Loss
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """
    L = L_cls
      + λ_reg     * L_reg
      + λ_physics * (λ_consist * L_consist + λ_rank * L_rank + λ_region * L_region)

    Physics loss is gated: only applied after `warmup_epochs`.
    Pass `apply_physics=False` during warmup.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        loss_cfg = cfg["losses"]["physics"]

        self.lambda_reg     = loss_cfg["lambda_reg"]
        self.lambda_physics = loss_cfg["lambda_physics"]
        self.lambda_consist = loss_cfg["lambda_consist"]
        self.lambda_rank    = loss_cfg["lambda_rank"]
        self.lambda_region  = loss_cfg["lambda_region"]
        self.rank_margin    = loss_cfg["rank_margin"]
        self.pos_margin     = loss_cfg["pos_margin"]
        self.neg_margin     = loss_cfg["neg_margin"]

    def forward(self, logit_cls: torch.Tensor,
                pred_stab: torch.Tensor,
                pred_physics: torch.Tensor,
                label: torch.Tensor,
                stab_target: torch.Tensor,
                apply_physics: bool = True) -> tuple:
        """
        Args:
            logit_cls    : [B, 1]
            pred_stab    : [B, 1]
            pred_physics : [B, 1]
            label        : [B]   float
            stab_target  : [B]   float
            apply_physics: bool  — False during warmup epochs
        """
        label       = label.view(-1)
        stab_target = stab_target.view(-1, 1)

        L_cls = F.binary_cross_entropy_with_logits(
            logit_cls, label.view(-1, 1).float()
        )
        L_reg = F.mse_loss(torch.sigmoid(pred_stab), stab_target)

        breakdown = {
            "L_cls":     L_cls.item(),
            "L_reg":     L_reg.item(),
            "L_consist": 0.0,
            "L_rank":    0.0,
            "L_region":  0.0,
            "L_physics": 0.0,
        }

        total = L_cls + self.lambda_reg * L_reg

        if apply_physics:
            L_consist = consistency_loss(pred_physics, stab_target.squeeze())
            L_rank    = ranking_loss(logit_cls, label, self.rank_margin)
            L_region  = region_margin_loss(logit_cls, label,
                                           self.pos_margin, self.neg_margin)

            L_phys = (self.lambda_consist * L_consist +
                      self.lambda_rank    * L_rank    +
                      self.lambda_region  * L_region)

            total = total + self.lambda_physics * L_phys

            breakdown.update({
                "L_consist": L_consist.item(),
                "L_rank":    L_rank.item(),
                "L_region":  L_region.item(),
                "L_physics": L_phys.item(),
            })

        breakdown["L_total"] = total.item()
        return total, breakdown
