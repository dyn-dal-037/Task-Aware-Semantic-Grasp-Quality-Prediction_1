"""
Physics Model: ResNet50 + FiLM + Physics Head
Adds a third head that predicts physics consistency,
supervised by ranking, consistency, and region-margin losses.
"""

import torch
import torch.nn as nn

from models.baseline import SharedBackbone


class ResNetFiLMPhysics(nn.Module):
    """
    Three-head model:
        logit_cls    : [B, 1]  binary classification logit
        pred_stab    : [B, 1]  regression head (pre-sigmoid)
        pred_physics : [B, 1]  physics consistency (pre-sigmoid)

    Physics head is slightly deeper than the other heads
    to give it capacity to learn constraint satisfaction.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = SharedBackbone(cfg)
        d = self.backbone.out_dim  # 128

        # ── Heads ──
        self.cls_head = nn.Linear(d, 1)
        self.reg_head = nn.Linear(d, 1)

        self.physics_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self._init_heads()

    def _init_heads(self):
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)
        nn.init.xavier_uniform_(self.reg_head.weight)
        nn.init.zeros_(self.reg_head.bias)
        for m in self.physics_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor,
                task_id: torch.Tensor,
                grasp_point: torch.Tensor):
        """
        Returns:
            logit_cls    : [B, 1]
            pred_stab    : [B, 1]
            pred_physics : [B, 1]
        """
        z = self.backbone(image, task_id, grasp_point)   # [B, 128]
        return (
            self.cls_head(z),       # logit
            self.reg_head(z),       # stability (pre-sigmoid)
            self.physics_head(z)    # physics consistency (pre-sigmoid)
        )

    def predict(self, image, task_id, grasp_point):
        """Inference-mode — returns probabilities for all three heads."""
        self.eval()
        with torch.no_grad():
            logit_cls, pred_stab, pred_physics = self(image, task_id, grasp_point)
            p_cls     = torch.sigmoid(logit_cls).squeeze()
            p_stab    = torch.sigmoid(pred_stab).squeeze()
            p_physics = torch.sigmoid(pred_physics).squeeze()
        return p_cls, p_stab, p_physics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def physics_head_parameters(self):
        """Returns only physics head params — useful for separate LR scheduling."""
        return self.physics_head.parameters()
