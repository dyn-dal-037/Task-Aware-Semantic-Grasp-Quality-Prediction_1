"""
Baseline Model: ResNet50 + FiLM
- Classification head : P(label=1) — task-semantic correctness
- Regression head     : predicted stability score
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet34_Weights

from models.film import FiLM, GraspPointEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Shared Backbone
# ─────────────────────────────────────────────────────────────────────────────

class SharedBackbone(nn.Module):
    """
    ResNet → FiLM conditioning → fuse grasp point → trunk → shared 128-d repr.
    Used by both baseline and physics models to avoid code duplication.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        backbone_name   = cfg["model"]["backbone"]
        task_embed_dim  = cfg["model"]["task_embed_dim"]
        dropout         = cfg["model"]["dropout"]
        trunk_dims      = cfg["model"]["trunk_dims"]   # e.g. [512, 128]
        num_tasks       = len(next(iter(cfg["dataset"]["objects"].values()))["tasks"])

        # ── ResNet backbone ──
        if backbone_name == "resnet50":
            resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            feat_dim = 2048
        elif backbone_name == "resnet34":
            resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Strip classifier head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = feat_dim

        # ── Task conditioning ──
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.film = FiLM(task_embed_dim, feat_dim)

        # ── Grasp point encoding ──
        grasp_embed_dim = 64
        self.grasp_encoder = GraspPointEncoder(in_dim=2, embed_dim=grasp_embed_dim)

        # ── Trunk ──
        # Input: feat_dim + grasp_embed_dim after fusion
        trunk_in = feat_dim + grasp_embed_dim
        layers = []
        in_d = trunk_in
        for out_d in trunk_dims:
            layers += [
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_d = out_d
        self.trunk = nn.Sequential(*layers)
        self.out_dim = trunk_dims[-1]   # 128

        self._frozen = False

    def freeze_backbone(self):
        """Freeze ResNet weights — called for first N epochs."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        """Unfreeze ResNet — called after warmup."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, image: torch.Tensor,
                task_id: torch.Tensor,
                grasp_point: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image       : [B, 3, 224, 224]
            task_id     : [B]  long
            grasp_point : [B, 2]  normalized to [-1, 1]
        Returns:
            z           : [B, 128]  shared representation
        """
        # Visual features
        feat = self.backbone(image).squeeze(-1).squeeze(-1)   # [B, feat_dim]

        # FiLM conditioning
        t_emb = self.task_embedding(task_id)                  # [B, task_embed_dim]
        feat  = self.film(feat, t_emb)                        # [B, feat_dim]

        # Grasp point encoding & fusion
        g_emb = self.grasp_encoder(grasp_point)               # [B, 64]
        fused = torch.cat([feat, g_emb], dim=1)               # [B, feat_dim+64]

        return self.trunk(fused)                               # [B, 128]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Model
# ─────────────────────────────────────────────────────────────────────────────

class ResNetFiLM(nn.Module):
    """
    Baseline: ResNet + FiLM
    Outputs:
        logit_cls : [B, 1]  — raw logit for binary classification
        pred_stab : [B, 1]  — predicted stability score (pre-sigmoid)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = SharedBackbone(cfg)
        d = self.backbone.out_dim  # 128

        self.cls_head = nn.Linear(d, 1)
        self.reg_head = nn.Linear(d, 1)

        self._init_heads()

    def _init_heads(self):
        for head in [self.cls_head, self.reg_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, image: torch.Tensor,
                task_id: torch.Tensor,
                grasp_point: torch.Tensor):
        z = self.backbone(image, task_id, grasp_point)  # [B, 128]
        return self.cls_head(z), self.reg_head(z)        # logit, pre-sigmoid stab

    def predict(self, image, task_id, grasp_point):
        """Inference-mode forward — returns probabilities."""
        self.eval()
        with torch.no_grad():
            logit_cls, pred_stab = self(image, task_id, grasp_point)
            p_cls  = torch.sigmoid(logit_cls).squeeze()
            p_stab = torch.sigmoid(pred_stab).squeeze()
        return p_cls, p_stab

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
