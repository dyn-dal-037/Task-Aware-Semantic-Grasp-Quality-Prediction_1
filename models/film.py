"""
Feature-wise Linear Modulation (FiLM) layer.
Conditions visual feature maps on task embedding via learned affine transform.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM(F) = γ(task) ⊙ F + β(task)
    γ, β are generated from task embedding via learned linear projections.
    """

    def __init__(self, task_embed_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(task_embed_dim, feature_dim)
        self.beta_fc  = nn.Linear(task_embed_dim, feature_dim)

        # Initialize close to identity transform for training stability
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, features: torch.Tensor,
                task_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features   : [B, feature_dim]
            task_embed : [B, task_embed_dim]
        Returns:
            modulated  : [B, feature_dim]
        """
        γ = self.gamma_fc(task_embed)   # [B, feature_dim]
        β = self.beta_fc(task_embed)    # [B, feature_dim]
        return γ * features + β


class GraspPointEncoder(nn.Module):
    """
    Projects normalized (x,y) grasp point into embedding space.
    Fused with visual features before trunk.
    """

    def __init__(self, in_dim: int = 2, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )

    def forward(self, grasp_point: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grasp_point : [B, 2]  normalized to [-1, 1]
        Returns:
            embed       : [B, embed_dim]
        """
        return self.net(grasp_point)
