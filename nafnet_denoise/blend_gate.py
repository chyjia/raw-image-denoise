"""Learnable soft gate for SOTA↔edge DualHead fusion (P3-b Spatial MoE-lite).

Zero-init residual on Sobel soft mask → identity with fixed T at start.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_head_nafnet import soft_edge_mask


class BlendGate(nn.Module):
    """Predict ``edge_map`` from a guide DN image (Sobel + residual CNN)."""

    def __init__(self, mid: int = 16, temperature: float = 8.0):
        super().__init__()
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, guide_dn: torch.Tensor) -> torch.Tensor:
        sobel = soft_edge_mask(guide_dn, temperature=self.temperature)
        logit = torch.logit(sobel.clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(logit + self.net(guide_dn))


def blend_with_gate(
    sota_dn: torch.Tensor,
    edge_dn: torch.Tensor,
    edge_map: torch.Tensor,
) -> torch.Tensor:
    return edge_map * edge_dn + (1.0 - edge_map) * sota_dn
