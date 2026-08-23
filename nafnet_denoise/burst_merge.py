"""Learnable soft burst merge (HDR+/KPN-inspired, zero-init → center frame)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftBurstMerge(nn.Module):
    """Blend aligned VST neighbors into the center with residual gates.

    ``out = center + Σ_i gate_i * (frame_i - center)`` with ``gate`` zero-init,
    so the module starts as identity (center only) and learns multi-frame merge.
    """

    def __init__(self, input_frames: int, feat_width: int = 16, reference_index: int | None = None):
        super().__init__()
        self.input_frames = int(input_frames)
        if reference_index is None:
            self.reference_index = 0 if self.input_frames == 1 else self.input_frames // 2 - 1
        else:
            self.reference_index = int(reference_index)
        mid = max(int(feat_width), 8)
        # Per-neighbor gate from concat(center, neighbor) → 1 channel in [0,1].
        self.gate_net = nn.Sequential(
            nn.Conv2d(2, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 3, padding=1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        # sigmoid(bias)≈0 so warm-start keeps the center frame.
        nn.init.constant_(self.gate_net[-1].bias, -6.0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Merge ``B,N,H,W`` or ``B,N*1,H,W`` stacked VST frames → ``B,1,H,W``."""
        if frames.ndim == 4 and frames.shape[1] == self.input_frames:
            # B,N,H,W already; keep as is via unfold channels
            burst = frames.unsqueeze(2)  # B,N,1,H,W
        elif frames.ndim == 4:
            burst = frames.reshape(frames.shape[0], self.input_frames, 1, frames.shape[2], frames.shape[3])
        else:
            raise ValueError(f"Expected BNHW or B,N,H,W-compatible, got {tuple(frames.shape)}")
        center = burst[:, self.reference_index]
        merged = center
        for index in range(self.input_frames):
            if index == self.reference_index:
                continue
            neighbor = burst[:, index]
            gate = torch.sigmoid(self.gate_net(torch.cat((center, neighbor), dim=1)))
            merged = merged + gate * (neighbor - center)
        return merged


class TemporalMeanMerge(nn.Module):
    """Non-learned mean merge (baseline / teacher helper)."""

    def __init__(self, input_frames: int, reference_index: int | None = None):
        super().__init__()
        self.input_frames = int(input_frames)
        self.reference_index = (
            0 if input_frames == 1 else input_frames // 2 - 1
        ) if reference_index is None else int(reference_index)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            burst = frames.reshape(frames.shape[0], self.input_frames, 1, frames.shape[2], frames.shape[3])
        else:
            burst = frames
        return burst.mean(dim=1)
