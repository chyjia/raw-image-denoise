"""Alignment-aware temporal attention front-end for Burst NAFNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nafnet import NAFBlock


def flow_warp(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp BCHW features by pixel-space B2HW flow."""
    batch, _channels, height, width = features.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
        torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
    flow_x = flow[:, 0] * (2.0 / max(width - 1, 1))
    flow_y = flow[:, 1] * (2.0 / max(height - 1, 1))
    grid = base_grid + torch.stack((flow_x, flow_y), dim=-1)
    return F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class TemporalAttentionFusion(nn.Module):
    """Predict local residual flow and confidence for each reference-frame pair."""

    def __init__(self, width: int, max_residual_flow: float = 2.0):
        super().__init__()
        self.max_residual_flow = max_residual_flow
        self.pair_net = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 3, 3, padding=1),
        )
        nn.init.zeros_(self.pair_net[-1].weight)
        nn.init.zeros_(self.pair_net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        reference_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: B,T,C,H,W
        reference = features[:, reference_index]
        aligned = []
        logits = []
        flows = []
        for frame_index in range(features.shape[1]):
            current = features[:, frame_index]
            pair_output = self.pair_net(torch.cat((current, reference), dim=1))
            flow = torch.tanh(pair_output[:, :2]) * self.max_residual_flow
            if frame_index == reference_index:
                flow = torch.zeros_like(flow)
            warped = flow_warp(current, flow)
            # Similarity discourages attention on badly aligned or moving pixels.
            similarity = -(warped - reference).abs().mean(dim=1, keepdim=True)
            logit = pair_output[:, 2:3] + similarity
            aligned.append(warped)
            logits.append(logit)
            flows.append(flow)

        aligned_tensor = torch.stack(aligned, dim=1)
        logit_tensor = torch.stack(logits, dim=1)
        attention = torch.softmax(logit_tensor, dim=1)
        fused = (aligned_tensor * attention).sum(dim=1)
        return fused, attention, torch.stack(flows, dim=1)


class BurstNAFNet(nn.Module):
    """16-frame NAFNet with shared frame encoding and temporal attention."""

    def __init__(
        self,
        frame_channels: int = 3,
        width: int = 48,
        input_frames: int = 16,
        enc_blk_nums: list[int] | None = None,
        middle_blk_num: int = 6,
        dec_blk_nums: list[int] | None = None,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        enc_blk_nums = enc_blk_nums or [2, 2, 2, 2]
        dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
        self.input_frames = input_frames
        self.reference_index = input_frames // 2 - 1
        self.use_checkpoint = use_checkpoint

        self.frame_intro = nn.Conv2d(frame_channels, width, 3, padding=1)
        self.frame_blocks = nn.ModuleList([NAFBlock(width) for _ in range(2)])
        self.temporal_fusion = TemporalAttentionFusion(width)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for block_count in enc_blk_nums:
            self.encoders.append(
                nn.ModuleList([NAFBlock(channels) for _ in range(block_count)])
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = nn.ModuleList(
            [NAFBlock(channels) for _ in range(middle_blk_num)]
        )
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for block_count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.ModuleList([NAFBlock(channels) for _ in range(block_count)])
            )
        self.ending = nn.Conv2d(width, 1, 3, padding=1)

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        features: torch.Tensor,
    ) -> torch.Tensor:
        for block in blocks:
            if self.use_checkpoint and self.training:
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        return features

    def forward(
        self,
        burst: torch.Tensor,
        return_aux: bool = False,
    ):
        if burst.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W burst, got {tuple(burst.shape)}")
        if burst.shape[1] != self.input_frames:
            raise ValueError(
                f"Model expects {self.input_frames} frames, got {burst.shape[1]}"
            )
        batch, frame_count, channels, height, width = burst.shape
        reference_vst = burst[:, self.reference_index, :1]

        features = self.frame_intro(
            burst.reshape(batch * frame_count, channels, height, width)
        )
        features = self._run_blocks(self.frame_blocks, features)
        features = features.reshape(batch, frame_count, -1, height, width)
        fused, attention, flows = self.temporal_fusion(
            features,
            self.reference_index,
        )

        skips = []
        x = fused
        for encoder, down in zip(self.encoders, self.downs):
            x = self._run_blocks(encoder, x)
            skips.append(x)
            x = down(x)
        x = self._run_blocks(self.middle, x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x) + skip
            x = self._run_blocks(decoder, x)
        output = reference_vst - self.ending(x)
        if return_aux:
            return output, {"attention": attention, "flows": flows}
        return output


def build_burst_nafnet(
    width: int = 48,
    input_frames: int = 16,
    use_checkpoint: bool = True,
) -> BurstNAFNet:
    return BurstNAFNet(
        width=width,
        input_frames=input_frames,
        use_checkpoint=use_checkpoint,
    )
