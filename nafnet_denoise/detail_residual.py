"""Detail residual head that cleans flats toward BM3D without blurring edges."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import center_frame_index


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.net(inp)


class DetailResidualNet(nn.Module):
    """Predict a VST residual added to a frozen base denoiser.

    ``depth=0`` keeps the original shallow conv stack. ``depth>=1`` builds a
    U-Net with that many downsample levels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 16,
        depth: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.depth = depth

        if depth <= 0:
            self.encoder = None
            self.pool = None
            self.bottleneck = None
            self.up = None
            self.decoder = None
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(width, 1, 3, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            return

        channels = [width * (2**level) for level in range(depth + 1)]
        self.encoder = nn.ModuleList()
        prev = in_channels
        for channel in channels[:-1]:
            self.encoder.append(_ConvBlock(prev, channel))
            prev = channel
        self.pool = nn.ModuleList(
            [nn.AvgPool2d(2) for _ in range(depth)]
        )
        self.bottleneck = _ConvBlock(channels[-2], channels[-1])
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[level + 1], channels[level], 2, 2)
                for level in range(depth - 1, -1, -1)
            ]
        )
        self.decoder = nn.ModuleList()
        for level in range(depth - 1, -1, -1):
            self.decoder.append(_ConvBlock(channels[level] * 2, channels[level]))
        self.head = nn.Conv2d(channels[0], 1, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.net = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.depth <= 0:
            assert self.net is not None
            return self.net(inp)

        skips: list[torch.Tensor] = []
        x = inp
        assert self.encoder is not None and self.pool is not None
        for block, pool in zip(self.encoder, self.pool):
            x = block(x)
            skips.append(x)
            x = pool(x)
        assert self.bottleneck is not None and self.up is not None
        assert self.decoder is not None
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def residual_conditioning_input(
    model_input: torch.Tensor,
    base_pred: torch.Tensor,
    input_frames: int,
) -> torch.Tensor:
    """Build ``[noisy_center, base_pred, sigma_or_ones, exposure]`` for the head.

    Works for both sigma-conditioned (N+2 channels) and legacy (N+1) base inputs.
    """
    center = center_frame_index(input_frames)
    noisy = model_input[:, center : center + 1]
    channels = int(model_input.shape[1])
    if channels == input_frames + 2:
        sigma = model_input[:, input_frames : input_frames + 1]
        exposure = model_input[:, input_frames + 1 : input_frames + 2]
    elif channels == input_frames + 1:
        sigma = torch.ones_like(noisy)
        exposure = model_input[:, input_frames : input_frames + 1]
    else:
        raise ValueError(
            f"Unexpected model input channels={channels} for input_frames={input_frames}"
        )
    return torch.cat([noisy, base_pred, sigma, exposure], dim=1)


def load_detail_residual(
    checkpoint: Path,
    device: torch.device,
    width: int | None = None,
    depth: int | None = None,
) -> DetailResidualNet:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        saved_args = payload.get("args", {}) or {}
        if width is None:
            width = int(saved_args.get("residual_width", 16))
        if depth is None:
            depth = int(saved_args.get("residual_depth", 0))
        state = payload["model"]
    else:
        if width is None:
            width = 16
        if depth is None:
            depth = 0
        state = payload
    model = DetailResidualNet(width=width, depth=depth)
    model.load_state_dict(state)
    return model.eval().to(device)
