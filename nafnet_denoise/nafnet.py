"""NAFNet architecture for conditional VST-domain Mono10 denoising.

Adapted from NAFNet (Chen et al., ECCV 2022). The network accepts a stack of
normalized VST frames plus an exposure-conditioning channel and predicts a
residual subtracted from the center-frame VST channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import center_frame_index, model_input_channels


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1, groups=dw_channels)
        self.conv3 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1, 1, 0),
        )
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(dw_channels // 2, channels, 1, 1, 0)
        ffn_channels = channels * ffn_expand
        self.conv5 = nn.Conv2d(channels, ffn_channels, 1, 1, 0)
        self.conv6 = nn.Conv2d(ffn_channels // 2, channels, 1, 1, 0)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv4(x)
        x = self.beta * x + identity

        identity = x
        x = self.norm2(x)
        x = self.conv5(x)
        x = self.sg(x)
        x = self.conv6(x)
        return x * self.gamma + identity


class NAFNet(nn.Module):
    def __init__(
        self,
        inp_channels: int = 2,
        out_channels: int = 1,
        width: int = 32,
        enc_blk_nums: list[int] | None = None,
        middle_blk_num: int = 6,
        dec_blk_nums: list[int] | None = None,
        image_channel_index: int = 0,
    ):
        super().__init__()
        enc_blk_nums = enc_blk_nums or [2, 2, 2, 2]
        dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
        self.image_channel_index = image_channel_index

        self.intro = nn.Conv2d(inp_channels, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
            channels *= 2

        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(num)])
            )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        image = inp[:, self.image_channel_index : self.image_channel_index + 1, :, :]
        x = self.intro(inp)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)
        residual = self.ending(x)
        return image - residual


def build_nafnet(
    width: int = 32,
    input_frames: int = 4,
    use_exposure: bool = True,
) -> NAFNet:
    center = center_frame_index(input_frames)
    inp_channels = model_input_channels(input_frames, use_exposure)
    return NAFNet(
        inp_channels=inp_channels,
        out_channels=1,
        width=width,
        image_channel_index=center,
    )
