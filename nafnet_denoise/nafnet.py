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


class NonLocalBlock(nn.Module):
    """Embedded-Gaussian non-local block with zero-init residual (warm-start safe).

    Lives at the NAFNet bottleneck so a 256 patch maps to ~16×16 tokens.
    When spatial tokens exceed ``max_tokens``, ``phi``/``g`` are pooled.
    """

    def __init__(
        self,
        channels: int,
        inter_channels: int | None = None,
        max_tokens: int = 1024,
    ):
        super().__init__()
        inter = inter_channels if inter_channels is not None else max(channels // 4, 32)
        self.max_tokens = max_tokens
        self.norm = LayerNorm2d(channels)
        self.theta = nn.Conv2d(channels, inter, 1, 1, 0)
        self.phi = nn.Conv2d(channels, inter, 1, 1, 0)
        self.g = nn.Conv2d(channels, inter, 1, 1, 0)
        self.proj = nn.Conv2d(inter, channels, 1, 1, 0)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        batch, _channels, height, width = x.shape
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)
        tokens = height * width
        if tokens > self.max_tokens:
            scale = int((tokens / self.max_tokens) ** 0.5 + 0.999)
            scale = max(scale, 2)
            phi = F.avg_pool2d(phi, kernel_size=scale, stride=scale)
            g = F.avg_pool2d(g, kernel_size=scale, stride=scale)
        theta_flat = theta.flatten(2)  # B, C', N
        phi_flat = phi.flatten(2)  # B, C', M
        g_flat = g.flatten(2)  # B, C', M
        attn_scale = theta_flat.shape[1] ** -0.5
        attn = torch.bmm(theta_flat.transpose(1, 2), phi_flat) * attn_scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(g_flat, attn.transpose(1, 2))  # B, C', N
        out = out.view(batch, -1, height, width)
        return identity + self.gamma * self.proj(out)


def build_bottleneck_nonlocal(
    channels: int,
    nonlocal_count: int = 1,
) -> nn.Module:
    if nonlocal_count <= 0:
        return nn.Identity()
    return nn.Sequential(*[NonLocalBlock(channels) for _ in range(nonlocal_count)])


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
        use_nonlocal: bool = False,
        nonlocal_count: int = 1,
    ):
        super().__init__()
        enc_blk_nums = enc_blk_nums or [2, 2, 2, 2]
        dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
        self.image_channel_index = image_channel_index
        self.use_nonlocal = use_nonlocal
        self.nonlocal_count = nonlocal_count

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
        self.bottleneck_nl = (
            build_bottleneck_nonlocal(channels, nonlocal_count)
            if use_nonlocal
            else nn.Identity()
        )

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
        x = self.bottleneck_nl(x)
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
    use_sigma: bool = False,
    enc_blk_nums: list[int] | None = None,
    middle_blk_num: int = 6,
    dec_blk_nums: list[int] | None = None,
    use_nonlocal: bool = False,
    nonlocal_count: int = 1,
) -> NAFNet:
    center = center_frame_index(input_frames)
    inp_channels = model_input_channels(
        input_frames,
        use_exposure=use_exposure,
        use_sigma=use_sigma,
    )
    return NAFNet(
        inp_channels=inp_channels,
        out_channels=1,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
        image_channel_index=center,
        use_nonlocal=use_nonlocal,
        nonlocal_count=nonlocal_count,
    )


def expand_state_dict_width(
    source: dict[str, torch.Tensor],
    target_model: NAFNet,
) -> dict[str, torch.Tensor]:
    """Copy / zero-pad tensors so a narrower NAFNet can warm-start a wider one."""
    target = target_model.state_dict()
    expanded: dict[str, torch.Tensor] = {}
    for key, dst in target.items():
        if key not in source:
            expanded[key] = dst
            continue
        src = source[key]
        if src.shape == dst.shape:
            expanded[key] = src
            continue
        if src.ndim != dst.ndim:
            expanded[key] = dst
            continue
        out = dst.new_zeros(dst.shape)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, dst.shape))
        out[slices] = src[slices]
        expanded[key] = out
    return expanded
