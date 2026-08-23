"""Full Restormer architecture adapted for conditional VST-domain denoising.

Ported from the official Restormer (Zamir et al., CVPR 2022, BSD-3-Clause) with
two adaptations: the input embedding accepts a 2-channel tensor (normalized VST
image plus an exposure-condition channel) and the network predicts a residual
that is subtracted from the noisy VST image. Optional gradient checkpointing on
every transformer block keeps activation memory within an 8 GB budget.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, bias):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """Conditional Restormer that predicts a residual in the normalized VST domain."""

    def __init__(
        self,
        inp_channels: int = 2,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: tuple[int, ...] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: tuple[int, ...] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        image_channel_index: int = 0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.image_channel_index = image_channel_index

        def block(dim_, head_):
            return TransformerBlock(dim_, head_, ffn_expansion_factor, bias, layer_norm_type)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        self.encoder_level1 = nn.ModuleList([block(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.ModuleList([block(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.ModuleList([block(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.ModuleList([block(dim * 8, heads[3]) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([block(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([block(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.ModuleList([block(dim * 2, heads[0]) for _ in range(num_blocks[0])])

        self.refinement = nn.ModuleList([block(dim * 2, heads[0]) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _run(self, blocks: nn.ModuleList, x):
        for layer in blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def forward(self, inp):
        image = inp[:, self.image_channel_index : self.image_channel_index + 1, :, :]

        x1 = self.patch_embed(inp)
        x1 = self._run(self.encoder_level1, x1)
        x2 = self._run(self.encoder_level2, self.down1_2(x1))
        x3 = self._run(self.encoder_level3, self.down2_3(x2))
        latent = self._run(self.latent, self.down3_4(x3))

        d3 = self.up4_3(latent)
        d3 = self.reduce_chan_level3(torch.cat([d3, x3], dim=1))
        d3 = self._run(self.decoder_level3, d3)

        d2 = self.up3_2(d3)
        d2 = self.reduce_chan_level2(torch.cat([d2, x2], dim=1))
        d2 = self._run(self.decoder_level2, d2)

        d1 = self.up2_1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self._run(self.decoder_level1, d1)
        d1 = self._run(self.refinement, d1)

        residual = self.output(d1)
        return image - residual


def build_restormer(use_checkpoint: bool = True) -> Restormer:
    return Restormer(use_checkpoint=use_checkpoint)
