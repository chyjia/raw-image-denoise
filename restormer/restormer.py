"""Full Restormer architecture (PyTorch) adapted for conditional Mono10 denoising.

Reference: Zamir et al., "Restormer: Efficient Transformer for High-Resolution
Image Restoration", CVPR 2022. This implementation keeps the full multi-scale
encoder-decoder with MDTA attention and GDFN feed-forward, adds per-block
gradient checkpointing for 8 GB GPUs, and predicts a residual over the noisy
input channel so it can accept an extra exposure-condition channel.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layernorm_type: str) -> None:
        super().__init__()
        if layernorm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv feed-forward network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    """Multi-Dconv head transposed attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layernorm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layernorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
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
        layernorm_type: str = "WithBias",
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        def stage(level_dim: int, count: int, head: int) -> nn.ModuleList:
            return nn.ModuleList(
                TransformerBlock(
                    level_dim, head, ffn_expansion_factor, bias, layernorm_type
                )
                for _ in range(count)
            )

        self.encoder_level1 = stage(dim, num_blocks[0], heads[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = stage(dim * 2, num_blocks[1], heads[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = stage(dim * 4, num_blocks[2], heads[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = stage(dim * 8, num_blocks[3], heads[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = stage(dim * 4, num_blocks[2], heads[2])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = stage(dim * 2, num_blocks[1], heads[1])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = stage(dim * 2, num_blocks[0], heads[0])

        self.refinement = stage(dim * 2, num_refinement_blocks, heads[0])
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        base = inp[:, :1]  # noisy channel used as the residual base

        enc1 = self._run(self.encoder_level1, self.patch_embed(inp))
        enc2 = self._run(self.encoder_level2, self.down1_2(enc1))
        enc3 = self._run(self.encoder_level3, self.down2_3(enc2))
        latent = self._run(self.latent, self.down3_4(enc3))

        dec3 = self.up4_3(latent)
        dec3 = self.reduce_chan_level3(torch.cat([dec3, enc3], dim=1))
        dec3 = self._run(self.decoder_level3, dec3)

        dec2 = self.up3_2(dec3)
        dec2 = self.reduce_chan_level2(torch.cat([dec2, enc2], dim=1))
        dec2 = self._run(self.decoder_level2, dec2)

        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self._run(self.decoder_level1, dec1)

        refined = self._run(self.refinement, dec1)
        return self.output(refined) + base
