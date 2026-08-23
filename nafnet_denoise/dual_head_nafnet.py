"""Dual-head NAFNet: flat/edge residual heads blended by a soft edge mask."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .burst_merge import SoftBurstMerge
from .common import center_frame_index, model_input_channels
from .nafnet import NAFBlock, build_bottleneck_nonlocal
from .temporal_fusion import DeformableFrameAlign


def soft_edge_mask(image: torch.Tensor, temperature: float = 8.0) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]
    )
    kernel_y = image.new_tensor(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]
    )
    channels = image.shape[1]
    grad_x = F.conv2d(image, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    grad_y = F.conv2d(image, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
    flat = magnitude.flatten(2)
    lo = torch.quantile(flat, 0.50, dim=-1, keepdim=True).unsqueeze(-1)
    hi = torch.quantile(flat, 0.85, dim=-1, keepdim=True).unsqueeze(-1)
    scaled = (magnitude - lo) / (hi - lo).clamp_min(1e-6)
    return torch.sigmoid(temperature * (scaled - 0.5))


class SigmaFiLM(nn.Module):
    """FFDNet/KPN-style σ conditioning: ``y = γ(σ)·x + β(σ)``.

    Zero-init last layer → identity at start (warm-start safe).
    Expects a per-sample scalar σ in DN (after optional jitter).
    """

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, sigma_dn: torch.Tensor) -> torch.Tensor:
        # sigma_dn: (B,) or (B,1)
        s = sigma_dn.reshape(-1, 1).float()
        # log1p keeps dark/bright σ on a similar scale; +eps for safety.
        cond = torch.log1p(s.clamp_min(0.0))
        gb = self.mlp(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


def image_laplacian(image: torch.Tensor) -> torch.Tensor:
    """3×3 Laplacian of a single-channel image tensor ``(B,1,H,W)``."""
    kernel = image.new_tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]
    )
    return F.conv2d(image, kernel, padding=1)


class LearnableEdgeMask(nn.Module):
    """Predict a residual logit on top of Sobel soft mask (zero-init → Sobel at start)."""

    def __init__(self, width: int):
        super().__init__()
        mid = max(width // 2, 8)
        self.net = nn.Sequential(
            nn.Conv2d(width, mid, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        sobel = soft_edge_mask(image)
        logit = torch.logit(sobel.clamp(1e-4, 1.0 - 1e-4))
        return torch.sigmoid(logit + self.net(features))


class DualHeadNAFNet(nn.Module):
    """Shared NAFNet trunk with separate flat/edge residual heads.

    Final residual = ``edge_map * res_edge + (1 - edge_map) * res_flat``,
    subtracted from the center-frame VST (same contract as ``NAFNet``).
    """

    def __init__(
        self,
        inp_channels: int = 6,
        out_channels: int = 1,
        width: int = 32,
        enc_blk_nums: list[int] | None = None,
        middle_blk_num: int = 6,
        dec_blk_nums: list[int] | None = None,
        image_channel_index: int = 0,
        use_nonlocal: bool = False,
        nonlocal_count: int = 1,
        learnable_mask: bool = False,
        use_deformable: bool = False,
        deform_feat_width: int = 16,
        deform_max_flow: float = 2.0,
        use_burst_merge: bool = False,
        merge_feat_width: int = 16,
        input_frames: int | None = None,
        region_towers: int = 0,
        fusion_edge_harden: float = 0.0,
        use_sigma_film: bool = False,
        use_sigma_film_flat: bool = False,
        use_lap_edge: bool = False,
        use_lap_edge_ms: bool = False,
    ):
        super().__init__()
        enc_blk_nums = enc_blk_nums or [2, 2, 2, 2]
        dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
        self.image_channel_index = image_channel_index
        self.width = width
        self.use_nonlocal = use_nonlocal
        self.nonlocal_count = nonlocal_count
        self.learnable_mask = learnable_mask
        self.use_deformable = use_deformable
        self.deform_feat_width = deform_feat_width
        self.deform_max_flow = deform_max_flow
        self.use_burst_merge = use_burst_merge
        self.merge_feat_width = merge_feat_width
        self.input_frames = int(input_frames) if input_frames is not None else None
        self.region_towers = int(max(region_towers, 0))
        self.fusion_edge_harden = float(max(fusion_edge_harden, 0.0))
        self.use_sigma_film = bool(use_sigma_film)
        self.use_sigma_film_flat = bool(use_sigma_film_flat)
        self.use_lap_edge = bool(use_lap_edge)
        self.use_lap_edge_ms = bool(use_lap_edge_ms) and self.use_lap_edge

        if use_deformable:
            if input_frames is None:
                raise ValueError("input_frames is required when use_deformable=True")
            self.deform_align = DeformableFrameAlign(
                input_frames=input_frames,
                feat_width=deform_feat_width,
                max_residual_flow=deform_max_flow,
                reference_index=image_channel_index,
            )
        else:
            self.deform_align = None

        if use_burst_merge:
            if input_frames is None:
                raise ValueError("input_frames is required when use_burst_merge=True")
            self.burst_merge = SoftBurstMerge(
                input_frames=input_frames,
                feat_width=merge_feat_width,
                reference_index=image_channel_index,
            )
        else:
            self.burst_merge = None

        self.intro = nn.Conv2d(inp_channels, width, 3, 1, 1)
        self.ending_flat = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.ending_edge = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.mask_head = LearnableEdgeMask(width) if learnable_mask else None
        self.sigma_film = SigmaFiLM(width) if self.use_sigma_film else None
        # P2-b: FiLM only on flat tower (zero-init → identity).
        self.sigma_film_flat = (
            SigmaFiLM(width) if self.use_sigma_film_flat else None
        )
        # P2-c: Laplacian → edge features (zero-init 1×1 → no change at start).
        if self.use_lap_edge:
            self.lap_to_edge = nn.Conv2d(1, width, 1, 1, 0)
            nn.init.zeros_(self.lap_to_edge.weight)
            if self.lap_to_edge.bias is not None:
                nn.init.zeros_(self.lap_to_edge.bias)
            # P3-a: half-resolution Laplacian branch (also zero-init).
            if self.use_lap_edge_ms:
                self.lap_to_edge_ms = nn.Conv2d(1, width, 1, 1, 0)
                nn.init.zeros_(self.lap_to_edge_ms.weight)
                if self.lap_to_edge_ms.bias is not None:
                    nn.init.zeros_(self.lap_to_edge_ms.bias)
            else:
                self.lap_to_edge_ms = None
        else:
            self.lap_to_edge = None
            self.lap_to_edge_ms = None
        # Region Adaptive Denoise-style towers (NAF β/γ=0 → identity at init).
        if self.region_towers > 0:
            self.tower_flat = nn.Sequential(
                *[NAFBlock(width) for _ in range(self.region_towers)]
            )
            self.tower_edge = nn.Sequential(
                *[NAFBlock(width) for _ in range(self.region_towers)]
            )
        else:
            self.tower_flat = nn.Identity()
            self.tower_edge = nn.Identity()

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

    def encode_decode(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deform_align is not None:
            inp = self.deform_align(inp)
        if self.burst_merge is not None:
            n_frames = self.input_frames
            assert n_frames is not None
            frames = inp[:, :n_frames]
            extras = inp[:, n_frames:]
            merged = self.burst_merge(frames)
            # Replace only the center/reference channel so zero-init gates → identity.
            frames = frames.clone()
            frames[:, self.image_channel_index : self.image_channel_index + 1] = merged
            inp = torch.cat([frames, extras], dim=1)
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
        return image, x

    def _sigma_scalar_from_input(self, inp: torch.Tensor) -> torch.Tensor:
        """Mean of the σ channel (layout: frames | sigma | exposure)."""
        n_frames = int(self.input_frames or 0)
        if n_frames <= 0 or inp.shape[1] <= n_frames:
            return inp.new_ones(inp.shape[0])
        sigma_map = inp[:, n_frames : n_frames + 1]
        return sigma_map.flatten(1).mean(dim=1)

    def forward(self, inp: torch.Tensor, return_aux: bool = False):
        image, features = self.encode_decode(inp)
        sigma_scalar = None
        if self.sigma_film is not None or self.sigma_film_flat is not None:
            sigma_scalar = self._sigma_scalar_from_input(inp)
        if self.sigma_film is not None:
            features = self.sigma_film(features, sigma_scalar)
        feat_flat = self.tower_flat(features)
        if self.sigma_film_flat is not None:
            feat_flat = self.sigma_film_flat(feat_flat, sigma_scalar)
        feat_edge = self.tower_edge(features)
        if self.lap_to_edge is not None:
            feat_edge = feat_edge + self.lap_to_edge(image_laplacian(image))
            if self.lap_to_edge_ms is not None:
                half = F.avg_pool2d(image, kernel_size=2, stride=2)
                lap_ms = image_laplacian(half)
                lap_ms = F.interpolate(
                    lap_ms,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                feat_edge = feat_edge + self.lap_to_edge_ms(lap_ms)
        res_flat = self.ending_flat(feat_flat)
        res_edge = self.ending_edge(feat_edge)
        if self.mask_head is not None:
            edge_map = self.mask_head(features, image)
        else:
            edge_map = soft_edge_mask(image)
        harden = float(getattr(self, "fusion_edge_harden", 0.0) or 0.0)
        if harden > 1.0:
            # Push soft mask toward 0/1 so heads specialize (DRA-style).
            edge_map = edge_map.clamp(1e-4, 1.0 - 1e-4)
            logit = torch.logit(edge_map)
            edge_map = torch.sigmoid(harden * logit)
        flat_map = 1.0 - edge_map
        residual = edge_map * res_edge + flat_map * res_flat
        output = image - residual
        if return_aux:
            return output, {
                "pred_flat": image - res_flat,
                "pred_edge": image - res_edge,
                "edge_map": edge_map,
                "flat_map": flat_map,
            }
        return output


def build_dual_head_nafnet(
    width: int = 32,
    input_frames: int = 4,
    use_exposure: bool = True,
    use_sigma: bool = True,
    use_nonlocal: bool = False,
    nonlocal_count: int = 1,
    learnable_mask: bool = False,
    use_deformable: bool = False,
    deform_feat_width: int = 16,
    deform_max_flow: float = 2.0,
    use_burst_merge: bool = False,
    merge_feat_width: int = 16,
    region_towers: int = 0,
    fusion_edge_harden: float = 0.0,
    use_sigma_film: bool = False,
    use_sigma_film_flat: bool = False,
    use_lap_edge: bool = False,
    use_lap_edge_ms: bool = False,
) -> DualHeadNAFNet:
    center = center_frame_index(input_frames)
    inp_channels = model_input_channels(
        input_frames,
        use_exposure=use_exposure,
        use_sigma=use_sigma,
    )
    return DualHeadNAFNet(
        inp_channels=inp_channels,
        out_channels=1,
        width=width,
        image_channel_index=center,
        use_nonlocal=use_nonlocal,
        nonlocal_count=nonlocal_count,
        learnable_mask=learnable_mask,
        use_deformable=use_deformable,
        deform_feat_width=deform_feat_width,
        deform_max_flow=deform_max_flow,
        use_burst_merge=use_burst_merge,
        merge_feat_width=merge_feat_width,
        input_frames=input_frames,
        region_towers=region_towers,
        fusion_edge_harden=fusion_edge_harden,
        use_sigma_film=use_sigma_film,
        use_sigma_film_flat=use_sigma_film_flat,
        use_lap_edge=use_lap_edge,
        use_lap_edge_ms=use_lap_edge_ms,
    )


def copy_edge_head_weights(
    student: DualHeadNAFNet,
    teacher_state: dict[str, torch.Tensor],
) -> int:
    """Copy ending_edge (+ tower_edge if present) from a DualHead teacher."""
    own = student.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in teacher_state.items():
        if not (key.startswith("ending_edge.") or key.startswith("tower_edge.")):
            continue
        if key in own and own[key].shape == value.shape:
            matched[key] = value
    if matched:
        student.load_state_dict(matched, strict=False)
    return len(matched)


def load_dual_head_from_nafnet_state(
    model: DualHeadNAFNet,
    state: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Load a single-head NAFNet checkpoint into the dual-head model."""
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state.items():
        if key == "ending.weight":
            matched["ending_flat.weight"] = value
            matched["ending_edge.weight"] = value.clone()
            continue
        if key == "ending.bias":
            matched["ending_flat.bias"] = value
            matched["ending_edge.bias"] = value.clone()
            continue
        if key in target and target[key].shape == value.shape:
            matched[key] = value
        else:
            skipped += 1
    model.load_state_dict(matched, strict=False)
    return len(matched), skipped
