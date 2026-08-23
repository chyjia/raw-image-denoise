"""16-frame BurstRestormer: temporal attention fusion + Restormer backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Allow importing the sibling ``restormer`` package from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from restormer.restormer import Restormer  # noqa: E402

from .nafnet import NAFBlock
from .temporal_fusion import TemporalAttentionFusion


class BurstRestormer(nn.Module):
    """Align-aware burst front-end feeding a conditional Restormer denoiser.

    Burst layout matches BurstNAFNet: ``B,T,3,H,W`` = VST, PTC sigma, exposure.
    Restormer input is ``[ref_vst, sigma_ref, exposure_ref, fused_1ch]``.
    """

    def __init__(
        self,
        frame_channels: int = 3,
        width: int = 48,
        input_frames: int = 16,
        dim: int = 48,
        num_blocks: tuple[int, ...] = (2, 3, 3, 4),
        num_refinement_blocks: int = 2,
        heads: tuple[int, ...] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.reference_index = input_frames // 2 - 1
        self.width = width
        self.dim = dim
        self.num_blocks = tuple(num_blocks)
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = tuple(heads)
        self.use_checkpoint = use_checkpoint

        self.frame_intro = nn.Conv2d(frame_channels, width, 3, padding=1)
        self.frame_blocks = nn.ModuleList([NAFBlock(width) for _ in range(2)])
        self.temporal_fusion = TemporalAttentionFusion(width)
        self.fused_proj = nn.Conv2d(width, 1, 1)

        self.restormer = Restormer(
            inp_channels=4,
            out_channels=1,
            dim=dim,
            num_blocks=self.num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=self.heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=False,
            layernorm_type="WithBias",
            use_checkpoint=use_checkpoint,
        )

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
        reference = burst[:, self.reference_index]
        reference_vst = reference[:, :1]
        reference_sigma = reference[:, 1:2]
        reference_exposure = reference[:, 2:3]

        features = self.frame_intro(
            burst.reshape(batch * frame_count, channels, height, width)
        )
        features = self._run_blocks(self.frame_blocks, features)
        features = features.reshape(batch, frame_count, -1, height, width)
        fused, attention, flows = self.temporal_fusion(
            features,
            self.reference_index,
        )
        fused_1ch = self.fused_proj(fused)
        restormer_input = torch.cat(
            [reference_vst, reference_sigma, reference_exposure, fused_1ch],
            dim=1,
        )
        output = self.restormer(restormer_input)
        if return_aux:
            return output, {"attention": attention, "flows": flows}
        return output


def build_burst_restormer(
    width: int = 48,
    input_frames: int = 16,
    dim: int = 48,
    num_blocks: tuple[int, ...] = (2, 3, 3, 4),
    num_refinement_blocks: int = 2,
    heads: tuple[int, ...] = (1, 2, 4, 8),
    use_checkpoint: bool = True,
) -> BurstRestormer:
    return BurstRestormer(
        width=width,
        input_frames=input_frames,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        use_checkpoint=use_checkpoint,
    )


def load_restormer_weights_partial(
    model: BurstRestormer,
    checkpoint: Path,
    device: torch.device | None = None,
) -> tuple[int, int]:
    """Load shape-matching Restormer backbone weights; skip stem/fusion/embed mismatches."""
    import pathlib

    # Checkpoints may have been pickled on Linux.
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

    map_location = device or "cpu"
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    source = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    target = model.restormer.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in source.items():
        # Strip optional prefixes.
        bare = key
        for prefix in ("module.", "restormer.", "model."):
            if bare.startswith(prefix):
                bare = bare[len(prefix) :]
        if bare not in target:
            skipped += 1
            continue
        if target[bare].shape != value.shape:
            skipped += 1
            continue
        matched[bare] = value
    model.restormer.load_state_dict(matched, strict=False)
    return len(matched), skipped


def load_burst_restormer_weights_partial(
    model: BurstRestormer,
    checkpoint: Path,
    device: torch.device | None = None,
) -> tuple[int, int]:
    """Load any shape-matching tensors from another BurstRestormer checkpoint.

    Useful when changing ``input_frames`` (e.g. 16f → 4f): stem / fusion /
    Restormer weights transfer; only T-dependent buffers are skipped if any.
    """
    map_location = device or "cpu"
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    source = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in source.items():
        if key not in target or target[key].shape != value.shape:
            skipped += 1
            continue
        matched[key] = value
    model.load_state_dict(matched, strict=False)
    return len(matched), skipped
