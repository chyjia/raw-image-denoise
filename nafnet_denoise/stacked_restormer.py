"""Stacked multi-frame Restormer spatial head (no temporal attention)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from restormer.restormer import Restormer  # noqa: E402

from .common import center_frame_index, model_input_channels


class StackedRestormer(nn.Module):
    """Denoise with Restormer on stacked VST frames (+ optional sigma/exposure).

    Channel 0 fed to Restormer is always the center-frame VST so the additive
    residual matches the NAFNet center-frame contract used by ``denoise_frame``.
    """

    def __init__(
        self,
        input_frames: int = 4,
        use_sigma: bool = True,
        use_exposure: bool = True,
        dim: int = 48,
        num_blocks: tuple[int, ...] = (2, 3, 3, 4),
        num_refinement_blocks: int = 2,
        heads: tuple[int, ...] = (1, 2, 4, 8),
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.use_sigma = use_sigma
        self.use_exposure = use_exposure
        self.image_channel_index = center_frame_index(input_frames)
        self.dim = dim
        self.num_blocks = tuple(num_blocks)
        self.num_refinement_blocks = num_refinement_blocks
        self.heads = tuple(heads)
        inp_channels = model_input_channels(
            input_frames,
            use_exposure=use_exposure,
            use_sigma=use_sigma,
        )
        self.restormer = Restormer(
            inp_channels=inp_channels,
            out_channels=1,
            dim=dim,
            num_blocks=self.num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=self.heads,
            use_checkpoint=use_checkpoint,
        )

    def _center_first(self, inp: torch.Tensor) -> torch.Tensor:
        center = self.image_channel_index
        if center == 0:
            return inp
        pieces = [inp[:, center : center + 1]]
        for index in range(inp.shape[1]):
            if index == center:
                continue
            pieces.append(inp[:, index : index + 1])
        return torch.cat(pieces, dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.restormer(self._center_first(inp))


def build_stacked_restormer(
    input_frames: int = 4,
    use_sigma: bool = True,
    dim: int = 48,
    num_blocks: tuple[int, ...] = (2, 3, 3, 4),
    num_refinement_blocks: int = 2,
    heads: tuple[int, ...] = (1, 2, 4, 8),
    use_checkpoint: bool = True,
) -> StackedRestormer:
    return StackedRestormer(
        input_frames=input_frames,
        use_sigma=use_sigma,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        use_checkpoint=use_checkpoint,
    )


def load_stacked_restormer_partial(
    model: StackedRestormer,
    checkpoint: Path,
    device: torch.device | None = None,
) -> tuple[int, int]:
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    payload = torch.load(checkpoint, map_location=device or "cpu", weights_only=False)
    source = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    target = model.restormer.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in source.items():
        bare = key
        for prefix in ("module.", "restormer.", "model."):
            if bare.startswith(prefix):
                bare = bare[len(prefix) :]
        if bare not in target or target[bare].shape != value.shape:
            skipped += 1
            continue
        matched[bare] = value
    model.restormer.load_state_dict(matched, strict=False)
    return len(matched), skipped
