"""FPS-aware Mono10 denoising with alignment-confidence fallback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .common import center_frame_index, decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .detail_residual import load_detail_residual
from .infer import denoise_frame, load_model, save_mono10_png
from .infer_burst import (
    alignment_confidence,
    denoise_burst,
    load_burst_model,
    prepare_aligned_burst,
)


def frames_for_fps(fps: float, max_span_s: float = 0.25) -> int:
    """Always prefer distilled 4f; fall back to 1f only when alignment is poor.

    ``fps`` / ``max_span_s`` are kept for call-site compatibility but ignored:
    high-FPS clips also stay on 4f after BM3D-teacher distillation.
    """
    _ = (fps, max_span_s)
    return 4


@dataclass
class AdaptiveModels:
    model_1f: torch.nn.Module
    model_4f: torch.nn.Module
    model_16f: torch.nn.Module
    frames_1f: int
    frames_4f: int
    frames_16f: int
    residual_4f: torch.nn.Module | None = None


def load_adaptive_models(
    checkpoint_1f: Path,
    checkpoint_4f: Path,
    checkpoint_16f: Path,
    device: torch.device,
    checkpoint_residual_4f: Path | None = None,
) -> AdaptiveModels:
    model_1f, frames_1f, _ = load_model(checkpoint_1f, None, device)
    model_4f, frames_4f, _ = load_model(checkpoint_4f, None, device)
    model_16f, _width, frames_16f = load_burst_model(checkpoint_16f, device)
    residual_4f = None
    if checkpoint_residual_4f is not None:
        residual_4f = load_detail_residual(checkpoint_residual_4f, device)
    if frames_1f != 1:
        raise ValueError(f"Expected a 1-frame checkpoint, got {frames_1f}: {checkpoint_1f}")
    if frames_4f != 4:
        raise ValueError(f"Expected a 4-frame checkpoint, got {frames_4f}: {checkpoint_4f}")
    if frames_16f != 16:
        raise ValueError(f"Expected a 16-frame checkpoint, got {frames_16f}: {checkpoint_16f}")
    return AdaptiveModels(
        model_1f=model_1f,
        model_4f=model_4f,
        model_16f=model_16f,
        frames_1f=frames_1f,
        frames_4f=frames_4f,
        frames_16f=frames_16f,
        residual_4f=residual_4f,
    )


def _confidence_for_window(
    frames: np.memmap,
    target_index: int,
    input_frames: int,
    threshold: float,
) -> tuple[list[np.ndarray], np.ndarray, float, float]:
    aligned, responses = prepare_aligned_burst(
        frames,
        target_index,
        input_frames,
        min_response=threshold,
        reject_unreliable=True,
        return_responses=True,
    )
    reference_index = center_frame_index(input_frames)
    confidence = alignment_confidence(responses, reference_index)
    companion_mask = np.ones(input_frames, dtype=bool)
    companion_mask[reference_index] = False
    reliable_fraction = float(np.mean(responses[companion_mask] >= threshold))
    return aligned, responses, confidence, reliable_fraction


@torch.no_grad()
def denoise_adaptive(
    models: AdaptiveModels,
    frames: np.memmap,
    target_index: int,
    fps: float,
    exposure_ms: float,
    device: torch.device,
    align_threshold: float = 0.03,
    min_reliable_fraction: float = 0.60,
    tile_size: int = 256,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    requested = frames_for_fps(fps)
    confidence = 1.0
    reliable_fraction = 1.0

    if requested == 16:
        aligned, _responses, confidence, reliable_fraction = _confidence_for_window(
            frames,
            target_index,
            16,
            align_threshold,
        )
        if confidence >= align_threshold and reliable_fraction >= min_reliable_fraction:
            output = denoise_burst(
                models.model_16f,
                aligned,
                exposure_ms,
                device,
                tile_size=tile_size,
            )
            return output, {
                "requested_frames": requested,
                "used_frames": 16,
                "alignment_confidence": confidence,
                "reliable_fraction": reliable_fraction,
                "fallback_used": False,
                "route": "16f",
            }
        requested = 4

    if requested == 4:
        _aligned, _responses, confidence, reliable_fraction = _confidence_for_window(
            frames,
            target_index,
            4,
            align_threshold,
        )
        if confidence >= align_threshold and reliable_fraction >= min_reliable_fraction:
            output = denoise_frame(
                models.model_4f,
                frames,
                target_index,
                exposure_ms,
                device,
                input_frames=4,
                tile=tile_size,
                residual_model=models.residual_4f,
            )
            return output, {
                "requested_frames": frames_for_fps(fps),
                "used_frames": 4,
                "alignment_confidence": confidence,
                "reliable_fraction": reliable_fraction,
                "fallback_used": frames_for_fps(fps) != 4,
                "route": "4f+residual" if models.residual_4f is not None else "4f",
            }

    output = denoise_frame(
        models.model_1f,
        frames,
        target_index,
        exposure_ms,
        device,
        input_frames=1,
        tile=tile_size,
    )
    return output, {
        "requested_frames": frames_for_fps(fps),
        "used_frames": 1,
        "alignment_confidence": confidence,
        "reliable_fraction": reliable_fraction,
        "fallback_used": frames_for_fps(fps) != 1,
        "route": "1f",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--checkpoint-1f", type=Path, required=True)
    parser.add_argument("--checkpoint-4f", type=Path, required=True)
    parser.add_argument("--checkpoint-16f", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-residual-4f",
        type=Path,
        default=None,
        help="Optional detail residual head applied on top of 4f.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/inference_adaptive"))
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--min-reliable-fraction", type=float, default=0.60)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_adaptive_models(
        args.checkpoint_1f,
        args.checkpoint_4f,
        args.checkpoint_16f,
        device,
        checkpoint_residual_4f=args.checkpoint_residual_4f,
    )
    width, height, fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, width, height)
    target_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
    output, metadata = denoise_adaptive(
        models,
        frames,
        target_index,
        fps,
        exposure_ms,
        device,
        align_threshold=args.align_threshold,
        min_reliable_fraction=args.min_reliable_fraction,
        tile_size=args.tile_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_input.png",
        decode_mono10(frames[target_index]),
    )
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_adaptive.png",
        output,
    )
    print(f"Saved adaptive result to {args.output_dir}: {metadata}")


if __name__ == "__main__":
    main()
