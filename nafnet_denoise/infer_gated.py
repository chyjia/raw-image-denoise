"""Alignment-confidence soft gating for 4f NAFNet inference.

Bad ECC matches pull neighbor frames toward the center frame (less fusion);
very low global confidence also blends toward the 1f model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import center_frame_index, decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import denoise_frame, denoise_from_dn_frames, load_model, save_mono10_png
from .infer_burst import alignment_confidence, prepare_aligned_burst


def soft_frame_weights(
    responses: np.ndarray,
    reference_index: int,
    threshold: float = 0.03,
    temperature: float = 0.015,
) -> np.ndarray:
    """Per-frame weight in [0, 1]: 1 keeps aligned neighbor, 0 falls back to center."""
    weights = np.ones(responses.shape[0], dtype=np.float32)
    for index, response in enumerate(responses):
        if index == reference_index:
            weights[index] = 1.0
            continue
        weights[index] = float(
            1.0 / (1.0 + np.exp(-(float(response) - threshold) / max(temperature, 1e-6)))
        )
    return weights


def soft_gate_frames(
    aligned: list[np.ndarray],
    responses: np.ndarray,
    reference_index: int,
    threshold: float = 0.03,
    temperature: float = 0.015,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Blend each neighbor toward the center when ECC response is weak."""
    reference = aligned[reference_index]
    weights = soft_frame_weights(responses, reference_index, threshold, temperature)
    gated: list[np.ndarray] = []
    for index, frame in enumerate(aligned):
        if index == reference_index:
            gated.append(frame.astype(np.float32, copy=False))
            continue
        w = float(weights[index])
        gated.append((w * frame + (1.0 - w) * reference).astype(np.float32))
    return gated, weights


def global_fusion_weight(
    confidence: float,
    reliable_fraction: float,
    soft_lo: float = 0.35,
    soft_hi: float = 0.85,
) -> float:
    """Map alignment quality to [0, 1] weight on the gated 4f output (vs 1f)."""
    score = 0.5 * float(reliable_fraction) + 0.5 * float(np.clip(confidence / 0.15, 0.0, 1.0))
    if soft_hi <= soft_lo:
        return 1.0 if score >= soft_hi else 0.0
    return float(np.clip((score - soft_lo) / (soft_hi - soft_lo), 0.0, 1.0))


@torch.no_grad()
def denoise_gated(
    model_4f: torch.nn.Module,
    model_1f: torch.nn.Module,
    frames: np.memmap,
    target_index: int,
    exposure_ms: float,
    device: torch.device,
    input_frames: int = 4,
    align_threshold: float = 0.03,
    temperature: float = 0.015,
    soft_lo: float = 0.35,
    soft_hi: float = 0.85,
    tile_size: int = 256,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    aligned, responses = prepare_aligned_burst(
        frames,
        target_index,
        input_frames,
        min_response=align_threshold,
        reject_unreliable=False,
        return_responses=True,
    )
    reference_index = center_frame_index(input_frames)
    confidence = alignment_confidence(responses, reference_index)
    companion = np.ones(input_frames, dtype=bool)
    companion[reference_index] = False
    reliable_fraction = float(np.mean(responses[companion] >= align_threshold))
    gated, frame_weights = soft_gate_frames(
        aligned,
        responses,
        reference_index,
        threshold=align_threshold,
        temperature=temperature,
    )
    fusion_w = global_fusion_weight(
        confidence,
        reliable_fraction,
        soft_lo=soft_lo,
        soft_hi=soft_hi,
    )

    out_4f = denoise_from_dn_frames(
        model_4f,
        gated,
        exposure_ms,
        device,
        tile=tile_size,
    )
    if fusion_w >= 0.999:
        output = out_4f
        route = "gated_4f"
    elif fusion_w <= 0.001:
        output = denoise_frame(
            model_1f,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=1,
            tile=tile_size,
        )
        route = "gated_1f"
    else:
        out_1f = denoise_frame(
            model_1f,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=1,
            tile=tile_size,
        )
        output = (fusion_w * out_4f + (1.0 - fusion_w) * out_1f).astype(np.float32)
        route = "gated_blend"

    meta: dict[str, float | int | str] = {
        "route": route,
        "used_frames": input_frames if fusion_w > 0.001 else 1,
        "alignment_confidence": confidence,
        "reliable_fraction": reliable_fraction,
        "fusion_weight": fusion_w,
        "frame_weight_mean": float(np.mean(frame_weights[companion])),
    }
    return output, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--checkpoint-4f", type=Path, required=True)
    parser.add_argument("--checkpoint-1f", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/inference_gated"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.015)
    parser.add_argument("--soft-lo", type=float, default=0.35)
    parser.add_argument("--soft-hi", type=float, default=0.85)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_4f, frames_4f, _ = load_model(args.checkpoint_4f, None, device)
    model_1f, frames_1f, _ = load_model(args.checkpoint_1f, None, device)
    if frames_4f != 4:
        raise SystemExit(f"Expected 4f checkpoint, got {frames_4f}")
    if frames_1f != 1:
        raise SystemExit(f"Expected 1f checkpoint, got {frames_1f}")

    width, height, _fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, width, height)
    target_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
    output, meta = denoise_gated(
        model_4f,
        model_1f,
        frames,
        target_index,
        exposure_ms,
        device,
        tile_size=args.tile_size,
        align_threshold=args.align_threshold,
        temperature=args.temperature,
        soft_lo=args.soft_lo,
        soft_hi=args.soft_hi,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_input.png",
        decode_mono10(frames[target_index]),
    )
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_gated.png",
        output,
    )
    print(f"Saved gated result to {args.output_dir}: {meta}")


if __name__ == "__main__":
    main()
