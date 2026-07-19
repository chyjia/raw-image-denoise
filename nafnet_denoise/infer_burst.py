"""Run alignment-aware BurstNAFNet inference on a Mono10 RAW sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .alignment import estimate_translation, warp_translation
from .burst_dataset import make_burst_tensor
from .common import (
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    center_frame_index,
    decode_mono10,
    gather_frame_indices,
    memmap_frames,
    model_output_to_raw,
    parse_exposure_ms,
    parse_geometry,
)
from .infer import save_mono10_png
from .temporal_fusion import build_burst_nafnet


def load_burst_model(
    checkpoint: Path,
    device: torch.device,
    width: int | None = None,
    input_frames: int | None = None,
) -> tuple[torch.nn.Module, int, int]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = state.get("args", {}) if isinstance(state, dict) else {}
    width = width or int(saved_args.get("width", 48))
    input_frames = input_frames or int(saved_args.get("input_frames", 16))
    model = build_burst_nafnet(
        width=width,
        input_frames=input_frames,
        use_checkpoint=False,
    )
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    return model.eval().to(device), width, input_frames


def prepare_aligned_burst(
    frames: np.memmap,
    target_index: int,
    input_frames: int,
) -> list[np.ndarray]:
    indices = gather_frame_indices(target_index, frames.shape[0], input_frames)
    reference_index = center_frame_index(input_frames)
    decoded = [decode_mono10(frames[index]) for index in indices]
    reference = decoded[reference_index]
    height, width = reference.shape
    y0, y1 = int(height * 0.2), int(height * 0.8)
    x0, x1 = int(width * 0.2), int(width * 0.8)
    reference_roi = reference[y0:y1, x0:x1]
    reference_mean = float(reference_roi.mean())
    aligned = []
    for index, image in enumerate(decoded):
        offset = float(image[y0:y1, x0:x1].mean()) - reference_mean
        corrected = np.clip(image - offset, 0.0, RAW_MAX)
        if index == reference_index:
            aligned.append(corrected.astype(np.float32))
            continue
        tx, ty, response = estimate_translation(
            reference_roi,
            corrected[y0:y1, x0:x1],
            downsample=4,
            max_shift=16.0,
        )
        if response < 0.03:
            tx, ty = 0.0, 0.0
        aligned.append(warp_translation(corrected, tx, ty))
    return aligned


@torch.no_grad()
def denoise_burst(
    model: torch.nn.Module,
    aligned_frames: list[np.ndarray],
    exposure_ms: float,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    height, width = aligned_frames[0].shape
    accumulation = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    step = tile_size - overlap
    window_base = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(
        np.float32
    )
    window_base = np.maximum(window_base, 1e-3)
    visited: set[tuple[int, int]] = set()
    for proposed_y in range(0, height, step):
        y1 = min(height, proposed_y + tile_size)
        y0 = max(0, y1 - tile_size)
        for proposed_x in range(0, width, step):
            x1 = min(width, proposed_x + tile_size)
            x0 = max(0, x1 - tile_size)
            if (y0, x0) in visited:
                continue
            visited.add((y0, x0))
            patches = [frame[y0:y1, x0:x1] for frame in aligned_frames]
            burst = make_burst_tensor(
                patches,
                exposure_ms,
                PTC_SLOPE,
                PTC_INTERCEPT,
            )
            tensor = torch.from_numpy(burst[None]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(tensor)
            prediction_dn = model_output_to_raw(
                prediction.float().cpu().numpy()[0, 0]
            )
            window = window_base[: y1 - y0, : x1 - x0]
            accumulation[y0:y1, x0:x1] += prediction_dn * window
            weights[y0:y1, x0:x1] += window
    return accumulation / np.maximum(weights, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/inference_16f"))
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--input-frames", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, width, input_frames = load_burst_model(
        args.checkpoint,
        device,
        width=args.width,
        input_frames=args.input_frames,
    )
    frame_width, frame_height, _fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, frame_width, frame_height)
    aligned = prepare_aligned_burst(frames, args.frame_index, input_frames)
    output = denoise_burst(
        model,
        aligned,
        exposure_ms,
        device,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    input_image = decode_mono10(frames[args.frame_index])
    save_mono10_png(
        args.output_dir / f"{stem}_frame{args.frame_index:03d}_input.png",
        input_image,
    )
    save_mono10_png(
        args.output_dir / f"{stem}_frame{args.frame_index:03d}_burst16.png",
        output,
    )
    print(
        f"Saved 16-frame result to {args.output_dir} "
        f"(width={width}, frames={input_frames})"
    )


if __name__ == "__main__":
    main()
