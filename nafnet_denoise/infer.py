"""Run NAFNet inference on Mono10 RAW frame bursts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import (
    DEFAULT_INPUT_FRAMES,
    center_frame_index,
    decode_mono10,
    frames_to_model_input,
    gather_frame_indices,
    memmap_frames,
    model_output_to_raw,
    parse_exposure_ms,
    parse_geometry,
)
from .nafnet import build_nafnet


def load_model(
    checkpoint: Path,
    width: int,
    device: torch.device,
    input_frames: int | None = None,
) -> tuple[torch.nn.Module, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        saved_args = payload.get("args", {})
        if input_frames is None:
            input_frames = int(saved_args.get("input_frames", DEFAULT_INPUT_FRAMES))
        state = payload["model"]
    else:
        if input_frames is None:
            input_frames = DEFAULT_INPUT_FRAMES
        state = payload

    model = build_nafnet(width=width, input_frames=input_frames)
    model.load_state_dict(state)
    return model.eval().to(device), input_frames


def denoise_frame(
    model: torch.nn.Module,
    frames: np.memmap,
    target_index: int,
    exposure_ms: float,
    device: torch.device,
    input_frames: int = DEFAULT_INPUT_FRAMES,
    tile: int = 512,
    overlap: int = 32,
) -> np.ndarray:
    frame_indices = gather_frame_indices(target_index, frames.shape[0], input_frames)
    center = center_frame_index(input_frames)
    decoded = [decode_mono10(frames[index]) for index in frame_indices]
    height, width = decoded[center].shape
    output = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    step = tile - overlap

    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile)
        y0 = max(0, y1 - tile)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile)
            x0 = max(0, x1 - tile)
            patches = [frame[y0:y1, x0:x1] for frame in decoded]
            model_input = frames_to_model_input(
                patches,
                exposure_ms,
                reference_index=center,
            )
            tensor = torch.from_numpy(model_input[None]).float().to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                pred_vst = model(tensor).float().cpu().numpy()[0, 0]
            pred_dn = model_output_to_raw(pred_vst)
            window = np.outer(np.hanning(y1 - y0), np.hanning(x1 - x0)).astype(np.float32)
            output[y0:y1, x0:x1] += pred_dn * window
            weights[y0:y1, x0:x1] += window

    return output / np.maximum(weights, 1e-6)


def save_mono10_png(path: Path, image: np.ndarray) -> None:
    output = np.clip(np.rint(image), 0, 1023).astype(np.uint16) << 6
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/inference"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--input-frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--tile", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_frames = load_model(
        args.checkpoint,
        args.width,
        device,
        input_frames=args.input_frames,
    )

    width, height, _fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, width, height)
    if args.frame_index < 0 or args.frame_index >= frames.shape[0]:
        raise SystemExit(f"frame-index out of range: 0-{frames.shape[0] - 1}")

    input_dn = decode_mono10(frames[args.frame_index])
    output_dn = denoise_frame(
        model,
        frames,
        args.frame_index,
        exposure_ms,
        device,
        input_frames=input_frames,
        tile=args.tile,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    save_mono10_png(args.output_dir / f"{stem}_frame{args.frame_index:03d}_input.png", input_dn)
    save_mono10_png(args.output_dir / f"{stem}_frame{args.frame_index:03d}_denoised.png", output_dn)
    print(f"Saved denoised frame to {args.output_dir} (input_frames={input_frames})")


if __name__ == "__main__":
    main()
