"""Validate a 16-frame BurstNAFNet against held-out Mono10 RAW sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .common import RAW_MAX, decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import save_mono10_png
from .infer_burst import denoise_burst, load_burst_model, prepare_aligned_burst
from .validate import (
    aligned_temporal_trimmed_mean,
    central_roi,
    gradient_error,
    highpass_noise_sigma,
    save_preview,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\训练素材\不参加训练，用来验证训练模型效果"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/validation_results_16f"),
    )
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files under {args.input_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, width, input_frames = load_burst_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for path in files:
        frame_width, frame_height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, frame_width, frame_height)
        frame_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
        if frame_index >= frames.shape[0]:
            raise ValueError(f"{path.name}: frame index outside sequence")
        ys, xs = central_roi(frame_height, frame_width)
        input_dn = decode_mono10(frames[frame_index])
        temporal = aligned_temporal_trimmed_mean(
            frames,
            frame_index,
            ys,
            xs,
            frames.shape[0],
            args.temporal_window,
            trim_fraction=0.10,
            exclude_frame_index=frame_index,
        )
        print(f"{path.name}: 16-frame inference...")
        aligned = prepare_aligned_burst(frames, frame_index, input_frames)
        prediction = np.clip(
            denoise_burst(
                model,
                aligned,
                exposure_ms,
                device,
                tile_size=args.tile_size,
            ),
            0.0,
            RAW_MAX,
        ).astype(np.float32)

        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "input": input_dn,
            "temporal_trimmed_mean": temporal,
            "burst16": prediction,
        }
        prefix = f"frame_{frame_index:03d}"
        for method, image in outputs.items():
            save_mono10_png(sequence_dir / f"{prefix}_{method}_mono10.png", image)
            rows.append(
                {
                    "file": path.name,
                    "fps": f"{fps:g}",
                    "frame_index": str(frame_index),
                    "method": method,
                    "checkpoint": args.checkpoint.name if method == "burst16" else "",
                    "roi_mean_dn": f"{float(image[ys, xs].mean()):.6f}",
                    "roi_highpass_noise_sigma_dn": f"{highpass_noise_sigma(image, ys, xs):.6f}",
                    "mae_to_temporal_reference_dn": f"{float(np.mean(np.abs(image - temporal))):.6f}",
                    "gradient_mae_to_temporal_reference": f"{gradient_error(image, temporal):.6f}",
                }
            )
        save_preview(
            sequence_dir / f"{prefix}_comparison_preview.png",
            list(outputs.values()),
            ["Input", "Temporal mean", "Burst16"],
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {metrics_path}; model width={width}, frames={input_frames}")


if __name__ == "__main__":
    main()
