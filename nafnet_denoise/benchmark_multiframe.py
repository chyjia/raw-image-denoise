"""Benchmark 1f, 4f, 16f and adaptive denoising on identical RAW frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import (
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    center_frame_index,
    decode_mono10,
    memmap_frames,
    parse_exposure_ms,
    parse_geometry,
)
from .infer import denoise_frame, save_mono10_png
from .infer_adaptive import denoise_adaptive, load_adaptive_models
from .infer_burst import (
    alignment_confidence,
    denoise_burst,
    prepare_aligned_burst,
)
from .validate import (
    aligned_temporal_trimmed_mean,
    central_roi,
    gradient_error,
    highpass_noise_sigma,
    save_preview,
)


def reliable_reference_mask(input_dn: np.ndarray, reference: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.maximum(PTC_SLOPE * reference + PTC_INTERCEPT, 1e-6))
    threshold = np.maximum(3.0 * sigma, 8.0)
    return (
        (np.abs(input_dn - reference) <= threshold)
        & (input_dn < 1000.0)
        & (reference < 1000.0)
    )


def masked_ssim(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    image = image.astype(np.float32) / RAW_MAX
    reference = reference.astype(np.float32) / RAW_MAX
    mu_x = cv2.GaussianBlur(image, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(reference, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(image * image, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(reference * reference, (11, 11), 1.5) - mu_y * mu_y
    covariance = cv2.GaussianBlur(image * reference, (11, 11), 1.5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / np.maximum(
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2),
        1e-8,
    )
    return float(score[mask].mean()) if np.any(mask) else float("nan")


def metric_row(
    path: Path,
    fps: float,
    frame_index: int,
    method: str,
    image: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    ys: slice,
    xs: slice,
    checkpoint: str = "",
    input_frames: int = 0,
    alignment_response: float = 1.0,
    fallback_used: bool = False,
) -> dict[str, str]:
    difference = np.abs(image - reference)
    return {
        "file": path.name,
        "fps": f"{fps:g}",
        "frame_index": str(frame_index),
        "method": method,
        "checkpoint": checkpoint,
        "input_frames": str(input_frames),
        "temporal_span_s": f"{max(input_frames - 1, 0) / fps:.6f}",
        "alignment_response_median": f"{alignment_response:.6f}",
        "fallback_used": str(bool(fallback_used)).lower(),
        "reliable_reference_fraction": f"{float(mask.mean()):.6f}",
        "roi_mean_dn": f"{float(image[ys, xs].mean()):.6f}",
        "mean_bias_dn": f"{float((image - reference)[mask].mean()):.6f}",
        "roi_highpass_noise_sigma_dn": f"{highpass_noise_sigma(image, ys, xs):.6f}",
        "masked_mae_to_temporal_reference_dn": f"{float(difference[mask].mean()):.6f}",
        "masked_ssim_to_temporal_reference": f"{masked_ssim(image, reference, mask):.8f}",
        "gradient_mae_to_temporal_reference": f"{gradient_error(image, reference):.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-1f", type=Path, required=True)
    parser.add_argument("--checkpoint-4f", type=Path, required=True)
    parser.add_argument("--checkpoint-16f", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/benchmark_multiframe"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--min-reliable-fraction", type=float, default=0.60)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files under {args.input_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_adaptive_models(
        args.checkpoint_1f,
        args.checkpoint_4f,
        args.checkpoint_16f,
        device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for path in files:
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        target_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
        ys, xs = central_roi(height, width)
        input_dn = decode_mono10(frames[target_index])
        temporal = aligned_temporal_trimmed_mean(
            frames,
            target_index,
            ys,
            xs,
            frames.shape[0],
            min(args.temporal_window, frames.shape[0]),
            trim_fraction=0.10,
            exclude_frame_index=target_index,
        )
        mask = reliable_reference_mask(input_dn, temporal)

        output_1f = denoise_frame(
            models.model_1f,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=1,
            tile=args.tile_size,
        )
        output_4f = denoise_frame(
            models.model_4f,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=4,
            tile=args.tile_size,
        )
        aligned_16f, responses_16f = prepare_aligned_burst(
            frames,
            target_index,
            16,
            min_response=args.align_threshold,
            reject_unreliable=True,
            return_responses=True,
        )
        confidence_16f = alignment_confidence(
            responses_16f,
            center_frame_index(16),
        )
        output_16f = denoise_burst(
            models.model_16f,
            aligned_16f,
            exposure_ms,
            device,
            tile_size=args.tile_size,
        )
        output_adaptive, adaptive = denoise_adaptive(
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
        outputs = {
            "input": input_dn,
            "temporal_reference": temporal,
            "nafnet_1f": output_1f,
            "nafnet_4f": output_4f,
            "burst_16f": output_16f,
            "adaptive": output_adaptive,
        }
        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{target_index:03d}"
        for method, image in outputs.items():
            save_mono10_png(sequence_dir / f"{prefix}_{method}.png", image)
        save_preview(
            sequence_dir / f"{prefix}_comparison.png",
            list(outputs.values()),
            ["Input", "Temporal ref", "1f", "4f", "16f", f"Adaptive {adaptive['route']}"],
        )

        rows.append(
            metric_row(path, fps, target_index, "input", input_dn, temporal, mask, ys, xs)
        )
        rows.append(
            metric_row(
                path,
                fps,
                target_index,
                "temporal_reference",
                temporal,
                temporal,
                mask,
                ys,
                xs,
            )
        )
        for method, image, checkpoint, count, confidence in (
            ("nafnet_1f", output_1f, args.checkpoint_1f.name, 1, 1.0),
            ("nafnet_4f", output_4f, args.checkpoint_4f.name, 4, 1.0),
            ("burst_16f", output_16f, args.checkpoint_16f.name, 16, confidence_16f),
        ):
            rows.append(
                metric_row(
                    path,
                    fps,
                    target_index,
                    method,
                    image,
                    temporal,
                    mask,
                    ys,
                    xs,
                    checkpoint=checkpoint,
                    input_frames=count,
                    alignment_response=confidence,
                )
            )
        rows.append(
            metric_row(
                path,
                fps,
                target_index,
                "adaptive",
                output_adaptive,
                temporal,
                mask,
                ys,
                xs,
                checkpoint=str(adaptive["route"]),
                input_frames=int(adaptive["used_frames"]),
                alignment_response=float(adaptive["alignment_confidence"]),
                fallback_used=bool(adaptive["fallback_used"]),
            )
        )
        print(
            f"{path.name}: benchmarked frame {target_index}, "
            f"adaptive={adaptive['route']} confidence={adaptive['alignment_confidence']:.4f}"
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved unified benchmark to {metrics_path}")


if __name__ == "__main__":
    main()
