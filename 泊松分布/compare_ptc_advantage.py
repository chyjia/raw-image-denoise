"""Compare blind BM3D against PTC-guided VST + BM3D on validation videos.

The comparison exports frame 10 from every Mono10 video:
* temporal_trimmed_mean: 16-frame brightness-aligned reference;
* blind_bm3d: raw-domain BM3D with a noise level estimated from frame differences;
* ptc_vst_bm3d: generalized-Anscombe VST + BM3D using validated PTC parameters.

The temporal result is a proxy reference for this static-video experiment, not
an independently captured ground truth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import compare_cv_denoising as cvd


def estimate_blind_sigma(
    data: np.memmap,
    frame_count: int,
    ys: slice,
    xs: slice,
    target_index: int,
    pair_count: int = 12,
) -> float:
    """Robustly estimate raw-domain sigma from brightness-aligned frame pairs."""
    start = max(0, min(target_index - pair_count // 2, frame_count - pair_count - 1))
    samples = []
    for index in range(start, start + pair_count):
        first = cvd.decode_frame(data[index])
        second = cvd.decode_frame(data[index + 1])
        first_mean = float(first[ys, xs].mean())
        second_mean = float(second[ys, xs].mean())
        difference = (first - first_mean) - (second - second_mean)
        samples.append(difference[::8, ::8].ravel())
    values = np.concatenate(samples)
    median = float(np.median(values))
    mad_sigma = float(np.median(np.abs(values - median)) / 0.6745)
    return max(mad_sigma / np.sqrt(2.0), 0.5)


def gradient_error(image: np.ndarray, reference: np.ndarray) -> float:
    image_gradient = cv2.magnitude(
        cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3),
    )
    reference_gradient = cv2.magnitude(
        cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3),
    )
    return float(np.mean(np.abs(image_gradient - reference_gradient)))


def process_video(
    path: Path,
    input_root: Path,
    output_root: Path,
    target_index: int,
    temporal_window: int,
    metrics: list[dict[str, str]],
) -> None:
    width, height, fps = cvd.parse_metadata(path)
    bytes_per_frame = width * height * 2
    frame_count = path.stat().st_size // bytes_per_frame
    if frame_count <= target_index:
        raise ValueError(f"{path.name} has only {frame_count} frames; cannot export frame {target_index}.")
    data = np.memmap(path, mode="r", dtype="<u2", shape=(frame_count, height, width))
    ys, xs = cvd.central_roi(height, width)
    input_image = cvd.decode_frame(data[target_index])
    temporal = cvd.aligned_temporal_trimmed_mean(
        data,
        target_index,
        ys,
        xs,
        frame_count,
        temporal_window,
        trim_fraction=0.10,
        exclude_frame_index=target_index,
    )
    blind_sigma = estimate_blind_sigma(data, frame_count, ys, xs, target_index)
    print(f"{path.name}: frame {target_index}, blind sigma={blind_sigma:.3f} DN")
    blind_bm3d = np.clip(
        cvd.tiled_bm3d(input_image, sigma_psd=blind_sigma), 0.0, cvd.RAW_MAX
    )
    ptc_vst_bm3d = cvd.vst_inverse(cvd.tiled_bm3d(cvd.vst_forward(input_image)))

    relative_stem = path.relative_to(input_root).with_suffix("")
    output_dir = output_root / "__".join(relative_stem.parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "input": input_image,
        "temporal_trimmed_mean": temporal,
        "blind_bm3d": blind_bm3d,
        "ptc_vst_bm3d": ptc_vst_bm3d,
    }
    for name, image in outputs.items():
        cvd.save_mono10_png(output_dir / f"frame_{target_index:03d}_{name}_mono10.png", image)
    cvd.save_preview(
        output_dir / f"frame_{target_index:03d}_comparison_preview.png",
        list(outputs.values()),
        ["Input", "Temporal mean", "Blind BM3D", "PTC VST + BM3D"],
    )

    for method, image in outputs.items():
        metrics.append(
            {
                "file": path.name,
                "fps": f"{fps:g}",
                "frame_index": str(target_index),
                "method": method,
                "blind_sigma_dn": f"{blind_sigma:.6f}",
                "roi_mean_dn": f"{float(image[ys, xs].mean()):.6f}",
                "roi_highpass_noise_sigma_dn": f"{cvd.highpass_noise_sigma(image, ys, xs):.6f}",
                "mae_to_temporal_reference_dn": f"{float(np.mean(np.abs(image - temporal))):.6f}",
                "gradient_mae_to_temporal_reference": f"{gradient_error(image, temporal):.6f}",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("泊松分布/ptc_known_vs_blind_validation_frame010"),
    )
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--temporal-window", type=int, default=16)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, str]] = []
    for path in files:
        process_video(
            path,
            args.input_dir,
            args.output_dir,
            args.frame_index,
            args.temporal_window,
            metrics,
        )

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved {len(metrics)} metric rows to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
