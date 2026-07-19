"""Export two frames per Mono10 flat-field video using three CV denoisers.

Methods:
1. Brightness-aligned temporal trimmed mean (16 nearby frames).
2. Generalized-Anscombe VST followed by tiled BM3D.
3. Generalized-Anscombe VST followed by OpenCV fast non-local means (NLM).

The PTC constants below are the mean of the two current independent Mono10
grayscale-chart calibrations.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import bm3d
import cv2
import numpy as np


PTC_SLOPE = 0.10618515
PTC_INTERCEPT = 2.00145577
RAW_MAX = 1023.0
FILENAME_RE = re.compile(
    r"_w(?P<width>\d+)_h(?P<height>\d+)_pMono10_f(?P<fps>\d+(?:\.\d+)?)\.raw$",
    re.IGNORECASE,
)


def parse_metadata(path: Path) -> tuple[int, int, float]:
    match = FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse Mono10 metadata from {path.name}")
    return int(match["width"]), int(match["height"]), float(match["fps"])


def central_roi(height: int, width: int, fraction: float = 0.40) -> tuple[slice, slice]:
    roi_height = int(height * fraction) // 2 * 2
    roi_width = int(width * fraction) // 2 * 2
    y0 = (height - roi_height) // 2
    x0 = (width - roi_width) // 2
    return slice(y0, y0 + roi_height), slice(x0, x0 + roi_width)


def decode_frame(raw: np.ndarray) -> np.ndarray:
    """Decode unpacked low-aligned Mono10 stored as uint16 little endian."""
    return (raw & 0x03FF).astype(np.float32)


def vst_forward(image: np.ndarray) -> np.ndarray:
    radicand = PTC_SLOPE * image + 0.375 * PTC_SLOPE**2 + PTC_INTERCEPT
    return (2.0 / PTC_SLOPE) * np.sqrt(np.maximum(radicand, 1e-8))


def vst_inverse(transformed: np.ndarray) -> np.ndarray:
    image = (
        (PTC_SLOPE / 4.0) * transformed**2
        - 0.375 * PTC_SLOPE
        - PTC_INTERCEPT / PTC_SLOPE
    )
    return np.clip(image, 0.0, RAW_MAX).astype(np.float32)


def vst_to_u8(transformed: np.ndarray) -> np.ndarray:
    minimum = float(vst_forward(np.array(0.0, dtype=np.float32)))
    maximum = float(vst_forward(np.array(RAW_MAX, dtype=np.float32)))
    scaled = (transformed - minimum) * (255.0 / (maximum - minimum))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def u8_to_vst(image: np.ndarray) -> np.ndarray:
    minimum = float(vst_forward(np.array(0.0, dtype=np.float32)))
    maximum = float(vst_forward(np.array(RAW_MAX, dtype=np.float32)))
    return image.astype(np.float32) * ((maximum - minimum) / 255.0) + minimum


def aligned_temporal_trimmed_mean(
    data: np.memmap,
    target_index: int,
    ys: slice,
    xs: slice,
    frame_count: int,
    window: int,
    trim_fraction: float,
    exclude_frame_index: int | None = None,
) -> np.ndarray:
    """Align nearby frames to the target ROI brightness and use a trimmed mean."""
    half = window // 2
    start = max(0, min(target_index - half, frame_count - window))
    indices = list(range(start, min(frame_count, start + window)))
    if exclude_frame_index is not None:
        indices = [index for index in indices if index != exclude_frame_index]
    if len(indices) < 3:
        raise ValueError("Temporal trimmed mean needs at least three source frames.")
    target_mean = float(decode_frame(data[target_index, ys, xs]).mean())
    frames = []
    for index in indices:
        image = decode_frame(data[index])
        offset = float(image[ys, xs].mean()) - target_mean
        frames.append(image - offset)
    stack = np.stack(frames, axis=0)
    trim = max(0, int(len(indices) * trim_fraction))
    if trim:
        stack.sort(axis=0)
        stack = stack[trim:-trim]
    return np.clip(stack.mean(axis=0), 0.0, RAW_MAX).astype(np.float32)


def tiled_bm3d(
    transformed: np.ndarray,
    sigma_psd: float = 1.0,
    tile_size: int = 512,
    overlap: int = 48,
) -> np.ndarray:
    """Run BM3D in overlapped tiles, blending tiles to avoid seams."""
    height, width = transformed.shape
    step = tile_size - overlap
    accumulation = np.zeros_like(transformed, dtype=np.float64)
    weights = np.zeros_like(transformed, dtype=np.float64)
    window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float64)
    window = np.maximum(window, 1e-3)

    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile_size)
        y0 = max(0, y1 - tile_size)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile_size)
            x0 = max(0, x1 - tile_size)
            tile = transformed[y0:y1, x0:x1]
            denoised = bm3d.bm3d(tile.astype(np.float64), sigma_psd=sigma_psd)
            tile_window = window[: y1 - y0, : x1 - x0]
            accumulation[y0:y1, x0:x1] += denoised * tile_window
            weights[y0:y1, x0:x1] += tile_window
    return (accumulation / np.maximum(weights, 1e-8)).astype(np.float32)


def vst_nlm(image: np.ndarray) -> np.ndarray:
    transformed = vst_forward(image)
    transformed_u8 = vst_to_u8(transformed)
    vst_min = float(vst_forward(np.array(0.0, dtype=np.float32)))
    vst_max = float(vst_forward(np.array(RAW_MAX, dtype=np.float32)))
    sigma_u8 = 255.0 / (vst_max - vst_min)
    denoised_u8 = cv2.fastNlMeansDenoising(
        transformed_u8,
        None,
        h=1.15 * sigma_u8,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return vst_inverse(u8_to_vst(denoised_u8))


def save_mono10_png(path: Path, image: np.ndarray) -> None:
    """Write Mono10 values expanded into the full 16-bit PNG range."""
    output = np.clip(np.rint(image), 0, RAW_MAX).astype(np.uint16) << 6
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Could not write {path}")


def save_preview(path: Path, images: list[np.ndarray], labels: list[str]) -> None:
    preview = [
        cv2.cvtColor(np.clip(image / RAW_MAX * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for image in images
    ]
    for image, label in zip(preview, labels):
        cv2.putText(
            image,
            label,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    canvas = np.hstack(preview)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write {path}")


def highpass_noise_sigma(image: np.ndarray, ys: slice, xs: slice) -> float:
    roi = image[ys, xs]
    lowpass = cv2.GaussianBlur(roi, (0, 0), 1.2)
    residual = roi - lowpass
    return float(np.median(np.abs(residual - np.median(residual))) / 0.6745)


def process_sequence(
    path: Path,
    input_root: Path,
    output_dir: Path,
    temporal_window: int,
    trim_fraction: float,
    target_indices: list[int] | None,
    metrics: list[dict[str, str]],
) -> None:
    width, height, fps = parse_metadata(path)
    bytes_per_frame = width * height * 2
    frame_count = path.stat().st_size // bytes_per_frame
    if path.stat().st_size % bytes_per_frame:
        raise ValueError(f"{path.name} is not a whole number of unpacked Mono10 frames.")
    data = np.memmap(path, mode="r", dtype="<u2", shape=(frame_count, height, width))
    ys, xs = central_roi(height, width)
    if target_indices is None:
        selected_indices = sorted({0, frame_count // 2})
    else:
        selected_indices = sorted(set(target_indices))
        invalid = [index for index in selected_indices if index < 0 or index >= frame_count]
        if invalid:
            raise ValueError(
                f"{path.name}: requested frame indices {invalid} are outside "
                f"the available range 0-{frame_count - 1}."
            )
    relative_stem = path.relative_to(input_root).with_suffix("")
    sequence_name = "__".join(relative_stem.parts)
    sequence_dir = output_dir / f"f{fps:g}_{sequence_name}"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{path.name}: {frame_count} frames; exporting target frames {selected_indices}")
    for target_index in selected_indices:
        input_image = decode_frame(data[target_index])
        temporal = aligned_temporal_trimmed_mean(
            data, target_index, ys, xs, frame_count, temporal_window, trim_fraction
        )
        print(f"  frame {target_index}: BM3D...")
        bm3d_image = vst_inverse(tiled_bm3d(vst_forward(input_image)))
        print(f"  frame {target_index}: NLM...")
        nlm_image = vst_nlm(input_image)

        prefix = f"frame_{target_index:03d}"
        outputs = {
            "input": input_image,
            "temporal_trimmed_mean": temporal,
            "vst_bm3d": bm3d_image,
            "vst_nlm": nlm_image,
        }
        for method, image in outputs.items():
            save_mono10_png(sequence_dir / f"{prefix}_{method}_mono10.png", image)
        save_preview(
            sequence_dir / f"{prefix}_comparison_preview.png",
            list(outputs.values()),
            ["Input", "Temporal mean", "VST + BM3D", "VST + NLM"],
        )

        for method, image in outputs.items():
            metrics.append(
                {
                    "file": path.name,
                    "fps": f"{fps:g}",
                    "frame_index": str(target_index),
                    "method": method,
                    "roi_mean_dn": f"{float(image[ys, xs].mean()):.5f}",
                    "roi_highpass_noise_sigma_dn": f"{highpass_noise_sigma(image, ys, xs):.5f}",
                    "mae_to_temporal_mean_dn": f"{float(np.mean(np.abs(image - temporal))):.5f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("泊松分布/cv_denoise_results"))
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    parser.add_argument(
        "--frame-index",
        type=int,
        action="append",
        help="Frame index to export. Repeat for multiple frames; default exports frame 0 and midpoint.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search all subdirectories for Mono10 RAW files.",
    )
    args = parser.parse_args()

    pattern = "*_pMono10_f*.raw"
    files = sorted(args.input_dir.rglob(pattern) if args.recursive else args.input_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No Mono10 RAW files found in {args.input_dir}")
    if args.temporal_window < 3:
        raise SystemExit("--temporal-window must be at least 3.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, str]] = []
    for path in files:
        process_sequence(
            path,
            args.input_dir,
            args.output_dir,
            args.temporal_window,
            args.trim_fraction,
            args.frame_index,
            metrics,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\nSaved {len(metrics)} metric rows to {metrics_path}")


if __name__ == "__main__":
    main()
