"""Validate the photon-transfer curve (PTC) from Mono10 flat-field videos.

Each RAW file is assumed to contain an unpacked 16-bit little-endian Mono10
frame sequence. The filename must include ``_w<width>_h<height>_pMono10_f<fps>``.
The validation uses temporal variance at every pixel, rather than spatial
variance, so fixed-pattern brightness differences across the wall are excluded.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILENAME_RE = re.compile(
    r"_w(?P<width>\d+)_h(?P<height>\d+)_pMono10_f(?P<fps>\d+(?:\.\d+)?)\.raw$",
    re.IGNORECASE,
)


@dataclass
class SequenceStats:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    sampled_frames: int
    layout: str
    mean_dn: float
    variance_dn2: float
    corrected_variance_dn2: float
    variance_sem_dn2: float
    illumination_cv_percent: float
    saturation_percent: float


def parse_metadata(path: Path) -> tuple[int, int, float]:
    match = FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(
            f"Cannot parse Mono10 width, height, and fps from filename: {path.name}"
        )
    return (
        int(match.group("width")),
        int(match.group("height")),
        float(match.group("fps")),
    )


def choose_roi(width: int, height: int, fraction: float) -> tuple[slice, slice]:
    if not 0 < fraction <= 1:
        raise ValueError("--roi-fraction must be in (0, 1].")
    roi_width = max(16, int(width * fraction))
    roi_height = max(16, int(height * fraction))
    roi_width -= roi_width % 2
    roi_height -= roi_height % 2
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    return slice(y0, y0 + roi_height), slice(x0, x0 + roi_width)


def infer_layout(probe: np.ndarray) -> str:
    """Return how 10-bit samples are stored in their uint16 container."""
    max_value = int(probe.max())
    if max_value <= 1023:
        return "lsb"
    if np.count_nonzero(probe & 0x003F) == 0:
        return "msb"
    raise ValueError(
        "The samples are neither low-aligned nor high-aligned unpacked Mono10. "
        "This file may use a packed pixel format."
    )


def decode_mono10(values: np.ndarray, layout: str) -> np.ndarray:
    if layout == "lsb":
        return (values & 0x03FF).astype(np.float64)
    if layout == "msb":
        return (values >> 6).astype(np.float64)
    raise ValueError(f"Unsupported Mono10 layout: {layout}")


def sequence_stats(path: Path, max_frames: int, roi_fraction: float) -> SequenceStats:
    width, height, fps = parse_metadata(path)
    frame_pixels = width * height
    bytes_per_frame = frame_pixels * np.dtype("<u2").itemsize
    file_size = path.stat().st_size
    if file_size % bytes_per_frame:
        raise ValueError(
            f"{path.name}: {file_size} bytes is not an integer number of "
            f"{width}x{height} unpacked Mono10 frames."
        )

    frame_count = file_size // bytes_per_frame
    if frame_count < 3:
        raise ValueError(f"{path.name}: at least three frames are required.")

    data = np.memmap(path, mode="r", dtype="<u2", shape=(frame_count, height, width))
    ys, xs = choose_roi(width, height, roi_fraction)
    frame_indices = np.linspace(
        0, frame_count - 1, min(frame_count, max_frames), dtype=np.int64
    )

    probe = data[int(frame_indices[0]), ys, xs]
    layout = infer_layout(probe)
    roi_shape = probe.shape
    pixel_sum = np.zeros(roi_shape, dtype=np.float64)
    pixel_sum_sq = np.zeros(roi_shape, dtype=np.float64)
    corrected_sum = np.zeros(roi_shape, dtype=np.float64)
    corrected_sum_sq = np.zeros(roi_shape, dtype=np.float64)
    frame_means: list[float] = []
    saturated_samples = 0

    for frame_index in frame_indices:
        frame = decode_mono10(data[int(frame_index), ys, xs], layout)
        frame_mean = float(frame.mean())
        frame_means.append(frame_mean)
        pixel_sum += frame
        pixel_sum_sq += frame * frame
        saturated_samples += int(np.count_nonzero(frame >= 1023))

    sample_count = len(frame_indices)
    pixel_mean = pixel_sum / sample_count
    pixel_variance = (
        pixel_sum_sq - pixel_sum * pixel_sum / sample_count
    ) / (sample_count - 1)

    # Remove frame-wide brightness drift (e.g. lamp flicker) before the PTC
    # fit. We keep the sequence mean while subtracting each frame's ROI offset.
    series_mean = float(np.mean(frame_means))
    for frame_index, frame_mean in zip(frame_indices, frame_means):
        frame = decode_mono10(data[int(frame_index), ys, xs], layout)
        corrected = frame - frame_mean + series_mean
        corrected_sum += corrected
        corrected_sum_sq += corrected * corrected

    corrected_variance = (
        corrected_sum_sq - corrected_sum * corrected_sum / sample_count
    ) / (sample_count - 1)
    corrected_variance = np.maximum(corrected_variance, 0.0)

    return SequenceStats(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        sampled_frames=sample_count,
        layout=layout,
        mean_dn=float(pixel_mean.mean()),
        variance_dn2=float(np.maximum(pixel_variance, 0.0).mean()),
        corrected_variance_dn2=float(corrected_variance.mean()),
        variance_sem_dn2=float(corrected_variance.std(ddof=1) / np.sqrt(corrected_variance.size)),
        illumination_cv_percent=float(np.std(frame_means, ddof=1) / series_mean * 100),
        saturation_percent=float(saturated_samples / (sample_count * probe.size) * 100),
    )


def write_csv(stats: list[SequenceStats], output_path: Path) -> None:
    fields = [
        "file",
        "fps",
        "frame_interval_ms",
        "frames_in_file",
        "frames_sampled",
        "mono10_layout",
        "mean_dn",
        "raw_temporal_variance_dn2",
        "flicker_corrected_variance_dn2",
        "variance_sem_dn2",
        "illumination_cv_percent",
        "saturation_percent",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in stats:
            writer.writerow(
                {
                    "file": item.path.name,
                    "fps": f"{item.fps:g}",
                    "frame_interval_ms": f"{1000.0 / item.fps:.6f}",
                    "frames_in_file": item.frame_count,
                    "frames_sampled": item.sampled_frames,
                    "mono10_layout": item.layout,
                    "mean_dn": f"{item.mean_dn:.6f}",
                    "raw_temporal_variance_dn2": f"{item.variance_dn2:.6f}",
                    "flicker_corrected_variance_dn2": f"{item.corrected_variance_dn2:.6f}",
                    "variance_sem_dn2": f"{item.variance_sem_dn2:.6f}",
                    "illumination_cv_percent": f"{item.illumination_cv_percent:.6f}",
                    "saturation_percent": f"{item.saturation_percent:.6f}",
                }
            )


def fit_ptc(stats: list[SequenceStats]) -> tuple[float, float, float, np.ndarray]:
    usable = np.array([item.saturation_percent < 0.1 for item in stats], dtype=bool)
    if usable.sum() < 3:
        raise ValueError("Need at least three unsaturated sequences for a PTC fit.")
    means = np.array([item.mean_dn for item in stats], dtype=np.float64)
    variances = np.array(
        [item.corrected_variance_dn2 for item in stats], dtype=np.float64
    )
    slope, intercept = np.polyfit(means[usable], variances[usable], deg=1)
    prediction = slope * means + intercept
    residual = variances[usable] - prediction[usable]
    total = variances[usable] - variances[usable].mean()
    r_squared = 1.0 - float(np.sum(residual * residual) / np.sum(total * total))
    return float(slope), float(intercept), r_squared, prediction


def write_plot(
    stats: list[SequenceStats],
    slope: float,
    intercept: float,
    r_squared: float,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    means = np.array([item.mean_dn for item in stats])
    raw_var = np.array([item.variance_dn2 for item in stats])
    corrected_var = np.array([item.corrected_variance_dn2 for item in stats])
    sem = np.array([item.variance_sem_dn2 for item in stats])

    figure, (axis, residual_axis) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    axis.scatter(means, raw_var, label="Raw temporal variance", color="#a0a9b8", s=55)
    axis.errorbar(
        means,
        corrected_var,
        yerr=sem,
        fmt="o",
        capsize=3,
        color="#0077b6",
        label="Flicker-corrected temporal variance",
    )
    x_line = np.linspace(means.min() * 0.97, means.max() * 1.03, 200)
    axis.plot(
        x_line,
        slope * x_line + intercept,
        color="#d1495b",
        linewidth=2,
        label=f"Fit: variance = {slope:.5f} × mean + {intercept:.3f}",
    )
    for item in stats:
        axis.annotate(
            f"f{item.fps:g}",
            (item.mean_dn, item.corrected_variance_dn2),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title(f"Mono10 PTC validation (R² = {r_squared:.5f})")
    axis.set_xlabel("Mean signal (DN)")
    axis.set_ylabel("Temporal variance (DN²)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    residual_axis.axhline(0, color="#5b6573", linewidth=1)
    residual_axis.scatter(means, corrected_var - prediction, color="#0077b6", s=55)
    residual_axis.set_title("Fit residuals")
    residual_axis.set_xlabel("Mean signal (DN)")
    residual_axis.set_ylabel("Variance residual (DN²)")
    residual_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_report(
    stats: list[SequenceStats],
    slope: float,
    intercept: float,
    r_squared: float,
    roi_fraction: float,
    output_path: Path,
) -> None:
    read_noise_dn = float(np.sqrt(max(intercept, 0.0)))
    lines = [
        "Mono10 flat-field photon-transfer-curve validation",
        "",
        f"Sequences analysed: {len(stats)}",
        f"Central ROI: {roi_fraction * 100:.1f}% of width and height",
        "Variance source: per-pixel temporal sample variance",
        "Flicker correction: each frame is shifted by its ROI mean before variance calculation",
        "",
        f"PTC fit: variance_dn2 = {slope:.8f} * mean_dn + {intercept:.8f}",
        f"R squared: {r_squared:.8f}",
        f"Read-noise intercept estimate: sqrt(max(intercept, 0)) = {read_noise_dn:.6f} DN",
        "",
        "Interpretation:",
        "- A positive, approximately linear slope supports a Poisson-Gaussian noise model.",
        "- Large residuals or high illumination CV can indicate lamp flicker, saturation, or changing camera settings.",
        "- The slope is in DN variance per DN mean; convert it to electrons only after camera conversion gain is calibrated.",
        "",
        "Per-sequence checks:",
    ]
    for item in stats:
        lines.append(
            f"- {item.path.name}: f={item.fps:g}, interval={1000 / item.fps:.3f} ms, "
            f"mean={item.mean_dn:.3f} DN, corrected_var={item.corrected_variance_dn2:.3f} DN², "
            f"illumination_CV={item.illumination_cv_percent:.4f}%, saturation={item.saturation_percent:.4f}%"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing Mono10 flat-field RAW videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("ptc_validation"),
        help="Directory for CSV, PNG, and report outputs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=96,
        help="Evenly sampled frames per video (default: 96).",
    )
    parser.add_argument(
        "--roi-fraction",
        type=float,
        default=0.40,
        help="Central ROI width/height as an image fraction (default: 0.40).",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[SequenceStats] = []
    for path in files:
        print(f"Analysing {path.name}...")
        result = sequence_stats(path, args.max_frames, args.roi_fraction)
        stats.append(result)
        print(
            f"  {result.frame_count} frames, f{result.fps:g}, mean={result.mean_dn:.3f} DN, "
            f"corrected variance={result.corrected_variance_dn2:.3f} DN²"
        )

    slope, intercept, r_squared, prediction = fit_ptc(stats)
    csv_path = args.output_dir / "ptc_summary.csv"
    plot_path = args.output_dir / "ptc_validation.png"
    report_path = args.output_dir / "ptc_report.txt"
    write_csv(stats, csv_path)
    write_plot(stats, slope, intercept, r_squared, prediction, plot_path)
    write_report(stats, slope, intercept, r_squared, args.roi_fraction, report_path)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
