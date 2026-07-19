"""Fit one PTC independently from each static Mono10 grayscale-chart video.

For every recording, the script estimates temporal variance at every pixel,
groups flat pixels by their temporal mean DN, and robustly fits
``variance_dn2 = slope * mean_dn + intercept``. Frame-wide affine brightness
normalization reduces lamp/display flicker without mixing spatial PRNU into the
temporal variance.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


META_RE = re.compile(
    r"_w(?P<width>\d+)_h(?P<height>\d+)_pMono10_f(?P<fps>\d+(?:\.\d+)?)\.raw$",
    re.IGNORECASE,
)


def parse_metadata(path: Path) -> tuple[int, int, float]:
    match = META_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse width, height and fps from {path.name}")
    return int(match["width"]), int(match["height"]), float(match["fps"])


def decode_mono10(values: np.ndarray, layout: str) -> np.ndarray:
    if layout == "lsb":
        return (values & 0x03FF).astype(np.float64)
    return (values >> 6).astype(np.float64)


def infer_layout(values: np.ndarray) -> str:
    if int(values.max()) <= 1023:
        return "lsb"
    if np.count_nonzero(values & 0x003F) == 0:
        return "msb"
    raise ValueError("RAW data is not unpacked LSB/MSB-aligned Mono10")


def robust_affine(template: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    x = template[mask][::8]
    y = frame[mask][::8]
    design = np.column_stack((x, np.ones_like(x)))
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (scale * x + offset)
    median = np.median(residual)
    mad = 1.4826 * np.median(np.abs(residual - median))
    if mad > 0:
        keep = np.abs(residual - median) <= 4.0 * mad
        if np.count_nonzero(keep) >= 100:
            scale, offset = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return float(scale), float(offset)


def robust_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    keep = np.ones(x.size, dtype=bool)
    for _ in range(6):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        center = np.median(residual[keep])
        mad = 1.4826 * np.median(np.abs(residual[keep] - center))
        if mad <= 0:
            break
        updated = np.abs(residual - center) <= 3.5 * mad
        if updated.sum() < 5 or np.array_equal(updated, keep):
            break
        keep = updated
    prediction = slope * x + intercept
    total = y[keep] - y[keep].mean()
    r_squared = 1.0 - float(
        np.sum((y[keep] - prediction[keep]) ** 2) / np.sum(total**2)
    )
    return float(slope), float(intercept), keep, r_squared


def analyse_video(
    path: Path,
    output_root: Path,
    max_frames: int,
    bin_width: float,
    min_dn: float,
    max_dn: float,
) -> None:
    width, height, fps = parse_metadata(path)
    frame_bytes = width * height * 2
    if path.stat().st_size % frame_bytes:
        raise ValueError(f"{path.name}: file size is not an integer number of frames")
    frame_count = path.stat().st_size // frame_bytes
    data = np.memmap(path, mode="r", dtype="<u2", shape=(frame_count, height, width))
    indices = np.linspace(0, frame_count - 1, min(frame_count, max_frames), dtype=np.int64)
    layout = infer_layout(data[int(indices[0])])

    pixel_sum = np.zeros((height, width), dtype=np.float64)
    for index in indices:
        pixel_sum += decode_mono10(data[int(index)], layout)
    template = pixel_sum / len(indices)

    grad_y, grad_x = np.gradient(template)
    gradient = np.hypot(grad_x, grad_y)
    inner = np.zeros_like(template, dtype=bool)
    margin_y, margin_x = max(8, height // 50), max(8, width // 50)
    inner[margin_y : height - margin_y, margin_x : width - margin_x] = True
    candidate = inner & (template >= min_dn) & (template <= max_dn)
    gradient_limit = max(2.0, float(np.percentile(gradient[candidate], 80)))
    flat_mask = candidate & (gradient <= gradient_limit)

    corrected_sum = np.zeros_like(template)
    corrected_sum_sq = np.zeros_like(template)
    scales: list[float] = []
    offsets: list[float] = []
    saturated = 0
    for index in indices:
        frame = decode_mono10(data[int(index)], layout)
        scale, offset = robust_affine(template, frame, flat_mask)
        if not 0.8 <= scale <= 1.2:
            raise ValueError(f"{path.name}: excessive frame brightness scale {scale:.4f}")
        corrected = (frame - offset) / scale
        corrected_sum += corrected
        corrected_sum_sq += corrected * corrected
        scales.append(scale)
        offsets.append(offset)
        saturated += int(np.count_nonzero(frame >= 1023))

    count = len(indices)
    mean_image = corrected_sum / count
    variance_image = (
        corrected_sum_sq - corrected_sum * corrected_sum / count
    ) / max(count - 1, 1)
    variance_image = np.maximum(variance_image, 0.0)

    rows: list[dict[str, float | int | bool]] = []
    edges = np.arange(min_dn, max_dn + bin_width, bin_width)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = flat_mask & (mean_image >= low) & (mean_image < high)
        pixels = int(selected.sum())
        if pixels < 500:
            continue
        means = mean_image[selected]
        variances = variance_image[selected]
        rows.append(
            {
                "mean_dn": float(np.median(means)),
                "variance_dn2": float(np.median(variances)),
                "variance_mean_dn2": float(np.mean(variances)),
                "pixels": pixels,
            }
        )

    if len(rows) < 5:
        raise ValueError(f"{path.name}: only {len(rows)} usable gray-level bins")
    x = np.array([float(row["mean_dn"]) for row in rows])
    y = np.array([float(row["variance_dn2"]) for row in rows])
    slope, intercept, fit_mask, r_squared = robust_line_fit(x, y)
    for row, used in zip(rows, fit_mask):
        row["used_for_fit"] = bool(used)

    output_dir = output_root / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ptc_gray_bins.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    read_noise = float(np.sqrt(max(intercept, 0.0)))
    report = [
        f"Source: {path}",
        f"Geometry: {width}x{height} Mono10 ({layout}), {fps:g} fps",
        f"Frames: {frame_count} total, {count} sampled",
        f"Usable flat pixels: {int(flat_mask.sum())}",
        f"Gray bins: {len(rows)} total, {int(fit_mask.sum())} used in robust fit",
        "",
        f"PTC: variance_dn2 = {slope:.8f} * mean_dn + {intercept:.8f}",
        f"R squared: {r_squared:.8f}",
        f"Read-noise intercept estimate: {read_noise:.6f} DN",
        "",
        f"Frame scale CV: {np.std(scales, ddof=1) / np.mean(scales) * 100:.6f}%",
        f"Frame offset std: {np.std(offsets, ddof=1):.6f} DN",
        f"Saturated samples: {saturated / (count * width * height) * 100:.6f}%",
        f"Flat-gradient threshold: {gradient_limit:.6f} DN/pixel",
    ]
    (output_dir / "ptc_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    axis.scatter(x[~fit_mask], y[~fit_mask], color="#a0a9b8", label="Rejected bins")
    axis.scatter(x[fit_mask], y[fit_mask], color="#0077b6", label="Used gray bins")
    line_x = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 200)
    axis.plot(
        line_x,
        slope * line_x + intercept,
        color="#d1495b",
        label=f"variance = {slope:.5f} × mean + {intercept:.3f}\nR² = {r_squared:.5f}",
    )
    axis.set(title=path.name, xlabel="Mean signal (DN)", ylabel="Temporal variance (DN²)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "ptc_validation.png", dpi=180)
    plt.close(figure)

    preview = np.clip(template / 4.0, 0, 255).astype(np.uint8)
    plt.imsave(output_dir / "mean_frame_preview.png", preview, cmap="gray", vmin=0, vmax=255)
    print(
        f"{path.name}: variance = {slope:.8f} * mean + {intercept:.8f}, "
        f"R2={r_squared:.6f}, bins={fit_mask.sum()}/{len(rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--bin-width", type=float, default=16.0)
    parser.add_argument("--min-dn", type=float, default=20.0)
    parser.add_argument("--max-dn", type=float, default=950.0)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW videos found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        analyse_video(
            path,
            args.output_dir,
            args.max_frames,
            args.bin_width,
            args.min_dn,
            args.max_dn,
        )


if __name__ == "__main__":
    main()
