"""Validate NAFNet on held-out Mono10 RAW videos after training.

For each validation sequence the script:
1. Builds a brightness-aligned temporal trimmed mean as a proxy reference.
2. Runs the trained NAFNet checkpoint on the target frame.
3. Exports Mono10 PNGs, comparison previews, and a metrics CSV.

The temporal mean is the same proxy reference used in
``泊松分布/compare_ptc_advantage.py``; it is not independent ground truth.

Usage:
    # Run immediately with the latest checkpoint:
    python -m nafnet_denoise.validate \\
        --checkpoint nafnet_denoise/checkpoints/last.pt

    # Wait until 200-epoch training finishes, then validate:
    python -m nafnet_denoise.validate --wait --target-epochs 200
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import RAW_MAX, decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import denoise_frame, load_model, save_mono10_png

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_DIR = Path(r"D:\denoise\素材\训练素材\不参加训练，用来验证训练模型效果")
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "nafnet_denoise" / "checkpoints_4f"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "nafnet_denoise" / "validation_results_4f"


def central_roi(height: int, width: int, fraction: float = 0.40) -> tuple[slice, slice]:
    roi_height = int(height * fraction) // 2 * 2
    roi_width = int(width * fraction) // 2 * 2
    y0 = (height - roi_height) // 2
    x0 = (width - roi_width) // 2
    return slice(y0, y0 + roi_height), slice(x0, x0 + roi_width)


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
    half = window // 2
    start = max(0, min(target_index - half, frame_count - window))
    indices = list(range(start, min(frame_count, start + window)))
    if exclude_frame_index is not None:
        indices = [index for index in indices if index != exclude_frame_index]
    if len(indices) < 3:
        raise ValueError("Temporal trimmed mean needs at least three source frames.")
    target_mean = float(decode_mono10(data[target_index, ys, xs]).mean())
    frames = []
    for index in indices:
        image = decode_mono10(data[index])
        offset = float(image[ys, xs].mean()) - target_mean
        frames.append(image - offset)
    stack = np.stack(frames, axis=0)
    trim = max(0, int(len(indices) * trim_fraction))
    if trim:
        stack.sort(axis=0)
        stack = stack[trim:-trim]
    return np.clip(stack.mean(axis=0), 0.0, RAW_MAX).astype(np.float32)


def highpass_noise_sigma(image: np.ndarray, ys: slice, xs: slice) -> float:
    roi = image[ys, xs]
    lowpass = cv2.GaussianBlur(roi, (0, 0), 1.2)
    residual = roi - lowpass
    return float(np.median(np.abs(residual - np.median(residual))) / 0.6745)


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


def resolve_checkpoint(
    checkpoint: Path | None,
    checkpoint_dir: Path,
) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    last_path = checkpoint_dir / "last.pt"
    if last_path.exists():
        return last_path

    epoch_paths = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if epoch_paths:
        return epoch_paths[-1]

    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")


def read_last_epoch(train_log: Path) -> int | None:
    if not train_log.exists():
        return None
    rows = train_log.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) <= 1:
        return None
    return int(rows[-1].split(",")[0])


def wait_for_training(
    checkpoint_dir: Path,
    target_epochs: int,
    poll_seconds: int,
) -> Path:
    train_log = checkpoint_dir / "train_log.csv"
    print(
        f"Waiting for training to reach epoch {target_epochs - 1} "
        f"(poll every {poll_seconds}s) ..."
    )
    last_mtime = 0.0
    stable_rounds = 0
    while True:
        checkpoint = resolve_checkpoint(None, checkpoint_dir)
        epoch = read_last_epoch(train_log)
        checkpoint_mtime = checkpoint.stat().st_mtime
        finished = epoch is not None and epoch >= target_epochs - 1
        if finished and checkpoint_mtime == last_mtime:
            stable_rounds += 1
        else:
            stable_rounds = 0
        last_mtime = checkpoint_mtime

        if finished and stable_rounds >= 1:
            print(f"Training finished at epoch {epoch}; using {checkpoint}")
            return checkpoint

        if epoch is None:
            status = "no train_log yet"
        else:
            status = f"epoch {epoch}/{target_epochs - 1}"
        print(f"  [{datetime.now():%H:%M:%S}] {status}, checkpoint={checkpoint.name}")
        time.sleep(poll_seconds)


def process_video(
    path: Path,
    input_root: Path,
    output_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint_name: str,
    target_index: int,
    temporal_window: int,
    input_frames: int,
    tile: int,
    metrics: list[dict[str, str]],
) -> None:
    width, height, fps = parse_geometry(path.name)
    exposure_ms = parse_exposure_ms(path.name)
    frames = memmap_frames(path, width, height)
    frame_count = frames.shape[0]
    if target_index < 0 or target_index >= frame_count:
        raise ValueError(
            f"{path.name}: frame {target_index} is outside 0-{frame_count - 1}."
        )

    ys, xs = central_roi(height, width)
    input_image = decode_mono10(frames[target_index])
    temporal = aligned_temporal_trimmed_mean(
        frames,
        target_index,
        ys,
        xs,
        frame_count,
        temporal_window,
        trim_fraction=0.10,
        exclude_frame_index=target_index,
    )

    print(f"{path.name}: denoising frame {target_index} with {input_frames}-frame NAFNet ...")
    nafnet_image = np.clip(
        denoise_frame(
            model,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=input_frames,
            tile=tile,
        ),
        0.0,
        RAW_MAX,
    ).astype(np.float32)

    relative_stem = path.relative_to(input_root).with_suffix("")
    sequence_dir = output_root / "__".join(relative_stem.parts)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "input": input_image,
        "temporal_trimmed_mean": temporal,
        "nafnet": nafnet_image,
    }
    prefix = f"frame_{target_index:03d}"
    for method, image in outputs.items():
        save_mono10_png(sequence_dir / f"{prefix}_{method}_mono10.png", image)

    save_preview(
        sequence_dir / f"{prefix}_comparison_preview.png",
        list(outputs.values()),
        ["Input", "Temporal mean", "NAFNet"],
    )

    for method, image in outputs.items():
        metrics.append(
            {
                "file": path.name,
                "fps": f"{fps:g}",
                "frame_index": str(target_index),
                "method": method,
                "checkpoint": checkpoint_name if method == "nafnet" else "",
                "roi_mean_dn": f"{float(image[ys, xs].mean()):.6f}",
                "roi_highpass_noise_sigma_dn": f"{highpass_noise_sigma(image, ys, xs):.6f}",
                "mae_to_temporal_reference_dn": f"{float(np.mean(np.abs(image - temporal))):.6f}",
                "gradient_mae_to_temporal_reference": f"{gradient_error(image, temporal):.6f}",
            }
        )


def write_report(output_dir: Path, metrics: list[dict[str, str]], checkpoint: Path) -> None:
    by_file: dict[str, dict[str, dict[str, str]]] = {}
    for row in metrics:
        by_file.setdefault(row["file"], {})[row["method"]] = row

    lines = [
        "NAFNet Mono10 validation report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Checkpoint: {checkpoint}",
        "",
    ]

    for file_name, methods in sorted(by_file.items()):
        input_row = methods["input"]
        temporal_row = methods["temporal_trimmed_mean"]
        nafnet_row = methods["nafnet"]
        input_sigma = float(input_row["roi_highpass_noise_sigma_dn"])
        nafnet_sigma = float(nafnet_row["roi_highpass_noise_sigma_dn"])
        temporal_sigma = float(temporal_row["roi_highpass_noise_sigma_dn"])
        noise_reduction = (1.0 - nafnet_sigma / max(input_sigma, 1e-6)) * 100.0
        temporal_gap = float(nafnet_row["mae_to_temporal_reference_dn"])

        lines.extend(
            [
                f"Sequence: {file_name}",
                f"  ROI high-pass sigma: input={input_sigma:.3f}, "
                f"nafnet={nafnet_sigma:.3f}, temporal={temporal_sigma:.3f} DN",
                f"  Noise reduction vs input: {noise_reduction:.1f}%",
                f"  MAE to temporal reference: {temporal_gap:.3f} DN",
                f"  Gradient MAE to temporal reference: "
                f"{float(nafnet_row['gradient_mae_to_temporal_reference']):.3f}",
                "",
            ]
        )

    report_path = output_dir / "validation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory containing held-out Mono10 RAW validation videos.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint (.pt). Defaults to checkpoints/last.pt.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--input-frames", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until training reaches --target-epochs before validating.",
    )
    parser.add_argument("--target-epochs", type=int, default=200)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.wait:
        checkpoint_path = wait_for_training(
            args.checkpoint_dir,
            args.target_epochs,
            args.poll_seconds,
        )
    else:
        checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    model, input_frames, _width = load_model(
        checkpoint_path,
        args.width,
        device,
        input_frames=args.input_frames,
    )
    print(f"Input frames: {input_frames}")

    metrics: list[dict[str, str]] = []
    for path in files:
        process_video(
            path,
            args.input_dir,
            args.output_dir,
            model,
            device,
            checkpoint_path.name,
            args.frame_index,
            args.temporal_window,
            input_frames,
            args.tile,
            metrics,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved {len(metrics)} metric rows to {metrics_path}")
    write_report(args.output_dir, metrics, checkpoint_path)


if __name__ == "__main__":
    main()
