"""Compare adaptive NAFNet against PTC-guided VST+BM3D on the same RAW frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .build_bm3d_teacher import tiled_bm3d
from .common import (
    DEFAULT_DARK_VARIANCE_PER_S,
    decode_mono10,
    exposure_intercept,
    memmap_frames,
    parse_exposure_ms,
    parse_geometry,
    vst_forward,
    vst_inverse,
)
from .infer import save_mono10_png
from .infer_adaptive import denoise_adaptive, load_adaptive_models
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview


def vst_bm3d(
    image_dn: np.ndarray,
    exposure_ms: float,
    tile_size: int = 512,
    overlap: int = 48,
    dark_variance_per_s: float = DEFAULT_DARK_VARIANCE_PER_S,
) -> np.ndarray:
    intercept = exposure_intercept(
        exposure_ms,
        dark_variance_per_s=dark_variance_per_s,
    )
    transformed = vst_forward(image_dn, intercept=intercept)
    denoised = tiled_bm3d(transformed, tile_size=tile_size, overlap=overlap)
    return vst_inverse(denoised, intercept=intercept).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_adaptive_vs_bm3d"),
    )
    parser.add_argument(
        "--checkpoint-1f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_1f_ptc_black60/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_ptc_black60/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-16f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-residual-4f",
        type=Path,
        default=None,
        help="Optional detail residual head for the 4f path.",
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files under {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)
    models = load_adaptive_models(
        args.checkpoint_1f,
        args.checkpoint_4f,
        args.checkpoint_16f,
        device,
        checkpoint_residual_4f=args.checkpoint_residual_4f,
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

        print(f"{path.name}: frame {target_index} adaptive...", flush=True)
        adaptive_dn, adaptive = denoise_adaptive(
            models,
            frames,
            target_index,
            fps,
            exposure_ms,
            device,
            tile_size=args.tile_size,
        )
        print(f"{path.name}: frame {target_index} vst+bm3d...", flush=True)
        bm3d_dn = vst_bm3d(
            input_dn,
            exposure_ms,
            tile_size=args.bm3d_tile_size,
        )

        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{target_index:03d}"
        outputs = {
            "input": input_dn,
            "temporal_reference": temporal,
            "vst_bm3d": bm3d_dn,
            "adaptive": adaptive_dn,
        }
        for method, image in outputs.items():
            save_mono10_png(sequence_dir / f"{prefix}_{method}.png", image)
        save_preview(
            sequence_dir / f"{prefix}_comparison.png",
            list(outputs.values()),
            [
                "Input",
                "Temporal ref",
                "VST+BM3D",
                f"Adaptive {adaptive['route']}",
            ],
        )

        method_images = {
            "input": input_dn,
            "temporal_reference": temporal,
            "vst_bm3d": bm3d_dn,
            "adaptive": adaptive_dn,
        }
        input_sigma = float(
            metric_row(
                path, fps, target_index, "input", input_dn, temporal, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
        for method, image in method_images.items():
            if method == "input":
                row = metric_row(
                    path, fps, target_index, method, image, temporal, mask, ys, xs
                )
            elif method == "temporal_reference":
                row = metric_row(
                    path, fps, target_index, method, image, temporal, mask, ys, xs
                )
            elif method == "vst_bm3d":
                row = metric_row(
                    path,
                    fps,
                    target_index,
                    method,
                    image,
                    temporal,
                    mask,
                    ys,
                    xs,
                    checkpoint="ptc_vst",
                    input_frames=1,
                )
            else:
                row = metric_row(
                    path,
                    fps,
                    target_index,
                    method,
                    image,
                    temporal,
                    mask,
                    ys,
                    xs,
                    checkpoint=str(adaptive["route"]),
                    input_frames=int(adaptive["used_frames"]),
                    alignment_response=float(adaptive["alignment_confidence"]),
                    fallback_used=bool(adaptive["fallback_used"]),
                )
            retention, edge_mae = edge_metrics_dn(image, temporal)
            sigma = float(row["roi_highpass_noise_sigma_dn"])
            des, noise_gain, edge_fidelity = denoise_edge_score(
                sigma,
                input_sigma,
                retention,
            )
            row["edge_retention"] = f"{retention:.6f}"
            row["edge_sobel_mae"] = f"{edge_mae:.6f}"
            row["noise_gain"] = f"{noise_gain:.6f}"
            row["edge_fidelity"] = f"{edge_fidelity:.6f}"
            row["des"] = f"{des:.6f}"
            rows.append(row)
        print(
            f"{path.name}: route={adaptive['route']} "
            f"adaptive_DES={rows[-1]['des']} bm3d_DES={rows[-2]['des']}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
