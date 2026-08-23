"""Compare current best deploy (P150 BiShrink+SURE-k) vs PTC VST+BM3D."""

from __future__ import annotations

import argparse
import csv
import json
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
from .infer import load_model, save_mono10_png
from .infer_blend_deploy import denoise_blend_deploy
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
        exposure_ms, dark_variance_per_s=dark_variance_per_s
    )
    transformed = vst_forward(image_dn, intercept=intercept)
    denoised = tiled_bm3d(transformed, tile_size=tile_size, overlap=overlap)
    return vst_inverse(denoised, intercept=intercept).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\降噪素材"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_p150_vs_bm3d"),
    )
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edgekd",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.raw"))
    if args.max_files > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)} P150 vs VST+BM3D", flush=True)
    model_sota, n_in, _ = load_model(args.checkpoint_sota, None, device)
    model_edge, n_edge, _ = load_model(args.checkpoint_edgekd, None, device)
    if n_in != n_edge:
        raise SystemExit(f"Frame mismatch {n_in} vs {n_edge}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    summary: list[dict[str, float | str]] = []

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

        print(f"\n{path.name}: P150 deploy...", flush=True)
        deploy_dn, deploy_meta = denoise_blend_deploy(
            model_sota,
            model_edge,
            n_in,
            frames,
            target_index,
            float(fps),
            exposure_ms,
            device,
            tile=args.tile_size,
            temporal_window=args.temporal_window,
        )
        print(
            f"  ans_mode={deploy_meta.get('ans_mode')} route={deploy_meta.get('route')}",
            flush=True,
        )

        print(f"{path.name}: VST+BM3D...", flush=True)
        bm3d_dn = vst_bm3d(input_dn, exposure_ms, tile_size=args.bm3d_tile_size)

        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{target_index:03d}"
        outputs = {
            "input": input_dn,
            "temporal_reference": temporal,
            "vst_bm3d": bm3d_dn,
            "p150_deploy": deploy_dn,
        }
        for method, image in outputs.items():
            save_mono10_png(sequence_dir / f"{prefix}_{method}.png", image)
        save_preview(
            sequence_dir / f"{prefix}_comparison.png",
            list(outputs.values()),
            ["Input", "Temporal ref", "VST+BM3D", "P150 deploy"],
        )

        input_sigma = float(
            metric_row(
                path, float(fps), target_index, "input", input_dn, temporal, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
        clip_des: dict[str, float | str] = {"file": path.name, "fps": float(fps)}
        for method, image in outputs.items():
            row = metric_row(
                path,
                float(fps),
                target_index,
                method,
                image,
                temporal,
                mask,
                ys,
                xs,
                checkpoint=("p150" if method == "p150_deploy" else ""),
                input_frames=16 if method == "p150_deploy" else 1,
            )
            retention, edge_mae = edge_metrics_dn(image, temporal)
            sigma = float(row["roi_highpass_noise_sigma_dn"])
            des, noise_gain, edge_fidelity = denoise_edge_score(
                sigma, input_sigma, retention
            )
            row["edge_retention"] = f"{retention:.6f}"
            row["edge_sobel_mae"] = f"{edge_mae:.6f}"
            row["noise_gain"] = f"{noise_gain:.6f}"
            row["edge_fidelity"] = f"{edge_fidelity:.6f}"
            row["des"] = f"{des:.6f}"
            rows.append(row)
            if method in ("vst_bm3d", "p150_deploy", "input"):
                clip_des[f"{method}_des"] = float(des)
                clip_des[f"{method}_ng"] = float(noise_gain)
                clip_des[f"{method}_ef"] = float(edge_fidelity)
        summary.append(clip_des)
        delta = float(clip_des["p150_deploy_des"]) - float(clip_des["vst_bm3d_des"])
        print(
            f"  DES p150={clip_des['p150_deploy_des']:.4f} "
            f"bm3d={clip_des['vst_bm3d_des']:.4f} delta={delta:+.4f}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_dir / "summary_by_clip.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    mean_p150 = float(np.mean([s["p150_deploy_des"] for s in summary]))
    mean_bm3d = float(np.mean([s["vst_bm3d_des"] for s in summary]))
    out = {
        "mean_p150_des": mean_p150,
        "mean_bm3d_des": mean_bm3d,
        "delta": mean_p150 - mean_bm3d,
        "n_clips": len(summary),
        "model": "P150 BiShrink+SURE-k",
        "bm3d": "PTC VST+BM3D",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\n=== Mean DES ===", flush=True)
    print(f"P150 deploy: {mean_p150:.4f}", flush=True)
    print(f"VST+BM3D:    {mean_bm3d:.4f}", flush=True)
    print(f"Delta:       {mean_p150 - mean_bm3d:+.4f}", flush=True)
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
