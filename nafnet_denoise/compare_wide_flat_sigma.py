"""Compare baseline / wide NAFNet / stacked Restormer 4f vs VST+BM3D."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import denoise_frame, load_model, save_mono10_png
from .infer_adaptive import denoise_adaptive, load_adaptive_models
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_wide_flat_sigma"),
    )
    parser.add_argument("--checkpoint-baseline-4f", type=Path, required=True)
    parser.add_argument("--checkpoint-wide-4f", type=Path, default=None)
    parser.add_argument(
        "--label-wide-4f",
        type=str,
        default="nafnet_wide_4f",
        help="Method name for --checkpoint-wide-4f in metrics.csv",
    )
    parser.add_argument("--checkpoint-stacked-restormer", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-1f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_1f_ptc_black60/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-16f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW under {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)

    models: dict[str, tuple[torch.nn.Module, int]] = {}
    baseline, frames_b, _ = load_model(args.checkpoint_baseline_4f, None, device)
    models["baseline_4f"] = (baseline, frames_b)
    if args.checkpoint_wide_4f is not None and args.checkpoint_wide_4f.exists():
        wide, frames_w, _ = load_model(args.checkpoint_wide_4f, None, device)
        models[args.label_wide_4f] = (wide, frames_w)
    if (
        args.checkpoint_stacked_restormer is not None
        and args.checkpoint_stacked_restormer.exists()
    ):
        resto, frames_r, _ = load_model(args.checkpoint_stacked_restormer, None, device)
        models["stacked_restormer_4f"] = (resto, frames_r)

    adaptive_models = None
    if args.checkpoint_1f.exists() and args.checkpoint_16f.exists():
        adaptive_models = load_adaptive_models(
            args.checkpoint_1f,
            args.checkpoint_baseline_4f,
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

        outputs: dict[str, object] = {
            "input": input_dn,
            "temporal_reference": temporal,
        }
        print(f"{path.name}: bm3d...", flush=True)
        outputs["vst_bm3d"] = vst_bm3d(
            input_dn, exposure_ms, tile_size=args.bm3d_tile_size
        )

        if adaptive_models is not None:
            print(f"{path.name}: adaptive...", flush=True)
            adaptive_dn, adaptive_meta = denoise_adaptive(
                adaptive_models,
                frames,
                target_index,
                fps,
                exposure_ms,
                device,
                tile_size=args.tile_size,
            )
            outputs["adaptive"] = adaptive_dn
        else:
            adaptive_meta = {}

        for name, (model, input_frames) in models.items():
            print(f"{path.name}: {name}...", flush=True)
            outputs[name] = denoise_frame(
                model,
                frames,
                target_index,
                exposure_ms,
                device,
                input_frames=input_frames,
                tile=args.tile_size,
            )

        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{target_index:03d}"
        for method, image in outputs.items():
            save_mono10_png(sequence_dir / f"{prefix}_{method}.png", image)
        save_preview(
            sequence_dir / f"{prefix}_comparison.png",
            list(outputs.values()),
            list(outputs.keys()),
        )

        input_sigma = float(
            metric_row(
                path, fps, target_index, "input", input_dn, temporal, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
        for method, image in outputs.items():
            frames_used = 0
            checkpoint = ""
            if method == "vst_bm3d":
                frames_used, checkpoint = 1, "ptc_vst"
            elif method == "adaptive":
                frames_used = int(adaptive_meta.get("used_frames", 0))
                checkpoint = str(adaptive_meta.get("route", "adaptive"))
            elif method in models:
                frames_used = models[method][1]
                checkpoint = method
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
                checkpoint=checkpoint,
                input_frames=frames_used,
            )
            retention, edge_mae = edge_metrics_dn(image, temporal)
            des, noise_gain, edge_fid = denoise_edge_score(
                float(row["roi_highpass_noise_sigma_dn"]),
                input_sigma,
                retention,
            )
            row["edge_retention"] = f"{retention:.6f}"
            row["edge_sobel_mae"] = f"{edge_mae:.6f}"
            row["noise_gain"] = f"{noise_gain:.6f}"
            row["edge_fidelity"] = f"{edge_fid:.6f}"
            row["des"] = f"{des:.6f}"
            rows.append(row)

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
