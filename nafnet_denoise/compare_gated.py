"""Compare dual+NL ungated vs alignment-gated inference vs Adaptive / BM3D."""

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
from .infer_gated import denoise_gated
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_align_gated"),
    )
    parser.add_argument(
        "--checkpoint-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_dual_head_nonlocal/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-baseline-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt"),
    )
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
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.015)
    parser.add_argument("--soft-lo", type=float, default=0.35)
    parser.add_argument("--soft-hi", type=float, default=0.85)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW under {args.input_dir}")
    for path in (args.checkpoint_4f, args.checkpoint_1f, args.checkpoint_baseline_4f):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)

    model_4f, frames_4f, _ = load_model(args.checkpoint_4f, None, device)
    model_1f, _frames_1f, _ = load_model(args.checkpoint_1f, None, device)
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

        print(f"{path.name}: dual_nl...", flush=True)
        dual_dn = denoise_frame(
            model_4f,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=frames_4f,
            tile=args.tile_size,
        )
        outputs["dual_nl_4f"] = dual_dn

        print(f"{path.name}: gated...", flush=True)
        gated_dn, gated_meta = denoise_gated(
            model_4f,
            model_1f,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=frames_4f,
            align_threshold=args.align_threshold,
            temperature=args.temperature,
            soft_lo=args.soft_lo,
            soft_hi=args.soft_hi,
            tile_size=args.tile_size,
        )
        outputs["gated_4f"] = gated_dn

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
        method_meta = {
            "adaptive": (
                int(adaptive_meta.get("used_frames", 0)),
                str(adaptive_meta.get("route", "adaptive")),
            ),
            "dual_nl_4f": (frames_4f, "dual_nl_4f"),
            "gated_4f": (
                int(gated_meta.get("used_frames", frames_4f)),
                str(gated_meta.get("route", "gated")),
            ),
            "vst_bm3d": (1, "ptc_vst"),
        }
        des_by: dict[str, float] = {}
        for method, image in outputs.items():
            frames_used, checkpoint = method_meta.get(method, (0, ""))
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
            if method == "gated_4f":
                row["alignment_response_median"] = f"{float(gated_meta['alignment_confidence']):.6f}"
            rows.append(row)
            des_by[method] = des

        print(
            f"  DES adaptive={des_by.get('adaptive', 0):.4f} "
            f"dual_nl={des_by.get('dual_nl_4f', 0):.4f} "
            f"gated={des_by.get('gated_4f', 0):.4f} "
            f"bm3d={des_by.get('vst_bm3d', 0):.4f} "
            f"fusion_w={float(gated_meta['fusion_weight']):.3f} "
            f"route={gated_meta['route']}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {metrics_path}", flush=True)

    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(float(row["des"]))
    for method, values in sorted(by_method.items()):
        if method in ("input", "temporal_reference"):
            continue
        print(f"mean DES {method}: {sum(values) / len(values):.4f} (n={len(values)})")


if __name__ == "__main__":
    main()
