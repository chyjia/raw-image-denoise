"""Ablate HDR+ Wiener merge vs gated DualHead / BM3D (DES)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import denoise_from_dn_frames, denoise_frame, load_model, save_mono10_png
from .infer_gated import denoise_gated
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview
from .wiener_merge import merge_from_memmap


def _spatial_on_merged(
    model,
    merged: np.ndarray,
    exposure_ms: float,
    device: torch.device,
    input_frames: int,
    tile: int,
) -> np.ndarray:
    """Run a multi-frame model on repeated merged frames (spatial residual only)."""
    frames = [merged.copy() for _ in range(input_frames)]
    # Avoid re-running Wiener inside denoise_from_dn_frames when caller already merged.
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    try:
        return denoise_from_dn_frames(model, frames, exposure_ms, device, tile=tile)
    finally:
        model.wiener_front_end = was


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_wiener_merge"),
    )
    parser.add_argument(
        "--checkpoint-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_des_align/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-1f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_1f_ptc_black60/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    parser.add_argument("--wiener-tile", type=int, default=32)
    parser.add_argument("--wiener-overlap", type=int, default=16)
    parser.add_argument("--c-factor", type=float, default=8.0)
    parser.add_argument(
        "--spatial-wiener",
        action="store_true",
        help="Also evaluate temporal+spatial Wiener hybrid variants.",
    )
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.015)
    parser.add_argument("--soft-lo", type=float, default=0.35)
    parser.add_argument("--soft-hi", type=float, default=0.85)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW under {args.input_dir}")
    if not args.checkpoint_4f.exists():
        raise SystemExit(f"Missing {args.checkpoint_4f}")
    if not args.checkpoint_1f.exists():
        raise SystemExit(f"Missing {args.checkpoint_1f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)
    model_4f, frames_4f, _ = load_model(args.checkpoint_4f, None, device)
    model_1f, frames_1f, _ = load_model(args.checkpoint_1f, None, device)

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

        outputs: dict[str, np.ndarray] = {
            "input": input_dn,
            "temporal_reference": temporal,
        }
        method_meta: dict[str, tuple[int, str]] = {
            "vst_bm3d": (1, "ptc_vst"),
            "gated_4f": (frames_4f, "gated_des_align"),
        }

        print(f"{path.name}: bm3d...", flush=True)
        outputs["vst_bm3d"] = vst_bm3d(
            input_dn, exposure_ms, tile_size=args.bm3d_tile_size
        )

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
        method_meta["gated_4f"] = (
            int(gated_meta.get("used_frames", frames_4f)),
            "gated_des_align",
        )

        for n_frames in (4, 8, 16):
            print(f"{path.name}: wiener_{n_frames}f...", flush=True)
            merged, merge_meta = merge_from_memmap(
                frames,
                target_index,
                input_frames=n_frames,
                tile_size=args.wiener_tile,
                overlap=args.wiener_overlap,
                c_factor=args.c_factor,
                align=True,
                spatial_wiener=False,
            )
            key = f"wiener_{n_frames}f"
            outputs[key] = merged
            method_meta[key] = (n_frames, "hdrplus_wiener")

            merged_sp = None
            if args.spatial_wiener and n_frames in (8, 16):
                print(f"{path.name}: wiener_{n_frames}f_spatial...", flush=True)
                merged_sp, _ = merge_from_memmap(
                    frames,
                    target_index,
                    input_frames=n_frames,
                    tile_size=args.wiener_tile,
                    overlap=args.wiener_overlap,
                    c_factor=args.c_factor,
                    align=True,
                    spatial_wiener=True,
                )
                key_sp = f"wiener_{n_frames}f_spatial"
                outputs[key_sp] = merged_sp
                method_meta[key_sp] = (n_frames, "hdrplus_wiener_spatial")

            if n_frames in (8, 16):
                print(f"{path.name}: wiener_{n_frames}f + dual_spatial...", flush=True)
                spatial_dual = _spatial_on_merged(
                    model_4f,
                    merged,
                    exposure_ms,
                    device,
                    input_frames=frames_4f,
                    tile=args.tile_size,
                )
                key_dual = f"wiener_{n_frames}f_dual"
                outputs[key_dual] = spatial_dual
                method_meta[key_dual] = (n_frames, "wiener+dual_spatial")

                if merged_sp is not None:
                    print(
                        f"{path.name}: wiener_{n_frames}f_spatial + dual...",
                        flush=True,
                    )
                    spatial_dual_sp = _spatial_on_merged(
                        model_4f,
                        merged_sp,
                        exposure_ms,
                        device,
                        input_frames=frames_4f,
                        tile=args.tile_size,
                    )
                    key_dual_sp = f"wiener_{n_frames}f_spatial_dual"
                    outputs[key_dual_sp] = spatial_dual_sp
                    method_meta[key_dual_sp] = (n_frames, "wiener_spatial+dual")

        # End-to-end pipeline from checkpoint flags (Wiener front-end FT models).
        if getattr(model_4f, "wiener_front_end", False):
            print(f"{path.name}: pipeline_ft...", flush=True)
            pipeline_dn = denoise_frame(
                model_4f,
                frames,
                target_index,
                exposure_ms,
                device,
                input_frames=frames_4f,
                tile=args.tile_size,
            )
            outputs["pipeline_ft"] = pipeline_dn
            method_meta["pipeline_ft"] = (
                int(getattr(model_4f, "wiener_merge_frames", frames_4f)),
                "wiener_front_end_ft",
            )

        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{target_index:03d}"
        # Save a compact subset of previews to limit disk use.
        preview_keys = [
            "input",
            "gated_4f",
            "wiener_16f_dual",
            "wiener_16f_spatial_dual",
            "pipeline_ft",
            "vst_bm3d",
            "temporal_reference",
        ]
        for method, image in outputs.items():
            if method in preview_keys:
                save_mono10_png(sequence_dir / f"{prefix}_{method}.png", image)
        save_preview(
            sequence_dir / f"{prefix}_comparison.png",
            [outputs[k] for k in preview_keys if k in outputs],
            [k for k in preview_keys if k in outputs],
        )

        input_sigma = float(
            metric_row(
                path, fps, target_index, "input", input_dn, temporal, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
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
            rows.append(row)
            des_by[method] = des

        print(
            f"  DES gated={des_by.get('gated_4f', 0):.4f} "
            f"w16+dual={des_by.get('wiener_16f_dual', 0):.4f} "
            f"w16+sp+dual={des_by.get('wiener_16f_spatial_dual', 0):.4f} "
            f"pipe={des_by.get('pipeline_ft', 0):.4f} "
            f"bm3d={des_by.get('vst_bm3d', 0):.4f}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("--- mean DES ---", flush=True)
    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(float(row["des"]))
    for method, values in sorted(by_method.items()):
        print(f"{method}: {sum(values) / len(values):.4f} (n={len(values)})", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
