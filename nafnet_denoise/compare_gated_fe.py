"""P0-a: fps/σ-gated FE vs fixed ea_mild_gamma / baseline on frozen DualHead."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import (
    BASELINE_SPATIAL,
    EA_MILD_GAMMA,
    apply_spatial_fe,
    gated_spatial_from_temporal,
)
from .infer import denoise_from_dn_frames, load_model, save_mono10_png
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
    frames = [merged.copy() for _ in range(input_frames)]
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
        default=Path("nafnet_denoise/compare_gated_fe"),
    )
    parser.add_argument(
        "--checkpoint-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    parser.add_argument("--wiener-tile", type=int, default=32)
    parser.add_argument("--wiener-overlap", type=int, default=16)
    parser.add_argument("--c-factor", type=float, default=8.0)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW under {args.input_dir}")
    if not args.checkpoint_4f.exists():
        raise SystemExit(f"Missing {args.checkpoint_4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)
    model, n_in, _ = load_model(args.checkpoint_4f, None, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for path in files:
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        target_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
        ys, xs = central_roi(height, width)
        input_dn = decode_mono10(frames[target_index])
        temporal_ref = aligned_temporal_trimmed_mean(
            frames,
            target_index,
            ys,
            xs,
            frames.shape[0],
            min(args.temporal_window, frames.shape[0]),
            trim_fraction=0.10,
            exclude_frame_index=target_index,
        )
        mask = reliable_reference_mask(input_dn, temporal_ref)

        print(f"{path.name}: temporal Wiener...", flush=True)
        temporal, _ = merge_from_memmap(
            frames,
            target_index,
            input_frames=args.temporal_window,
            tile_size=args.wiener_tile,
            overlap=args.wiener_overlap,
            c_factor=args.c_factor,
            align=True,
            spatial_wiener=False,
        )

        variants: dict[str, tuple[np.ndarray, str, float]] = {}
        # Fixed SOTA FE
        fe_fixed = apply_spatial_fe(
            temporal,
            EA_MILD_GAMMA,
            n_frames_averaged=args.temporal_window,
            tile_size=args.wiener_tile,
            overlap=args.wiener_overlap,
        )
        variants["fixed_ea_mild_gamma"] = (fe_fixed, EA_MILD_GAMMA.name, -1.0)

        fe_base = apply_spatial_fe(
            temporal,
            BASELINE_SPATIAL,
            n_frames_averaged=args.temporal_window,
            tile_size=args.wiener_tile,
            overlap=args.wiener_overlap,
        )
        variants["fixed_baseline"] = (fe_base, BASELINE_SPATIAL.name, -1.0)

        fe_gated, params, fe_sigma = gated_spatial_from_temporal(
            temporal,
            fps=float(fps),
            n_frames_averaged=args.temporal_window,
            tile_size=args.wiener_tile,
            overlap=args.wiener_overlap,
        )
        variants["gated_fps_sigma"] = (fe_gated, params.name, fe_sigma)
        print(
            f"{path.name}: fps={fps:g} fe_sigma={fe_sigma:.4f} -> {params.name}",
            flush=True,
        )

        outputs: dict[str, np.ndarray] = {
            "input": input_dn,
            "temporal_reference": temporal_ref,
        }
        method_meta: dict[str, tuple[int, str]] = {
            "input": (1, "raw"),
            "temporal_reference": (args.temporal_window, "trimmed_mean"),
        }

        for name, (merged, sched, _sig) in variants.items():
            dual = _spatial_on_merged(
                model, merged, exposure_ms, device, n_in, args.tile_size
            )
            key = f"dual_{name}"
            outputs[key] = dual
            method_meta[key] = (args.temporal_window, f"{name}:{sched}")

        print(f"{path.name}: vst_bm3d...", flush=True)
        outputs["vst_bm3d"] = vst_bm3d(
            input_dn, exposure_ms, tile_size=args.bm3d_tile_size
        )
        method_meta["vst_bm3d"] = (1, "bm3d")

        sequence_dir = args.output_dir / path.stem
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{target_index:03d}"
        preview_keys = [
            "input",
            "dual_fixed_baseline",
            "dual_fixed_ea_mild_gamma",
            "dual_gated_fps_sigma",
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
                path, fps, target_index, "input", input_dn, temporal_ref, mask, ys, xs
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
                temporal_ref,
                mask,
                ys,
                xs,
                checkpoint=checkpoint,
                input_frames=frames_used,
            )
            retention, edge_mae = edge_metrics_dn(image, temporal_ref)
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
            row["fe_sigma_dn"] = f"{fe_sigma:.6f}"
            row["gated_schedule"] = params.name
            rows.append(row)
            des_by[method] = des

        print(
            f"  DES gated={des_by.get('dual_gated_fps_sigma', 0):.4f} "
            f"ea_γ={des_by.get('dual_fixed_ea_mild_gamma', 0):.4f} "
            f"base={des_by.get('dual_fixed_baseline', 0):.4f} "
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
    for method, values in sorted(
        by_method.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True
    ):
        if method in {"input", "temporal_reference"}:
            continue
        print(f"{method}: {sum(values) / len(values):.4f} (n={len(values)})", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
