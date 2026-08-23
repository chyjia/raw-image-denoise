"""P2-d: zero-train fps / edge fuse of blend_edge_soft ↔ σ-FiLM.

Shared gated FE. Arms:
  - blend_edge_soft: edge*edgeKD + flat*SOTA  (deploy SOTA mean 0.9055)
  - film_gated: σ-FiLM DualHead alone
  - fuse_fps: fps>=high → FiLM; fps<=low → blend; else 0.5
  - fuse_edge: edge*FiLM + flat*blend
  - fuse_fps_edge: same with fps-scaled edge weight toward FiLM
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .compare_sota_edgekd_blend import (
    _spatial_on_merged,
    blend_edge_soft,
    fps_edge_weight,
)
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import gated_spatial_from_temporal
from .infer import load_model, save_mono10_png
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview
from .wiener_merge import merge_from_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_blend_film_fuse"),
    )
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edgekd",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_edge_kd_new_fe/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-film",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_sigma_film/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    parser.add_argument("--wiener-tile", type=int, default=32)
    parser.add_argument("--wiener-overlap", type=int, default=16)
    parser.add_argument("--c-factor", type=float, default=8.0)
    parser.add_argument("--edge-temperature", type=float, default=8.0)
    parser.add_argument("--low-fps", type=float, default=1.0)
    parser.add_argument("--high-fps", type=float, default=8.0)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW under {args.input_dir}")
    for ckpt in (args.checkpoint_sota, args.checkpoint_edgekd, args.checkpoint_film):
        if not ckpt.exists():
            raise SystemExit(f"Missing {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)
    model_sota, n_sota, _ = load_model(args.checkpoint_sota, None, device)
    model_kd, n_kd, _ = load_model(args.checkpoint_edgekd, None, device)
    model_film, n_film, _ = load_model(args.checkpoint_film, None, device)
    if len({n_sota, n_kd, n_film}) != 1:
        raise SystemExit(f"Frame mismatch: {n_sota}/{n_kd}/{n_film}")
    n_in = n_sota

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
        fe_gated, params, fe_sigma = gated_spatial_from_temporal(
            temporal,
            fps=float(fps),
            n_frames_averaged=args.temporal_window,
            tile_size=args.wiener_tile,
            overlap=args.wiener_overlap,
        )
        print(
            f"{path.name}: fps={fps:g} fe_sigma={fe_sigma:.4f} -> {params.name}",
            flush=True,
        )

        print(f"{path.name}: sota + edgekd + film...", flush=True)
        sota_dn = _spatial_on_merged(
            model_sota, fe_gated, exposure_ms, device, n_in, args.tile_size
        )
        edgekd_dn = _spatial_on_merged(
            model_kd, fe_gated, exposure_ms, device, n_in, args.tile_size
        )
        film_dn = _spatial_on_merged(
            model_film, fe_gated, exposure_ms, device, n_in, args.tile_size
        )

        guide_sk = (0.5 * sota_dn + 0.5 * edgekd_dn).astype(np.float32)
        blend_dn, _ = blend_edge_soft(
            sota_dn,
            edgekd_dn,
            guide_sk,
            temperature=args.edge_temperature,
            edge_weight=1.0,
        )

        w_fps = fps_edge_weight(float(fps), args.low_fps, args.high_fps)
        if float(fps) >= args.high_fps:
            fuse_fps = film_dn
            route_name = "film"
        elif float(fps) <= args.low_fps:
            fuse_fps = blend_dn
            route_name = "blend"
        else:
            fuse_fps = (0.5 * blend_dn + 0.5 * film_dn).astype(np.float32)
            route_name = "mix0.5"

        guide_bf = (0.5 * blend_dn + 0.5 * film_dn).astype(np.float32)
        fuse_edge, _ = blend_edge_soft(
            blend_dn,
            film_dn,
            guide_bf,
            temperature=args.edge_temperature,
            edge_weight=1.0,
        )
        fuse_fps_edge, _ = blend_edge_soft(
            blend_dn,
            film_dn,
            guide_bf,
            temperature=args.edge_temperature,
            edge_weight=w_fps,
        )

        outputs: dict[str, np.ndarray] = {
            "input": input_dn,
            "temporal_reference": temporal_ref,
            "blend_edge_soft": blend_dn,
            "film_gated": film_dn,
            "fuse_fps": fuse_fps,
            "fuse_edge": fuse_edge,
            "fuse_fps_edge": fuse_fps_edge,
        }
        method_meta: dict[str, tuple[int, str]] = {
            "input": (1, "raw"),
            "temporal_reference": (args.temporal_window, "trimmed_mean"),
            "blend_edge_soft": (args.temporal_window, "edge*kd+flat*sota"),
            "film_gated": (args.temporal_window, "sigma_film+gated"),
            "fuse_fps": (args.temporal_window, f"fps_route:{route_name}"),
            "fuse_edge": (args.temporal_window, "edge*film+flat*blend"),
            "fuse_fps_edge": (
                args.temporal_window,
                f"fps_edge:w={w_fps:.2f}",
            ),
        }

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
            "blend_edge_soft",
            "film_gated",
            "fuse_fps",
            "fuse_edge",
            "fuse_fps_edge",
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
            row["fps_edge_w"] = f"{w_fps:.4f}"
            rows.append(row)
            des_by[method] = des

        print(
            f"  DES blend={des_by.get('blend_edge_soft', 0):.4f} "
            f"film={des_by.get('film_gated', 0):.4f} "
            f"fuse_fps={des_by.get('fuse_fps', 0):.4f} "
            f"fuse_edge={des_by.get('fuse_edge', 0):.4f} "
            f"fuse_fps_edge={des_by.get('fuse_fps_edge', 0):.4f} "
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
    print(
        "refs: blend_edge_soft 0.9055 / film gated 0.9020 / "
        "goal mean>=0.9055 & f10 ef>=0.968",
        flush=True,
    )


if __name__ == "__main__":
    main()
