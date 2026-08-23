"""P1-a: fps / edge-aware blend of gated SOTA ↔ edgeKD DualHead.

Shared FE: soft-gate temporal Wiener + fps/σ-gated spatial (same as P0-a).
Neural: frozen split_edge_fid (SOTA mean) vs edge_kd_new_fe (better f10 ef).

Blends (no training):
  - fps_route: fps>=high → edgeKD; fps<=low → SOTA; else 0.5 mix
  - edge_soft: edge*edgeKD + flat*SOTA (guide=mean of both)
  - fps_edge: edge_soft with fps-scaled edge weight
             (high fps → more edgeKD on edges; low fps → mostly SOTA)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import gated_spatial_from_temporal
from .infer import denoise_from_dn_frames, load_model, save_mono10_png
from .infer_ensemble import soft_edge_mask_dn
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


def blend_edge_soft(
    sota_dn: np.ndarray,
    edgekd_dn: np.ndarray,
    guide_dn: np.ndarray,
    temperature: float = 8.0,
    edge_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``out = (w*edge)*edgeKD + (1-w*edge)*SOTA`` with ``w`` in [0,1]."""
    edge_map = soft_edge_mask_dn(guide_dn, temperature=temperature)
    w = float(np.clip(edge_weight, 0.0, 1.0))
    edge_eff = w * edge_map
    blended = edge_eff * edgekd_dn + (1.0 - edge_eff) * sota_dn
    return blended.astype(np.float32), edge_map.astype(np.float32)


def fps_edge_weight(
    fps: float,
    low_fps: float = 1.0,
    high_fps: float = 8.0,
) -> float:
    """0 at/below low_fps → 1 at/above high_fps (linear mid)."""
    if fps <= low_fps:
        return 0.0
    if fps >= high_fps:
        return 1.0
    return (float(fps) - low_fps) / max(high_fps - low_fps, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_sota_edgekd_blend"),
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
    for ckpt in (args.checkpoint_sota, args.checkpoint_edgekd):
        if not ckpt.exists():
            raise SystemExit(f"Missing {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)
    model_sota, n_sota, _ = load_model(args.checkpoint_sota, None, device)
    model_kd, n_kd, _ = load_model(args.checkpoint_edgekd, None, device)
    if n_sota != n_kd:
        raise SystemExit(f"Frame mismatch: sota={n_sota} edgekd={n_kd}")
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

        print(f"{path.name}: sota + edgekd...", flush=True)
        sota_dn = _spatial_on_merged(
            model_sota, fe_gated, exposure_ms, device, n_in, args.tile_size
        )
        edgekd_dn = _spatial_on_merged(
            model_kd, fe_gated, exposure_ms, device, n_in, args.tile_size
        )

        w_fps = fps_edge_weight(float(fps), args.low_fps, args.high_fps)
        if float(fps) >= args.high_fps:
            fps_route = edgekd_dn
            route_name = "edgekd"
        elif float(fps) <= args.low_fps:
            fps_route = sota_dn
            route_name = "sota"
        else:
            fps_route = (0.5 * sota_dn + 0.5 * edgekd_dn).astype(np.float32)
            route_name = "mix0.5"

        guide = (0.5 * sota_dn + 0.5 * edgekd_dn).astype(np.float32)
        edge_soft, _ = blend_edge_soft(
            sota_dn,
            edgekd_dn,
            guide,
            temperature=args.edge_temperature,
            edge_weight=1.0,
        )
        fps_edge, _ = blend_edge_soft(
            sota_dn,
            edgekd_dn,
            guide,
            temperature=args.edge_temperature,
            edge_weight=w_fps,
        )

        outputs: dict[str, np.ndarray] = {
            "input": input_dn,
            "temporal_reference": temporal_ref,
            "sota_gated": sota_dn,
            "edgekd_gated": edgekd_dn,
            "blend_fps_route": fps_route,
            "blend_edge_soft": edge_soft,
            "blend_fps_edge": fps_edge,
        }
        method_meta: dict[str, tuple[int, str]] = {
            "input": (1, "raw"),
            "temporal_reference": (args.temporal_window, "trimmed_mean"),
            "sota_gated": (args.temporal_window, "split_edge_fid+gated"),
            "edgekd_gated": (args.temporal_window, "edge_kd_new_fe+gated"),
            "blend_fps_route": (args.temporal_window, f"fps_route:{route_name}"),
            "blend_edge_soft": (args.temporal_window, "edge*kd+flat*sota"),
            "blend_fps_edge": (
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
            "sota_gated",
            "edgekd_gated",
            "blend_fps_route",
            "blend_edge_soft",
            "blend_fps_edge",
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
            f"  DES sota={des_by.get('sota_gated', 0):.4f} "
            f"kd={des_by.get('edgekd_gated', 0):.4f} "
            f"fps_route={des_by.get('blend_fps_route', 0):.4f} "
            f"edge_soft={des_by.get('blend_edge_soft', 0):.4f} "
            f"fps_edge={des_by.get('blend_fps_edge', 0):.4f} "
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
        "refs: gated SOTA 0.9050 / edgeKD gated 0.9043 / "
        "goal mean>=0.9050 & f10 ef>=0.968",
        flush=True,
    )


if __name__ == "__main__":
    main()
