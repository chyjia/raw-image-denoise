"""FE-only DES sweep: edge-aware spatial Wiener γ on frozen split_edge_fid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import denoise_from_dn_frames, load_model, save_mono10_png
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview
from .wiener_merge import merge_from_memmap

# (name, flat_c_mult, edge_c_mult, dark_boost, mask_harden, freq_gamma, adaptive)
CONFIGS: list[tuple[str, float, float, float, float, float, bool]] = [
    ("baseline_spatial", 1.0, 1.0, 0.0, 0.0, 0.0, False),
    ("ea_mild", 1.75, 0.45, 0.35, 8.0, 0.0, True),
    ("ea_mild_gamma", 1.75, 0.45, 0.35, 8.0, 0.15, True),
    ("ea_mid", 2.0, 0.40, 0.40, 10.0, 0.10, True),
    ("ea_protect", 2.25, 0.30, 0.50, 12.0, 0.12, True),
]


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
        default=Path("nafnet_denoise/compare_edge_aware_wiener"),
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
    print(f"device={device} files={len(files)} configs={len(CONFIGS)}", flush=True)
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
            "input": (1, "raw"),
            "temporal_reference": (args.temporal_window, "trimmed_mean"),
        }

        for name, flat_m, edge_m, dark_b, harden, gamma, adaptive in CONFIGS:
            print(f"{path.name}: {name}...", flush=True)
            merged, _ = merge_from_memmap(
                frames,
                target_index,
                input_frames=args.temporal_window,
                tile_size=args.wiener_tile,
                overlap=args.wiener_overlap,
                c_factor=args.c_factor,
                align=True,
                spatial_wiener=True,
                spatial_c_factor=1.0,
                spatial_adaptive=adaptive,
                spatial_flat_c_mult=flat_m,
                spatial_dark_boost=dark_b,
                spatial_edge_c_mult=edge_m,
                spatial_mask_harden=harden,
                spatial_freq_gamma=gamma,
            )
            dual = _spatial_on_merged(
                model, merged, exposure_ms, device, n_in, args.tile_size
            )
            key = f"dual_{name}"
            outputs[key] = dual
            method_meta[key] = (args.temporal_window, name)
            outputs[f"fe_{name}"] = merged
            method_meta[f"fe_{name}"] = (args.temporal_window, f"fe_{name}")

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
            "dual_baseline_spatial",
            "dual_ea_mild",
            "dual_ea_mild_gamma",
            "dual_ea_mid",
            "dual_ea_protect",
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
            if method.startswith("fe_"):
                continue
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
            "  DES "
            + " ".join(
                f"{k.replace('dual_', '')}={des_by[k]:.4f}"
                for k in des_by
                if k.startswith("dual_") or k == "vst_bm3d"
            ),
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("--- mean DES (dual_*) ---", flush=True)
    by_method: dict[str, list[float]] = {}
    for row in rows:
        if not row["method"].startswith("dual_") and row["method"] != "vst_bm3d":
            continue
        by_method.setdefault(row["method"], []).append(float(row["des"]))
    ranking = sorted(
        by_method.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True
    )
    for method, values in ranking:
        print(f"{method}: {sum(values) / len(values):.4f} (n={len(values)})", flush=True)

    best_path = args.output_dir / "best_config.txt"
    if ranking:
        best_name = ranking[0][0]
        best_des = sum(ranking[0][1]) / len(ranking[0][1])
        best_path.write_text(f"{best_name}\t{best_des:.6f}\n", encoding="utf-8")
        print(f"Best: {best_name} DES={best_des:.4f} -> {best_path}", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
