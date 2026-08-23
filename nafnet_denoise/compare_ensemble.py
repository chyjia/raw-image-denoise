"""Compare Adaptive / Dual-head / Ensemble vs VST+BM3D on DES."""

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
from .infer_ensemble import blend_adaptive_dual
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_ensemble_adaptive_dual"),
    )
    parser.add_argument(
        "--checkpoint-baseline-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-dual",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_dual_head_flat_sigma/best.pt"),
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
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--flat-bias", type=float, default=0.0)
    parser.add_argument(
        "--guide",
        choices=("dual", "adaptive", "input", "mean"),
        default="dual",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW under {args.input_dir}")
    if not args.checkpoint_baseline_4f.exists():
        raise SystemExit(f"Missing {args.checkpoint_baseline_4f}")
    if not args.checkpoint_dual.exists():
        raise SystemExit(f"Missing {args.checkpoint_dual}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} files={len(files)} "
        f"guide={args.guide} flat_bias={args.flat_bias}",
        flush=True,
    )

    adaptive_models = load_adaptive_models(
        args.checkpoint_1f,
        args.checkpoint_baseline_4f,
        args.checkpoint_16f,
        device,
    )
    dual_model, dual_frames, _ = load_model(args.checkpoint_dual, None, device)

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

        print(f"{path.name}: dual_head...", flush=True)
        dual_dn = denoise_frame(
            dual_model,
            frames,
            target_index,
            exposure_ms,
            device,
            input_frames=dual_frames,
            tile=args.tile_size,
        )
        outputs["dual_head_4f"] = dual_dn

        print(f"{path.name}: ensemble...", flush=True)
        if args.guide == "adaptive":
            guide_dn = adaptive_dn
        elif args.guide == "input":
            guide_dn = input_dn
        elif args.guide == "mean":
            guide_dn = 0.5 * (adaptive_dn + dual_dn)
        else:
            guide_dn = dual_dn
        ensemble_dn, _edge = blend_adaptive_dual(
            adaptive_dn,
            dual_dn,
            guide_dn,
            temperature=args.temperature,
            flat_bias=args.flat_bias,
        )
        outputs["ensemble"] = ensemble_dn
        ensemble_meta = {
            "route": f"ensemble({adaptive_meta.get('route', 'adaptive')}+dual)",
            "used_frames": int(adaptive_meta.get("used_frames", dual_frames)),
        }

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
            "dual_head_4f": (dual_frames, "dual_head_4f"),
            "ensemble": (
                int(ensemble_meta["used_frames"]),
                str(ensemble_meta["route"]),
            ),
            "vst_bm3d": (1, "ptc_vst"),
        }
        des_by_method: dict[str, float] = {}
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
            des_by_method[method] = des

        print(
            f"  DES adaptive={des_by_method.get('adaptive', 0):.4f} "
            f"dual={des_by_method.get('dual_head_4f', 0):.4f} "
            f"ensemble={des_by_method.get('ensemble', 0):.4f} "
            f"bm3d={des_by_method.get('vst_bm3d', 0):.4f}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {metrics_path}", flush=True)

    # Summary
    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(float(row["des"]))
    for method, values in sorted(by_method.items()):
        if method in ("input", "temporal_reference"):
            continue
        print(f"mean DES {method}: {sum(values) / len(values):.4f} (n={len(values)})")


if __name__ == "__main__":
    main()
