"""P5-b (edge-safe flat post) + P5-c (flat-mask BM3D) on deploy-faithful path."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_adaptive_vs_bm3d import vst_bm3d
from .compare_fusion_p5 import flat_mask_hybrid_bm3d
from .compare_sota_edgekd_blend import _spatial_on_merged
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import gated_spatial_from_temporal
from .infer import load_model
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost, flat_half_scale_boost
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi
from .wiener_merge import merge_from_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_p5bc"),
    )
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edge",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt"),
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--low-fps", type=float, default=1.0)
    parser.add_argument("--skip-bm3d", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sota, n_in, _ = load_model(args.checkpoint_sota, None, device)
    model_edge, n_e, _ = load_model(args.checkpoint_edge, None, device)
    assert n_in == n_e
    print(f"device={device} files={len(files)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for path in files:
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        target_index = frames.shape[0] // 2
        ys, xs = central_roi(height, width)
        input_dn = decode_mono10(frames[target_index])
        temporal_ref = aligned_temporal_trimmed_mean(
            frames,
            target_index,
            ys,
            xs,
            frames.shape[0],
            min(16, frames.shape[0]),
            trim_fraction=0.10,
            exclude_frame_index=target_index,
        )
        mask = reliable_reference_mask(input_dn, temporal_ref)
        print(f"{path.name} fps={fps}", flush=True)
        temporal, _ = merge_from_memmap(
            frames,
            target_index,
            input_frames=16,
            tile_size=32,
            overlap=16,
            c_factor=8.0,
            align=True,
            spatial_wiener=False,
        )
        fe_gated, params, fe_sigma = gated_spatial_from_temporal(
            temporal,
            fps=float(fps),
            n_frames_averaged=16,
            tile_size=32,
            overlap=16,
        )
        sota_dn = _spatial_on_merged(
            model_sota, fe_gated, exposure_ms, device, n_in, args.tile_size
        )
        if float(fps) <= float(args.low_fps):
            base = sota_dn
            guide = sota_dn
            route = "sota"
        else:
            edge_dn = _spatial_on_merged(
                model_edge, fe_gated, exposure_ms, device, n_in, args.tile_size
            )
            base, _ = blend_sota_edgekd(
                sota_dn, edge_dn, temperature=8.0, harden=16.0, edge_weight=1.0
            )
            guide = (0.5 * sota_dn + 0.5 * edge_dn).astype(np.float32)
            route = "blend"

        outputs: dict[str, np.ndarray] = {
            "input": input_dn,
            "temporal_reference": temporal_ref,
            "p4_deploy": base,
        }

        # P5-b edge-safer posts (low-fps only)
        if float(fps) <= float(args.low_fps):
            outputs["p5b_bilat_s0.35_h24"] = flat_bilateral_boost(
                base, guide=guide, flat_strength=0.35, harden=24.0
            )
            outputs["p5b_bilat_s0.5_h32"] = flat_bilateral_boost(
                base, guide=guide, flat_strength=0.5, harden=32.0
            )
            outputs["p5b_bilat_s0.65_h40"] = flat_bilateral_boost(
                base, guide=guide, flat_strength=0.65, harden=40.0
            )
            outputs["p5b_hs0.25_h40"] = flat_half_scale_boost(
                base, guide=guide, flat_strength=0.25, harden=40.0
            )
            outputs["p5b_hs0.4_h40"] = flat_half_scale_boost(
                base, guide=guide, flat_strength=0.4, harden=40.0
            )
        else:
            for k in (
                "p5b_bilat_s0.35_h24",
                "p5b_bilat_s0.5_h32",
                "p5b_bilat_s0.65_h40",
                "p5b_hs0.25_h40",
                "p5b_hs0.4_h40",
            ):
                outputs[k] = base

        # P5-c flat-mask BM3D hybrids
        if not args.skip_bm3d:
            bm3d_dn = vst_bm3d(input_dn, exposure_ms, tile_size=512)
            outputs["vst_bm3d"] = bm3d_dn
            outputs["p5c_flatbm3d"] = flat_mask_hybrid_bm3d(base, bm3d_dn, guide)
            # low-fps: BM3D flat on top of best-ish bilat
            if float(fps) <= float(args.low_fps):
                bilat = outputs["p5b_bilat_s0.5_h32"]
                outputs["p5c_bilat_then_flatbm3d"] = flat_mask_hybrid_bm3d(
                    bilat, bm3d_dn, guide
                )
            else:
                outputs["p5c_bilat_then_flatbm3d"] = outputs["p5c_flatbm3d"]

        input_sigma = float(
            metric_row(
                path, fps, target_index, "input", input_dn, temporal_ref, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
        des_by: dict[str, float] = {}
        for method, image in outputs.items():
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
                checkpoint=method,
                input_frames=16,
            )
            retention, edge_mae = edge_metrics_dn(image, temporal_ref)
            des, ng, ef = denoise_edge_score(
                float(row["roi_highpass_noise_sigma_dn"]),
                input_sigma,
                retention,
            )
            row["edge_retention"] = f"{retention:.6f}"
            row["edge_sobel_mae"] = f"{edge_mae:.6f}"
            row["noise_gain"] = f"{ng:.6f}"
            row["edge_fidelity"] = f"{ef:.6f}"
            row["des"] = f"{des:.6f}"
            row["route"] = route
            row["fe_sigma_dn"] = f"{fe_sigma:.6f}"
            row["gated_schedule"] = params.name
            rows.append(row)
            des_by[method] = des
        keys = [
            "p4_deploy",
            "p5b_bilat_s0.5_h32",
            "p5c_flatbm3d",
            "vst_bm3d",
        ]
        print(
            "  "
            + " ".join(f"{k.split('_')[-1]}={des_by.get(k, 0):.4f}" for k in keys if k in des_by),
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by: dict[str, list[float]] = defaultdict(list)
    det: dict[str, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for row in rows:
        m = row["method"]
        if m in ("input", "temporal_reference"):
            continue
        des = float(row["des"])
        ng = float(row["noise_gain"])
        ef = float(row["edge_fidelity"])
        by[m].append(des)
        if "74824541" in row["file"]:
            det[m]["f10"] = (des, ng, ef)
        if "53940814" in row["file"]:
            det[m]["f05"] = (des, ng, ef)
        if "53741354" in row["file"]:
            det[m]["f1"] = (des, ng, ef)

    ranked = []
    for m, vals in by.items():
        mean_des = sum(vals) / len(vals)
        f10 = det[m].get("f10", (float("nan"),) * 3)
        f05 = det[m].get("f05", (float("nan"),) * 3)
        f1 = det[m].get("f1", (float("nan"),) * 3)
        # Edge-safe: don't regress f10 ef; keep low-fps ef near p4
        ok = (
            f10[2] >= 0.965
            and f05[2] >= 0.870
            and f1[2] >= 0.960
        )
        ranked.append((m, mean_des, f10, f05, f1, ok))
    ranked.sort(key=lambda x: (x[5], x[1], x[2][2]), reverse=True)

    print("--- ranked (edge-safe first) ---", flush=True)
    for m, mean_des, f10, f05, f1, ok in ranked:
        print(
            f"{m}: mean={mean_des:.4f} f10_ef={f10[2]:.4f} "
            f"f05_ef={f05[2]:.4f} f05_ng={f05[1]:.4f} f1_ef={f1[2]:.4f} ok={ok}",
            flush=True,
        )

    safe = [x for x in ranked if x[5]]
    best = safe[0] if safe else ranked[0]
    summary = {
        "best_edge_safe": {
            "method": best[0],
            "mean_des": best[1],
            "f10_ef": best[2][2],
            "f05_ng": best[3][1],
            "f05_ef": best[3][2],
            "ok": best[5],
        },
        "p4_mean": sum(by["p4_deploy"]) / len(by["p4_deploy"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"BEST_SAFE {best[0]} mean={best[1]:.4f}", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
