"""P5 zero-train fusion ladder on frozen SOTA + lap_ms DualHeads.

Shared gated FE + one forward each, then:
  P5-a: Laplacian multi-band fusion schedules
  P5-b: low-fps flat half-scale boost variants
  P5-c: flat-mask hybrid BM3D (optional, --with-bm3d-hybrid)

Baseline: pixel soft blend T=8 harden=16 (P4-a deploy).
"""

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
from .compare_sota_edgekd_blend import _spatial_on_merged
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import gated_spatial_from_temporal
from .infer import load_model
from .infer_ensemble import soft_edge_mask_dn
from .multiband_fuse import (
    flat_half_scale_boost,
    multiband_fuse,
    pixel_blend_baseline,
)
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi
from .wiener_merge import merge_from_memmap


def build_p5a_recipes() -> list[dict]:
    """Band weights: index0=highest freq … last=LF residual (prefer SOTA)."""
    recipes: list[dict] = [
        dict(name="pix_T8_h16", kind="pixel", temperature=8.0, harden=16.0),
        # Pure band schedules (no spatial)
        dict(name="mb3_1_0.5_0", kind="mb", weights=[1.0, 0.5, 0.0], spatial_mix=0.0),
        dict(name="mb3_1_0.7_0", kind="mb", weights=[1.0, 0.7, 0.0], spatial_mix=0.0),
        dict(name="mb3_1_0.85_0.15", kind="mb", weights=[1.0, 0.85, 0.15], spatial_mix=0.0),
        dict(name="mb3_0.9_0.5_0", kind="mb", weights=[0.9, 0.5, 0.0], spatial_mix=0.0),
        dict(name="mb4_1_0.8_0.3_0", kind="mb", weights=[1.0, 0.8, 0.3, 0.0], spatial_mix=0.0),
        dict(name="mb4_1_1_0.4_0", kind="mb", weights=[1.0, 1.0, 0.4, 0.0], spatial_mix=0.0),
        dict(name="mb4_1_0.7_0.2_0", kind="mb", weights=[1.0, 0.7, 0.2, 0.0], spatial_mix=0.0),
        dict(name="mb4_1_0.9_0.5_0.1", kind="mb", weights=[1.0, 0.9, 0.5, 0.1], spatial_mix=0.0),
        # Spatial-modulated multi-band
        dict(name="mb3_1_0.7_0_sp0.5", kind="mb", weights=[1.0, 0.7, 0.0], spatial_mix=0.5),
        dict(name="mb3_1_0.7_0_sp1", kind="mb", weights=[1.0, 0.7, 0.0], spatial_mix=1.0),
        dict(name="mb4_1_0.8_0.3_0_sp0.5", kind="mb", weights=[1.0, 0.8, 0.3, 0.0], spatial_mix=0.5),
        dict(name="mb4_1_1_0.4_0_sp0.35", kind="mb", weights=[1.0, 1.0, 0.4, 0.0], spatial_mix=0.35),
        # Hybrid: pixel blend then replace HF from edge via mb on (blend, edge)
        dict(
            name="pix_then_mb3_hf",
            kind="pix_mb",
            weights=[1.0, 0.6, 0.0],
            spatial_mix=0.0,
        ),
    ]
    return recipes


def apply_recipe(
    recipe: dict,
    sota: np.ndarray,
    edge: np.ndarray,
    guide: np.ndarray,
) -> np.ndarray:
    kind = recipe["kind"]
    if kind == "pixel":
        return pixel_blend_baseline(
            sota,
            edge,
            temperature=float(recipe.get("temperature", 8.0)),
            harden=float(recipe.get("harden", 16.0)),
        )
    if kind == "mb":
        return multiband_fuse(
            sota,
            edge,
            band_edge_weights=list(recipe["weights"]),
            guide=guide,
            spatial_mix=float(recipe.get("spatial_mix", 0.0)),
            temperature=8.0,
            harden=16.0,
        )
    if kind == "pix_mb":
        base = pixel_blend_baseline(sota, edge, temperature=8.0, harden=16.0)
        # Keep LF from pixel blend, pull HF from edge arm
        return multiband_fuse(
            base,
            edge,
            band_edge_weights=list(recipe["weights"]),
            guide=guide,
            spatial_mix=float(recipe.get("spatial_mix", 0.0)),
        )
    raise ValueError(f"unknown kind {kind}")


def flat_mask_hybrid_bm3d(
    neural: np.ndarray,
    bm3d_dn: np.ndarray,
    guide: np.ndarray,
    temperature: float = 8.0,
    harden: float = 16.0,
) -> np.ndarray:
    emap = soft_edge_mask_dn(guide, temperature=temperature)
    if harden > 0.0:
        emap = 1.0 / (1.0 + np.exp(-float(harden) * (emap - 0.5)))
    flat = 1.0 - emap
    out = emap * neural + flat * bm3d_dn
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_fusion_p5"),
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
    parser.add_argument("--bm3d-tile-size", type=int, default=512)
    parser.add_argument(
        "--with-bm3d-hybrid",
        action="store_true",
        help="Also evaluate P5-c flat-mask BM3D hybrids (slow).",
    )
    parser.add_argument("--skip-bm3d-ref", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    for ckpt in (args.checkpoint_sota, args.checkpoint_edge):
        if not ckpt.exists():
            raise SystemExit(f"Missing {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)}", flush=True)
    model_sota, n_sota, _ = load_model(args.checkpoint_sota, None, device)
    model_edge, n_edge, _ = load_model(args.checkpoint_edge, None, device)
    if n_sota != n_edge:
        raise SystemExit(f"Frame mismatch {n_sota}/{n_edge}")
    n_in = n_sota

    recipes = build_p5a_recipes()
    print(f"P5-a recipes={len(recipes)}", flush=True)

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
        print(f"{path.name}: FE + forwards...", flush=True)
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
        edge_dn = _spatial_on_merged(
            model_edge, fe_gated, exposure_ms, device, n_in, args.tile_size
        )
        guide = (0.5 * sota_dn + 0.5 * edge_dn).astype(np.float32)

        outputs: dict[str, np.ndarray] = {
            "input": input_dn,
            "temporal_reference": temporal_ref,
            "sota_gated": sota_dn,
            "edge_gated": edge_dn,
        }

        # P5-a
        for recipe in recipes:
            name = recipe["name"]
            out = apply_recipe(recipe, sota_dn, edge_dn, guide)
            outputs[f"p5a_{name}"] = out

            # P5-b: always-on half-scale flat boost variants on key recipes
            if name in ("pix_T8_h16", "mb3_1_0.7_0", "mb4_1_0.8_0.3_0", "mb4_1_1_0.4_0"):
                for strength in (0.45, 0.65, 0.85):
                    boosted = flat_half_scale_boost(
                        out, guide=guide, flat_strength=strength
                    )
                    outputs[f"p5b_{name}_hs{strength:g}"] = boosted
                # fps-gated: only apply when fps <= 1
                if float(fps) <= 1.0:
                    outputs[f"p5b_{name}_hs0.65_f05"] = flat_half_scale_boost(
                        out, guide=guide, flat_strength=0.65
                    )
                else:
                    outputs[f"p5b_{name}_hs0.65_f05"] = out

        bm3d_dn = None
        need_bm3d = args.with_bm3d_hybrid or not args.skip_bm3d_ref
        if need_bm3d:
            bm3d_dn = vst_bm3d(input_dn, exposure_ms, tile_size=args.bm3d_tile_size)
            if not args.skip_bm3d_ref:
                outputs["vst_bm3d"] = bm3d_dn
            if args.with_bm3d_hybrid:
                for base_key in ("p5a_pix_T8_h16", "p5a_mb4_1_0.8_0.3_0"):
                    if base_key not in outputs:
                        continue
                    outputs[f"p5c_{base_key}_flatbm3d"] = flat_mask_hybrid_bm3d(
                        outputs[base_key], bm3d_dn, guide
                    )

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
            row["fe_sigma_dn"] = f"{fe_sigma:.6f}"
            row["gated_schedule"] = params.name
            rows.append(row)
            des_by[method] = des

        keys = [
            "p5a_pix_T8_h16",
            "p5a_mb3_1_0.7_0",
            "p5a_mb4_1_0.8_0.3_0",
            "p5a_mb4_1_1_0.4_0",
            "p5b_pix_T8_h16_hs0.65_f05",
        ]
        msg = " ".join(
            f"{k.replace('p5a_', '').replace('p5b_', '')}={des_by.get(k, 0):.4f}"
            for k in keys
        )
        print(f"  {msg}", flush=True)

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by: dict[str, list[float]] = defaultdict(list)
    f10_ef: dict[str, float] = {}
    f05_ng: dict[str, float] = {}
    for row in rows:
        method = row["method"]
        if method in ("input", "temporal_reference", "sota_gated", "edge_gated"):
            continue
        by[method].append(float(row["des"]))
        if "74824541" in row["file"]:
            f10_ef[method] = float(row["edge_fidelity"])
        if "53940814" in row["file"]:
            f05_ng[method] = float(row["noise_gain"])

    ranked = sorted(
        (
            (
                method,
                sum(vals) / len(vals),
                f10_ef.get(method, float("nan")),
                f05_ng.get(method, float("nan")),
            )
            for method, vals in by.items()
        ),
        key=lambda item: (
            item[1],
            item[2] if item[2] == item[2] else -1.0,
        ),
        reverse=True,
    )
    print("--- top mean DES ---", flush=True)
    for method, des, ef, ng in ranked[:15]:
        print(f"{method}: DES={des:.4f} f10_ef={ef:.4f} f05_ng={ng:.4f}", flush=True)

    best_name, best_des, best_ef, best_ng = ranked[0]
    base_des = sum(by["p5a_pix_T8_h16"]) / len(by["p5a_pix_T8_h16"])
    p5a_only = [item for item in ranked if item[0].startswith("p5a_")]
    p5b_only = [item for item in ranked if item[0].startswith("p5b_")]
    p5c_only = [item for item in ranked if item[0].startswith("p5c_")]
    summary = {
        "best_method": best_name,
        "best_mean_des": best_des,
        "best_f10_ef": best_ef,
        "best_f05_ng": best_ng,
        "baseline_pix_T8_h16": base_des,
        "best_p5a": {
            "method": p5a_only[0][0],
            "mean_des": p5a_only[0][1],
            "f10_ef": p5a_only[0][2],
            "f05_ng": p5a_only[0][3],
        },
        "best_p5b": (
            {
                "method": p5b_only[0][0],
                "mean_des": p5b_only[0][1],
                "f10_ef": p5b_only[0][2],
                "f05_ng": p5b_only[0][3],
            }
            if p5b_only
            else None
        ),
        "best_p5c": (
            {
                "method": p5c_only[0][0],
                "mean_des": p5c_only[0][1],
                "f10_ef": p5c_only[0][2],
                "f05_ng": p5c_only[0][3],
            }
            if p5c_only
            else None
        ),
        "ref_p4": 0.9060,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"BEST {best_name} mean={best_des:.4f} (pix_base={base_des:.4f})",
        flush=True,
    )
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
