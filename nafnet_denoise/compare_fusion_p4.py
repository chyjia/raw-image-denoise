"""P4: zero-train fusion ladder on frozen SOTA + lap_ms DualHeads.

Shared gated FE + one forward each. Then evaluate many fuse recipes:
  P4-a: temperature / harden / edge_power / edge_gain
  P4-b: residual-disagreement gates
  P4-c: low-fps flat-favoring variants

Picks best mean DES (tie-break: f10 ef) and writes summary.json.
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
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi
from .wiener_merge import merge_from_memmap


def harden_edge_map(edge: np.ndarray, harden: float) -> np.ndarray:
    if harden <= 0:
        return edge
    return 1.0 / (1.0 + np.exp(-float(harden) * (edge - 0.5)))


def fuse(
    sota: np.ndarray,
    edge: np.ndarray,
    guide: np.ndarray,
    temperature: float = 8.0,
    harden: float = 0.0,
    edge_power: float = 1.0,
    edge_gain: float = 1.0,
    residual_mix: float = 0.0,
    residual_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend with optional residual-disagreement modulation."""
    emap = soft_edge_mask_dn(guide, temperature=temperature)
    emap = harden_edge_map(emap, harden)
    if edge_power != 1.0:
        emap = np.power(np.clip(emap, 0.0, 1.0), float(edge_power))
    if edge_gain != 1.0:
        emap = np.clip(emap * float(edge_gain), 0.0, 1.0)
    if residual_mix > 0.0:
        diff = np.abs(sota - edge)
        scale = float(np.percentile(diff, 90.0)) + 1e-6
        disagree = np.clip(diff / scale, 0.0, 1.0)
        if residual_power != 1.0:
            disagree = np.power(disagree, float(residual_power))
        emap = (1.0 - residual_mix) * emap + residual_mix * np.maximum(emap, disagree)
        emap = np.clip(emap, 0.0, 1.0)
    out = emap * edge + (1.0 - emap) * sota
    return out.astype(np.float32), emap.astype(np.float32)


def build_recipes() -> list[dict]:
    recipes: list[dict] = []
    recipes.append(
        dict(name="base_T8", temperature=8.0, harden=0.0, edge_power=1.0, edge_gain=1.0)
    )
    for t in (4.0, 12.0, 16.0, 24.0):
        recipes.append(
            dict(name=f"T{t:g}", temperature=t, harden=0.0, edge_power=1.0, edge_gain=1.0)
        )
    for h in (4.0, 8.0, 12.0, 16.0):
        recipes.append(
            dict(name=f"T8_h{h:g}", temperature=8.0, harden=h, edge_power=1.0, edge_gain=1.0)
        )
    for p in (0.5, 0.7, 1.3, 1.6):
        recipes.append(
            dict(name=f"T8_p{p:g}", temperature=8.0, harden=0.0, edge_power=p, edge_gain=1.0)
        )
    for g in (1.15, 1.3, 1.5):
        recipes.append(
            dict(name=f"T8_g{g:g}", temperature=8.0, harden=0.0, edge_power=1.0, edge_gain=g)
        )
    recipes.append(
        dict(name="T12_h8", temperature=12.0, harden=8.0, edge_power=1.0, edge_gain=1.0)
    )
    recipes.append(
        dict(name="T16_h8_g1.2", temperature=16.0, harden=8.0, edge_power=1.0, edge_gain=1.2)
    )
    recipes.append(
        dict(name="T12_p0.7_h4", temperature=12.0, harden=4.0, edge_power=0.7, edge_gain=1.0)
    )
    for mix in (0.35, 0.5, 0.65):
        recipes.append(
            dict(
                name=f"res_m{mix:g}",
                temperature=8.0,
                harden=0.0,
                edge_power=1.0,
                edge_gain=1.0,
                residual_mix=mix,
                residual_power=1.0,
            )
        )
    recipes.append(
        dict(
            name="res_m0.5_T12_h4",
            temperature=12.0,
            harden=4.0,
            edge_power=1.0,
            edge_gain=1.0,
            residual_mix=0.5,
            residual_power=0.8,
        )
    )
    recipes.append(
        dict(
            name="res_m0.65_T16_h8",
            temperature=16.0,
            harden=8.0,
            edge_power=1.0,
            edge_gain=1.15,
            residual_mix=0.65,
            residual_power=1.0,
        )
    )
    seen: set[str] = set()
    uniq: list[dict] = []
    for recipe in recipes:
        if recipe["name"] in seen:
            continue
        seen.add(recipe["name"])
        uniq.append(recipe)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_fusion_p4"),
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
    parser.add_argument("--skip-bm3d", action="store_true")
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

    recipes = build_recipes()
    p4c_bases = ["base_T8", "T12_h8", "res_m0.5_T12_h4"]
    print(f"recipes={len(recipes)} (+P4-c on {p4c_bases})", flush=True)

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
        for recipe in recipes:
            name = recipe["name"]
            out, _ = fuse(
                sota_dn,
                edge_dn,
                guide,
                temperature=float(recipe.get("temperature", 8.0)),
                harden=float(recipe.get("harden", 0.0)),
                edge_power=float(recipe.get("edge_power", 1.0)),
                edge_gain=float(recipe.get("edge_gain", 1.0)),
                residual_mix=float(recipe.get("residual_mix", 0.0)),
                residual_power=float(recipe.get("residual_power", 1.0)),
            )
            outputs[f"fuse_{name}"] = out
            if name in p4c_bases and float(fps) <= 1.0:
                out_c, _ = fuse(
                    sota_dn,
                    edge_dn,
                    guide,
                    temperature=float(recipe.get("temperature", 8.0)),
                    harden=max(float(recipe.get("harden", 0.0)), 8.0),
                    edge_power=max(float(recipe.get("edge_power", 1.0)), 1.4),
                    edge_gain=float(recipe.get("edge_gain", 1.0)),
                    residual_mix=0.0,
                    residual_power=1.0,
                )
                outputs[f"fuse_{name}_f05flat"] = out_c
            elif name in p4c_bases:
                outputs[f"fuse_{name}_f05flat"] = out

        if not args.skip_bm3d:
            outputs["vst_bm3d"] = vst_bm3d(
                input_dn, exposure_ms, tile_size=args.bm3d_tile_size
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
            "fuse_base_T8",
            "fuse_T12_h8",
            "fuse_res_m0.5_T12_h4",
            "fuse_res_m0.65_T16_h8",
            "vst_bm3d",
        ]
        msg = " ".join(f"{k.split('fuse_')[-1]}={des_by.get(k, 0):.4f}" for k in keys)
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
        if method in ("input", "temporal_reference"):
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
            if method.startswith("fuse_")
        ),
        key=lambda item: (item[1], item[2] if item[2] == item[2] else -1.0),
        reverse=True,
    )
    print("--- top fuse mean DES ---", flush=True)
    for method, des, ef, ng in ranked[:12]:
        print(f"{method}: DES={des:.4f} f10_ef={ef:.4f} f05_ng={ng:.4f}", flush=True)

    best_name, best_des, best_ef, best_ng = ranked[0]
    base_name = best_name.replace("fuse_", "").replace("_f05flat", "")
    recipe = next((item for item in recipes if item["name"] == base_name), recipes[0])
    summary = {
        "best_method": best_name,
        "best_mean_des": best_des,
        "best_f10_ef": best_ef,
        "best_f05_ng": best_ng,
        "baseline_mean_des": sum(by["fuse_base_T8"]) / len(by["fuse_base_T8"]),
        "recipe": recipe,
        "p4c_low_fps_flat": best_name.endswith("_f05flat"),
        "ref_blend_lap_ms": 0.9059,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"BEST {best_name} mean={best_des:.4f} "
        f"(base={summary['baseline_mean_des']:.4f})",
        flush=True,
    )
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
