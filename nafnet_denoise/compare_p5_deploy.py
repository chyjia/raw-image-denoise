"""Deploy-faithful P5-b check: fps>low → blend; fps<=low → SOTA (+ optional half-scale)."""

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
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_half_scale_boost
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi
from .wiener_merge import merge_from_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_p5_deploy"),
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
    model_edge, n_edge, _ = load_model(args.checkpoint_edge, None, device)
    assert n_in == n_edge
    print(f"device={device} files={len(files)}", flush=True)

    recipes = [
        ("p4_deploy", None),  # fps gate sota / blend h16
        ("p5_hs0.45", 0.45),
        ("p5_hs0.65", 0.65),
        ("p5_hs0.85", 0.85),
    ]
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
            route = "sota"
        else:
            edge_dn = _spatial_on_merged(
                model_edge, fe_gated, exposure_ms, device, n_in, args.tile_size
            )
            base, _ = blend_sota_edgekd(
                sota_dn, edge_dn, temperature=8.0, harden=16.0, edge_weight=1.0
            )
            route = "blend"

        outputs: dict[str, np.ndarray] = {"input": input_dn, "temporal_reference": temporal_ref}
        for name, strength in recipes:
            if strength is None or float(fps) > float(args.low_fps):
                outputs[name] = base
            else:
                outputs[name] = flat_half_scale_boost(
                    base, guide=base, flat_strength=float(strength)
                )
        if not args.skip_bm3d:
            outputs["vst_bm3d"] = vst_bm3d(input_dn, exposure_ms, tile_size=512)

        input_sigma = float(
            metric_row(
                path, fps, target_index, "input", input_dn, temporal_ref, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
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
        print(
            f"  route={route} "
            + " ".join(
                f"{n}={float([r for r in rows if r['file']==path.name and r['method']==n][-1]['des']):.4f}"
                for n, _ in recipes
            ),
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by: dict[str, list[float]] = defaultdict(list)
    f10: dict[str, tuple[float, float]] = {}
    f05: dict[str, tuple[float, float]] = {}
    for row in rows:
        m = row["method"]
        if m in ("input", "temporal_reference"):
            continue
        by[m].append(float(row["des"]))
        if "74824541" in row["file"]:
            f10[m] = (float(row["edge_fidelity"]), float(row["des"]))
        if "53940814" in row["file"]:
            f05[m] = (float(row["noise_gain"]), float(row["des"]))
    summary = {}
    for m, vals in by.items():
        summary[m] = {
            "mean_des": sum(vals) / len(vals),
            "f10_ef": f10.get(m, (float("nan"),))[0],
            "f10_des": f10.get(m, (float("nan"), float("nan")))[1],
            "f05_ng": f05.get(m, (float("nan"),))[0],
            "f05_des": f05.get(m, (float("nan"), float("nan")))[1],
        }
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean_des"], reverse=True)
    print("--- deploy-faithful ---", flush=True)
    for m, s in ranked:
        if m == "vst_bm3d":
            continue
        print(
            f"{m}: mean={s['mean_des']:.4f} f10_ef={s['f10_ef']:.4f} "
            f"f05_ng={s['f05_ng']:.4f}",
            flush=True,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"ranked": ranked, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
