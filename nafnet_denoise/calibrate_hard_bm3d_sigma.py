"""Per-clip Wiener-BM3D sigma_psd grid for hard scenes (f0.5 / f10).

For each hard clip, sample Wiener-merged patches and pick sigma that maximizes
a DES-like score of the BM3D teacher vs the leave-burst temporal target, with
an edge-retention floor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .alignment import soft_gate_burst
from .common import (
    DEFAULT_DARK_VARIANCE_PER_S,
    RAW_MAX,
    center_frame_index,
    decode_mono10,
    exposure_intercept,
    vst_forward,
    vst_inverse,
)
from .postmerge_noise import flat_mask, highpass_mad_sigma_masked
from .train_distill import denoise_edge_score, edge_metrics_dn
from .wiener_merge import merge_burst_dn

DEFAULT_TOKENS = ("20260719153940814", "20260725174824541")


def _aligned_crop(
    raw_frame: np.ndarray,
    x0: int,
    y0: int,
    size: int,
    tx: float,
    ty: float,
) -> np.ndarray:
    import cv2

    source_center_x = x0 + (size - 1) * 0.5 - tx
    source_center_y = y0 + (size - 1) * 0.5 - ty
    source_x0 = int(math.floor(source_center_x - (size - 1) * 0.5)) - 2
    source_y0 = int(math.floor(source_center_y - (size - 1) * 0.5)) - 2
    region_size = size + 4
    region = raw_frame[
        source_y0 : source_y0 + region_size,
        source_x0 : source_x0 + region_size,
    ]
    if np.issubdtype(region.dtype, np.integer):
        region = decode_mono10(region)
    else:
        region = region.astype(np.float32, copy=False)
    local_center = (
        source_center_x - source_x0,
        source_center_y - source_y0,
    )
    return cv2.getRectSubPix(region, (size, size), local_center)


def _find_entry(entries: list[dict], token: str) -> dict:
    for entry in entries:
        name = str(entry.get("name") or Path(entry["path"]).name)
        path = str(entry.get("path", ""))
        if token in name or token in path:
            return entry
    raise SystemExit(f"Token not found in manifest: {token}")


def _sample_wiener_and_target(
    entry: dict,
    rng: np.random.Generator,
    gather: int,
    patch_size: int,
    soft_gate: bool,
    align_threshold: float,
    align_temperature: float,
    wiener_tile: int,
    wiener_overlap: int,
    wiener_c_factor: float,
    spatial: bool,
    spatial_c: float,
    target_frames: int = 16,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    path = Path(entry["path"])
    height = int(entry["height"])
    width = int(entry["width"])
    frame_count = int(entry["frame_count"])
    exposure_ms = float(entry["exposure_ms"])
    bytes_per_frame = width * height * 2
    n_disk = path.stat().st_size // bytes_per_frame
    frames = np.memmap(path, mode="r", dtype="<u2", shape=(n_disk, height, width))
    shifts = np.load(entry["shifts"], mmap_mode="r")
    offsets = np.load(entry["offsets"], mmap_mode="r")
    margin = int(np.ceil(np.abs(shifts).max())) + 4
    ref = center_frame_index(gather)
    min_center = ref
    max_center = frame_count - (gather - ref)
    if max_center < min_center:
        return None
    center = int(rng.integers(min_center, max_center + 1))
    indices = [center - ref + offset for offset in range(gather)]
    y0 = int(rng.integers(margin, height - patch_size - margin + 1))
    x0 = int(rng.integers(margin, width - patch_size - margin + 1))
    noisy_frames = []
    for frame_index in indices:
        tx = float(shifts[frame_index, 0] - shifts[center, 0])
        ty = float(shifts[frame_index, 1] - shifts[center, 1])
        brightness_offset = float(offsets[frame_index] - offsets[center])
        patch = _aligned_crop(frames[frame_index], x0, y0, patch_size, tx, ty)
        patch = np.clip(patch - brightness_offset, 0.0, RAW_MAX).astype(np.float32)
        noisy_frames.append(patch)
    if soft_gate:
        noisy_frames, _ = soft_gate_burst(
            noisy_frames,
            ref,
            threshold=align_threshold,
            temperature=align_temperature,
        )
    merged, _ = merge_burst_dn(
        noisy_frames,
        reference_index=ref,
        align=False,
        tile_size=wiener_tile,
        overlap=wiener_overlap,
        c_factor=wiener_c_factor,
        spatial_wiener=spatial,
        spatial_c_factor=spatial_c,
    )
    input_set = set(indices)
    target_indices = sorted(
        (index for index in range(frame_count) if index not in input_set),
        key=lambda index: abs(index - center),
    )[:target_frames]
    if len(target_indices) < 4:
        return None
    target_frames_dn = []
    for frame_index in target_indices:
        tx = float(shifts[frame_index, 0] - shifts[center, 0])
        ty = float(shifts[frame_index, 1] - shifts[center, 1])
        brightness_offset = float(offsets[frame_index] - offsets[center])
        patch = _aligned_crop(frames[frame_index], x0, y0, patch_size, tx, ty)
        patch = np.clip(patch - brightness_offset, 0.0, RAW_MAX).astype(np.float32)
        target_frames_dn.append(patch)
    stack = np.stack(target_frames_dn, axis=0)
    stack.sort(axis=0)
    trim = max(1, int(len(target_frames_dn) * 0.10))
    target = stack[trim:-trim].mean(axis=0).astype(np.float32)
    return merged, target, exposure_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json"),
    )
    parser.add_argument("--tokens", type=str, default=",".join(DEFAULT_TOKENS))
    parser.add_argument("--patches-per-clip", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--merge-frames", type=int, default=16)
    parser.add_argument("--sigma-grid", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--edge-ret-min", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wiener-tile", type=int, default=32)
    parser.add_argument("--wiener-overlap", type=int, default=16)
    parser.add_argument("--wiener-c-factor", type=float, default=8.0)
    parser.add_argument("--wiener-spatial-c-factor", type=float, default=1.0)
    args = parser.parse_args()

    try:
        import bm3d  # noqa: F401
    except ImportError as error:
        raise SystemExit("pip install bm3d") from error

    import bm3d

    tokens = [part.strip() for part in args.tokens.split(",") if part.strip()]
    sigma_grid = [float(part) for part in args.sigma_grid.split(",") if part.strip()]
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    gather = int(args.merge_frames)
    chosen: dict[str, float] = {}
    report: dict[str, list[dict]] = {}

    for token in tokens:
        entry = _find_entry(entries, token)
        print(f"=== {token} fps={entry.get('fps')} ===", flush=True)
        samples = []
        for _ in range(int(args.patches_per_clip)):
            sample = _sample_wiener_and_target(
                entry,
                rng,
                gather=gather,
                patch_size=args.patch_size,
                soft_gate=True,
                align_threshold=0.03,
                align_temperature=0.015,
                wiener_tile=args.wiener_tile,
                wiener_overlap=args.wiener_overlap,
                wiener_c_factor=args.wiener_c_factor,
                spatial=True,
                spatial_c=args.wiener_spatial_c_factor,
            )
            if sample is not None:
                samples.append(sample)
        if not samples:
            raise SystemExit(f"No patches for {token}")

        rows = []
        for sigma in sigma_grid:
            des_scores = []
            flat_sigmas = []
            edge_rets = []
            for merged, target, exposure_ms in samples:
                intercept = exposure_intercept(
                    exposure_ms,
                    dark_variance_per_s=DEFAULT_DARK_VARIANCE_PER_S,
                )
                transformed = vst_forward(merged, intercept=intercept)
                denoised = bm3d.bm3d(
                    transformed.astype(np.float64),
                    sigma_psd=float(sigma),
                ).astype(np.float32)
                teacher = np.clip(
                    vst_inverse(denoised, intercept=intercept),
                    0.0,
                    RAW_MAX,
                )
                mask = flat_mask(merged)
                sigma_in = highpass_mad_sigma_masked(merged, mask)
                sigma_out = highpass_mad_sigma_masked(teacher, mask)
                retention, _ = edge_metrics_dn(teacher, target)
                if not np.isfinite(retention):
                    continue
                des, _, _ = denoise_edge_score(sigma_out, sigma_in, retention)
                des_scores.append(des)
                flat_sigmas.append(sigma_out)
                edge_rets.append(retention)
            if not des_scores:
                continue
            mean_des = float(np.mean(des_scores))
            mean_sigma = float(np.mean(flat_sigmas))
            mean_ret = float(np.mean(edge_rets))
            eligible = mean_ret >= args.edge_ret_min
            rows.append(
                {
                    "sigma": sigma,
                    "des": mean_des,
                    "flat_sigma": mean_sigma,
                    "edge_ret": mean_ret,
                    "eligible": eligible,
                }
            )
            print(
                f"  σ={sigma:.2f} DES={mean_des:.4f} flatσ={mean_sigma:.4f} "
                f"edge_ret={mean_ret:.4f} eligible={eligible}",
                flush=True,
            )

        eligible_rows = [row for row in rows if row["eligible"]]
        pool = eligible_rows if eligible_rows else rows
        best = max(pool, key=lambda row: (row["des"], -row["flat_sigma"]))
        chosen[token] = float(best["sigma"])
        report[token] = rows
        print(f"  -> chosen σ={best['sigma']:.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    args.out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}: {chosen}", flush=True)


if __name__ == "__main__":
    main()
