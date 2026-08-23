"""Calibrate post-Wiener flat residual σ on real Mono10 bursts.

Samples the same soft-gate + temporal/spatial Wiener front-end used at train/infer,
measures DES-compatible highpass MAD on flat regions, and fits:
    σ_res ≈ alpha * (σ_ptc / √N) + beta
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .alignment import soft_gate_burst
from .common import (
    DEFAULT_BLACK_LEVEL_DN,
    DEFAULT_DARK_VARIANCE_PER_S,
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    center_frame_index,
    decode_mono10,
    signal_variance_dn2,
)
from .postmerge_noise import (
    PostMergeNoiseCalib,
    fit_affine_sigma,
    flat_mask,
    highpass_mad_sigma_masked,
    save_postmerge_calib,
)
from .wiener_merge import merge_burst_dn


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("nafnet_denoise/cache/postmerge_noise.json"),
    )
    parser.add_argument("--patches-per-clip", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--merge-frames", type=int, default=16)
    parser.add_argument("--wiener-tile", type=int, default=32)
    parser.add_argument("--wiener-overlap", type=int, default=16)
    parser.add_argument("--wiener-c-factor", type=float, default=8.0)
    parser.add_argument("--wiener-spatial", action="store_true", default=True)
    parser.add_argument("--no-wiener-spatial", action="store_true")
    parser.add_argument("--wiener-spatial-c-factor", type=float, default=1.0)
    parser.add_argument("--align-soft-gate", action="store_true", default=True)
    parser.add_argument("--no-align-soft-gate", action="store_true")
    parser.add_argument("--align-threshold", type=float, default=0.03)
    parser.add_argument("--align-temperature", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--scene-substr",
        type=str,
        default="",
        help="Comma-separated name/path tokens; keep only matching clips.",
    )
    args = parser.parse_args()

    spatial = bool(args.wiener_spatial) and not bool(args.no_wiener_spatial)
    soft_gate = bool(args.align_soft_gate) and not bool(args.no_align_soft_gate)
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokens = tuple(
        part.strip() for part in str(args.scene_substr).split(",") if part.strip()
    )
    if tokens:
        filtered = []
        for entry in entries:
            name = str(entry.get("name") or Path(entry["path"]).name)
            path = str(entry.get("path", ""))
            if any(token in name or token in path for token in tokens):
                filtered.append(entry)
        entries = filtered
        print(f"scene filter {tokens}: {len(entries)} clips", flush=True)
    if args.max_clips > 0:
        entries = entries[: args.max_clips]
    if not entries:
        raise SystemExit(f"Empty manifest: {args.manifest}")

    rng = np.random.default_rng(args.seed)
    gather = int(args.merge_frames)
    ref = center_frame_index(gather)
    features: list[float] = []
    measured: list[float] = []
    ratios: list[float] = []
    rows: list[dict] = []

    for entry_index, entry in enumerate(entries):
        path = Path(entry["path"])
        height = int(entry["height"])
        width = int(entry["width"])
        frame_count = int(entry["frame_count"])
        exposure_ms = float(entry["exposure_ms"])
        bytes_per_frame = width * height * 2
        n_disk = path.stat().st_size // bytes_per_frame
        frames = np.memmap(
            path,
            mode="r",
            dtype="<u2",
            shape=(n_disk, height, width),
        )
        shifts = np.load(entry["shifts"], mmap_mode="r")
        offsets = np.load(entry["offsets"], mmap_mode="r")
        margin = int(np.ceil(np.abs(shifts).max())) + 4
        size = int(args.patch_size)
        min_center = ref
        max_center = frame_count - (gather - ref)
        if max_center < min_center:
            print(f"skip {path.name}: not enough frames", flush=True)
            continue

        clip_ok = 0
        for _ in range(int(args.patches_per_clip)):
            center = int(rng.integers(min_center, max_center + 1))
            indices = [center - ref + offset for offset in range(gather)]
            y0 = int(rng.integers(margin, height - size - margin + 1))
            x0 = int(rng.integers(margin, width - size - margin + 1))
            noisy_frames = []
            for frame_index in indices:
                tx = float(shifts[frame_index, 0] - shifts[center, 0])
                ty = float(shifts[frame_index, 1] - shifts[center, 1])
                brightness_offset = float(offsets[frame_index] - offsets[center])
                patch = _aligned_crop(frames[frame_index], x0, y0, size, tx, ty)
                patch = np.clip(patch - brightness_offset, 0.0, RAW_MAX).astype(
                    np.float32
                )
                noisy_frames.append(patch)
            if soft_gate:
                noisy_frames, _ = soft_gate_burst(
                    noisy_frames,
                    ref,
                    threshold=args.align_threshold,
                    temperature=args.align_temperature,
                )
            merged, _meta = merge_burst_dn(
                noisy_frames,
                reference_index=ref,
                align=False,
                tile_size=args.wiener_tile,
                overlap=args.wiener_overlap,
                c_factor=args.wiener_c_factor,
                spatial_wiener=spatial,
                spatial_c_factor=args.wiener_spatial_c_factor,
            )
            mask = flat_mask(merged)
            if int(mask.sum()) < 256:
                continue
            sigma_hp = highpass_mad_sigma_masked(merged, mask)
            mean_dn = float(merged[mask].mean())
            ptc = float(
                np.sqrt(
                    signal_variance_dn2(
                        np.asarray(mean_dn, dtype=np.float32),
                        exposure_ms,
                        PTC_SLOPE,
                        PTC_INTERCEPT,
                        DEFAULT_BLACK_LEVEL_DN,
                        DEFAULT_DARK_VARIANCE_PER_S,
                    )
                )
            )
            feature = ptc / math.sqrt(gather)
            if feature < 1e-6 or sigma_hp < 1e-6:
                continue
            features.append(feature)
            measured.append(sigma_hp)
            ratios.append(sigma_hp / feature)
            rows.append(
                {
                    "clip": path.name,
                    "fps": float(entry.get("fps", 0.0)),
                    "exposure_ms": exposure_ms,
                    "mean_dn": mean_dn,
                    "sigma_hp": sigma_hp,
                    "ptc_sigma": ptc,
                    "feature_ptc_over_sqrt_n": feature,
                    "ratio": sigma_hp / feature,
                }
            )
            clip_ok += 1
        print(
            f"[{entry_index + 1}/{len(entries)}] {path.name}: {clip_ok} patches",
            flush=True,
        )

    if len(features) < 8:
        raise SystemExit(f"Too few calibration samples: {len(features)}")

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    alpha, beta, rmse = fit_affine_sigma(x, y)
    # Keep alpha positive; if fit collapses, fall back to median ratio.
    if alpha <= 0.05:
        alpha = float(np.median(ratios))
        beta = 0.0
        pred = alpha * x
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

    median_ratio = float(np.median(ratios))
    if median_ratio > 5.0 or alpha > 5.0 or alpha < 0.05:
        raise SystemExit(
            f"Implausible post-merge fit: alpha={alpha:.4f} beta={beta:.4f} "
            f"median_ratio={median_ratio:.4f}. Check flat highpass measurement."
        )

    calib = PostMergeNoiseCalib(
        version=1,
        alpha=float(alpha),
        beta_dn=float(beta),
        sigma_floor_dn=0.05,
        divide_sqrt_n=True,
        n_merge=gather,
        spatial_wiener=spatial,
        wiener_tile=int(args.wiener_tile),
        wiener_overlap=int(args.wiener_overlap),
        wiener_c_factor=float(args.wiener_c_factor),
        wiener_spatial_c_factor=float(args.wiener_spatial_c_factor),
        n_samples=int(len(features)),
        rmse_dn=float(rmse),
        median_ratio=median_ratio,
        notes=(
            "σ_res = alpha*(σ_ptc/√N)+beta from Wiener-merged flat highpass MAD; "
            f"soft_gate={soft_gate}"
        ),
    )
    save_postmerge_calib(args.out, calib)
    samples_path = args.out.with_suffix(".samples.json")
    samples_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.out}\n"
        f"  n={calib.n_samples} alpha={calib.alpha:.4f} beta={calib.beta_dn:.4f} "
        f"rmse={calib.rmse_dn:.4f} median_ratio={calib.median_ratio:.4f}\n"
        f"  samples -> {samples_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
