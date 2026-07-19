"""Build aligned real-camera Burst training references and a manifest.

The explicit ``不参加训练`` directory is always excluded. The filename ``f``
field is FPS; explicit ``exposure_XXms`` wins, otherwise this camera setup uses
the frame interval (1000 / FPS) as exposure time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import estimate_translation, warp_translation
from .common import (
    PTC_INTERCEPT,
    PTC_SLOPE,
    decode_mono10,
    memmap_frames,
    parse_exposure_ms,
    parse_geometry,
)


def central_roi(height: int, width: int, fraction: float = 0.60) -> tuple[slice, slice]:
    roi_h = int(height * fraction)
    roi_w = int(width * fraction)
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    return slice(y0, y0 + roi_h), slice(x0, x0 + roi_w)


def safe_stem(path: Path, root: Path) -> str:
    return "__".join(path.relative_to(root).with_suffix("").parts)


def build_sequence(
    path: Path,
    input_root: Path,
    cache_dir: Path,
    reference_frames: int,
    alignment_downsample: int,
    max_shift: float,
    reliability_ratio: float,
) -> dict:
    width, height, fps = parse_geometry(path.name)
    exposure_ms = parse_exposure_ms(path.name)
    frames = memmap_frames(path, width, height)
    frame_count = int(frames.shape[0])
    anchor_index = frame_count // 2
    ys, xs = central_roi(height, width)

    anchor = decode_mono10(frames[anchor_index])
    anchor_roi = anchor[ys, xs]
    anchor_mean = float(anchor_roi.mean())
    shifts = np.zeros((frame_count, 2), dtype=np.float32)
    responses = np.ones(frame_count, dtype=np.float32)
    brightness_offsets = np.zeros(frame_count, dtype=np.float32)

    for index in range(frame_count):
        roi = decode_mono10(frames[index, ys, xs])
        offset = float(roi.mean()) - anchor_mean
        brightness_offsets[index] = offset
        if index == anchor_index:
            continue
        corrected_roi = roi - offset
        tx, ty, response = estimate_translation(
            anchor_roi,
            corrected_roi,
            downsample=alignment_downsample,
            max_shift=max_shift,
        )
        if response < 0.03:
            tx, ty = 0.0, 0.0
        shifts[index] = (tx, ty)
        responses[index] = response

    sample_count = min(reference_frames, frame_count)
    sample_indices = np.unique(
        np.linspace(0, frame_count - 1, sample_count, dtype=np.int32)
    )
    accumulation = np.zeros((height, width), dtype=np.float64)
    square_accumulation = np.zeros((height, width), dtype=np.float64)
    for index in sample_indices:
        image = decode_mono10(frames[int(index)])
        image = np.clip(image - float(brightness_offsets[index]), 0.0, 1023.0)
        aligned = warp_translation(
            image,
            float(shifts[index, 0]),
            float(shifts[index, 1]),
        )
        accumulation += aligned
        square_accumulation += np.square(aligned, dtype=np.float64)

    count = float(len(sample_indices))
    clean = np.clip(accumulation / count, 0.0, 1023.0).astype(np.float32)
    temporal_variance = np.maximum(
        square_accumulation / count - np.square(clean, dtype=np.float64),
        0.0,
    )
    expected_variance = PTC_SLOPE * clean + PTC_INTERCEPT
    reliability = temporal_variance <= reliability_ratio * np.maximum(expected_variance, 1.0)
    reliability[:8, :] = False
    reliability[-8:, :] = False
    reliability[:, :8] = False
    reliability[:, -8:] = False

    stem = safe_stem(path, input_root)
    clean_path = cache_dir / f"{stem}__clean.npy"
    mask_path = cache_dir / f"{stem}__mask.npy"
    shifts_path = cache_dir / f"{stem}__shifts.npy"
    offsets_path = cache_dir / f"{stem}__offsets.npy"
    responses_path = cache_dir / f"{stem}__responses.npy"
    np.save(clean_path, clean)
    np.save(mask_path, reliability.astype(np.uint8))
    np.save(shifts_path, shifts)
    np.save(offsets_path, brightness_offsets)
    np.save(responses_path, responses)

    return {
        "path": str(path),
        "scene": path.parent.name if path.parent != input_root else "__root__",
        "name": path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_interval_ms": 1000.0 / fps,
        "exposure_ms": exposure_ms,
        "frame_count": frame_count,
        "anchor_index": anchor_index,
        "reference_frame_count": int(len(sample_indices)),
        "anchor_mean_dn": anchor_mean,
        "alignment_response_median": float(np.median(responses)),
        "reliable_fraction": float(reliability.mean()),
        "clean": str(clean_path),
        "mask": str(mask_path),
        "shifts": str(shifts_path),
        "offsets": str(offsets_path),
        "responses": str(responses_path),
        "teacher": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("nafnet_denoise/burst_cache"),
    )
    parser.add_argument("--reference-frames", type=int, default=64)
    parser.add_argument("--alignment-downsample", type=int, default=4)
    parser.add_argument("--max-shift", type=float, default=16.0)
    parser.add_argument("--reliability-ratio", type=float, default=2.5)
    parser.add_argument("--exclude-contains", action="append", default=["不参加训练"])
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Disable the default 不参加训练 exclusion (for a separate validation cache).",
    )
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    exclude_tokens = [] if args.include_validation else args.exclude_contains
    files = [
        path
        for path in files
        if not any(token in str(path) for token in exclude_tokens)
    ]
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No training Mono10 RAW files found under {args.input_dir}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sequence_index, path in enumerate(files, start=1):
        print(f"[{sequence_index}/{len(files)}] {path}")
        entry = build_sequence(
            path,
            args.input_dir,
            args.cache_dir,
            args.reference_frames,
            args.alignment_downsample,
            args.max_shift,
            args.reliability_ratio,
        )
        manifest.append(entry)
        print(
            f"  frames={entry['frame_count']} exposure={entry['exposure_ms']:.3f} ms "
            f"align_response={entry['alignment_response_median']:.3f} "
            f"reliable={entry['reliable_fraction']:.1%}"
        )

    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(manifest)} sequences to {manifest_path}")


if __name__ == "__main__":
    main()
