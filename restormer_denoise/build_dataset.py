"""Precompute clean references for supervised VST-domain denoising.

For every static Mono10 video we build a near-clean target by averaging all
frames after removing per-frame brightness drift (lamp flicker). The clean
reference, the per-frame offsets, and metadata are cached to disk so training
can sample noisy/clean patch pairs cheaply and keep them in RAM.

Usage:
    python -m restormer_denoise.build_dataset \
        --input-dir "D:/denoise/素材/训练素材" \
        --cache-dir restormer_denoise/cache \
        --exclude-contains 不参加训练
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry


def build_reference(path: Path, subsample: int) -> dict:
    width, height, fps = parse_geometry(path.name)
    exposure_ms = parse_exposure_ms(path.name)
    frames = memmap_frames(path, width, height)
    frame_count = frames.shape[0]

    # Per-frame global mean drives the additive flicker correction.
    frame_means = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        sample = decode_mono10(frames[index, ::subsample, ::subsample])
        frame_means[index] = float(sample.mean())
    series_mean = float(frame_means.mean())
    offsets = (frame_means - series_mean).astype(np.float32)

    # Accumulate the flicker-corrected temporal mean as the clean target.
    accumulator = np.zeros((height, width), dtype=np.float64)
    for index in range(frame_count):
        frame = decode_mono10(frames[index])
        accumulator += frame - offsets[index]
    clean = (accumulator / frame_count).astype(np.float32)
    clean = np.clip(clean, 0.0, 1023.0)

    return {
        "path": str(path),
        "name": path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "exposure_ms": exposure_ms,
        "frame_count": int(frame_count),
        "series_mean": series_mean,
        "clean": clean,
        "offsets": offsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("restormer_denoise/cache"))
    parser.add_argument(
        "--exclude-contains",
        action="append",
        default=[],
        help="Skip files whose path contains this substring (repeatable).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=4,
        help="Pixel stride when estimating per-frame brightness (default: 4).",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    kept = [
        path
        for path in files
        if not any(token in str(path) for token in args.exclude_contains)
    ]
    if not kept:
        raise SystemExit(f"No training videos found under {args.input_dir}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in kept:
        print(f"Building reference: {path.name}")
        info = build_reference(path, args.subsample)
        stem = f"{Path(info['name']).stem}"
        clean_path = args.cache_dir / f"{stem}__clean.npy"
        offsets_path = args.cache_dir / f"{stem}__offsets.npy"
        np.save(clean_path, info["clean"])
        np.save(offsets_path, info["offsets"])
        manifest.append(
            {
                "path": info["path"],
                "name": info["name"],
                "width": info["width"],
                "height": info["height"],
                "fps": info["fps"],
                "exposure_ms": info["exposure_ms"],
                "frame_count": info["frame_count"],
                "series_mean": info["series_mean"],
                "clean": str(clean_path),
                "offsets": str(offsets_path),
            }
        )
        print(
            f"  exposure={info['exposure_ms']:g} ms, frames={info['frame_count']}, "
            f"mean={info['series_mean']:.2f} DN"
        )

    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(manifest)} references to {manifest_path}")


if __name__ == "__main__":
    main()
