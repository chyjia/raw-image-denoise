"""Build offline PTC+VST+BM3D teacher images for a real Burst manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import warp_translation
from .common import (
    DEFAULT_DARK_VARIANCE_PER_S,
    decode_mono10,
    exposure_intercept,
    memmap_frames,
    vst_forward,
    vst_inverse,
)


def tiled_bm3d(
    transformed: np.ndarray,
    tile_size: int = 512,
    overlap: int = 48,
) -> np.ndarray:
    try:
        import bm3d
    except ImportError as error:
        raise RuntimeError(
            "BM3D teacher generation requires: pip install bm3d"
        ) from error

    height, width = transformed.shape
    step = tile_size - overlap
    accumulation = np.zeros_like(transformed, dtype=np.float64)
    weights = np.zeros_like(transformed, dtype=np.float64)
    base_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))
    base_window = np.maximum(base_window, 1e-3)
    visited: set[tuple[int, int]] = set()
    for proposed_y in range(0, height, step):
        y1 = min(height, proposed_y + tile_size)
        y0 = max(0, y1 - tile_size)
        for proposed_x in range(0, width, step):
            x1 = min(width, proposed_x + tile_size)
            x0 = max(0, x1 - tile_size)
            if (y0, x0) in visited:
                continue
            visited.add((y0, x0))
            tile = transformed[y0:y1, x0:x1].astype(np.float64)
            denoised = bm3d.bm3d(tile, sigma_psd=1.0)
            window = base_window[: y1 - y0, : x1 - x0]
            accumulation[y0:y1, x0:x1] += denoised * window
            weights[y0:y1, x0:x1] += window
    return (accumulation / np.maximum(weights, 1e-8)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=48)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dark-variance-per-s",
        type=float,
        default=DEFAULT_DARK_VARIANCE_PER_S,
    )
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = entries[: args.max_files] if args.max_files else entries
    for sequence_index, entry in enumerate(selected, start=1):
        teacher_path = Path(entry["clean"]).with_name(
            Path(entry["clean"]).name.replace("__clean.npy", "__bm3d.npy")
        )
        if teacher_path.exists() and not args.overwrite:
            entry["teacher"] = str(teacher_path)
            print(f"[{sequence_index}/{len(selected)}] Reusing {teacher_path.name}")
            continue

        print(f"[{sequence_index}/{len(selected)}] BM3D {entry['name']}", flush=True)
        frames = memmap_frames(
            Path(entry["path"]),
            int(entry["width"]),
            int(entry["height"]),
        )
        anchor_index = int(entry["anchor_index"])
        shifts = np.load(entry["shifts"])
        offsets = np.load(entry["offsets"])
        exposure_ms = float(entry.get("exposure_ms", 0.0))
        intercept = exposure_intercept(
            exposure_ms,
            dark_variance_per_s=args.dark_variance_per_s,
        )
        image = decode_mono10(frames[anchor_index])
        image = np.clip(image - float(offsets[anchor_index]), 0.0, 1023.0)
        image = warp_translation(
            image,
            float(shifts[anchor_index, 0]),
            float(shifts[anchor_index, 1]),
        )
        teacher = vst_inverse(
            tiled_bm3d(
                vst_forward(image, intercept=intercept),
                tile_size=args.tile_size,
                overlap=args.overlap,
            ),
            intercept=intercept,
        ).astype(np.float32)
        np.save(teacher_path, teacher)
        entry["teacher"] = str(teacher_path)

    output_manifest = args.output_manifest or args.manifest
    output_manifest.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Updated teacher paths in {output_manifest}", flush=True)


if __name__ == "__main__":
    main()
