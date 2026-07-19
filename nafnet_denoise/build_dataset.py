"""Precompute mono luminance caches from PixelShift200 .mat files.

Each ``pixelshift_*.mat`` stores a ``ps4k`` array (H, W, 4) uint16. We convert
it to float32 luminance and save a compact cache so training can sample patches
without reloading 50 MB MAT files every step.

Usage:
    python -m nafnet_denoise.build_dataset \
        --input-dir "D:/denoise/素材/PixelShift200_train" \
        --cache-dir nafnet_denoise/cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from .common import ps4k_to_mono_luminance


def build_entry(path: Path) -> dict:
    data = scipy.io.loadmat(str(path))
    if "ps4k" not in data:
        raise KeyError(f"{path.name} does not contain a ps4k array.")
    mono = ps4k_to_mono_luminance(data["ps4k"])
    height, width = mono.shape
    return {
        "path": str(path),
        "name": path.name,
        "width": int(width),
        "height": int(height),
        "mono_min": float(mono.min()),
        "mono_max": float(mono.max()),
        "mono_mean": float(mono.mean()),
        "mono": mono.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("nafnet_denoise/cache"))
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("pixelshift_*.mat"))
    if not files:
        raise SystemExit(f"No pixelshift_*.mat files found in {args.input_dir}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in files:
        print(f"Caching {path.name} ...")
        info = build_entry(path)
        stem = path.stem
        mono_path = args.cache_dir / f"{stem}__mono.npy"
        np.save(mono_path, info["mono"].astype(np.float32))
        manifest.append(
            {
                "path": info["path"],
                "name": info["name"],
                "width": info["width"],
                "height": info["height"],
                "mono_min": info["mono_min"],
                "mono_max": info["mono_max"],
                "mono_mean": info["mono_mean"],
                "mono": str(mono_path),
            }
        )
        print(
            f"  {info['width']}x{info['height']} "
            f"mean={info['mono_mean']:.1f} range=[{info['mono_min']:.0f}, {info['mono_max']:.0f}]"
        )

    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
