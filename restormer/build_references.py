"""Precompute clean temporal references and a training manifest.

For every Mono10 video (excluding the validation folder) this computes a
brightness-aligned trimmed temporal mean as a near-clean supervision target,
saves it as float32 .npy, and records metadata in a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("restormer/cache"))
    parser.add_argument(
        "--exclude-contains",
        default="不参加训练",
        help="Skip files whose path contains this substring (validation set).",
    )
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    args = parser.parse_args()

    refs_dir = args.output_dir / "clean_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    files = common.list_mono10_files(args.input_dir, args.exclude_contains)
    if not files:
        raise SystemExit(f"No Mono10 training files found under {args.input_dir}")

    entries = []
    for path in files:
        meta = common.parse_meta(path)
        data = common.open_raw(path, meta)
        frame_count = int(data.shape[0])
        print(
            f"{path.name}: {frame_count} frames, exposure~{meta.exposure_ms:.1f} ms"
            f" -> computing reference"
        )
        reference = common.robust_temporal_reference(
            data, args.max_frames, args.trim_fraction
        )
        _, flicker_std = common.frame_roi_mean_series(data, args.max_frames)
        ref_name = "__".join(path.relative_to(args.input_dir).with_suffix("").parts)
        ref_path = refs_dir / f"{ref_name}.npy"
        np.save(ref_path, reference)

        entries.append(
            {
                "raw_path": str(path),
                "ref_path": str(ref_path),
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "exposure_ms": meta.exposure_ms,
                "exposure_condition": common.exposure_to_condition(meta.exposure_ms),
                "frame_count": frame_count,
                "ref_mean_dn": float(reference.mean()),
                "frame_roi_mean_std": flicker_std,
            }
        )

    manifest_path = args.output_dir / "train_manifest.json"
    manifest_path.write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    priors = common.build_exposure_brightness_priors(entries)
    priors_path = args.output_dir / "exposure_brightness_priors.json"
    priors_path.write_text(
        json.dumps({str(k): v for k, v in sorted(priors.items())}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(entries)} references and manifest to {manifest_path}")
    print(f"Wrote exposure brightness priors to {priors_path}")


if __name__ == "__main__":
    main()
