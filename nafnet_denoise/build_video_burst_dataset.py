"""Index DAVIS full-resolution JPEG sequences for synthetic Mono10 burst training.

The resulting manifest points at ordered per-sequence JPEG frames. Training
code loads consecutive frames as clean motion bursts and synthesizes Mono10
sensor noise from the current PTC/black-level/dark-current model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def discover_sequences(images_root: Path) -> list[Path]:
    sequences = [path for path in sorted(images_root.iterdir()) if path.is_dir()]
    if not sequences:
        raise SystemExit(f"No sequence folders found under {images_root}")
    return sequences


def list_frames(sequence_dir: Path) -> list[Path]:
    frames = sorted(sequence_dir.glob("*.jpg")) + sorted(sequence_dir.glob("*.jpeg"))
    if len(frames) < 2:
        frames = sorted(sequence_dir.glob("*.png"))
    if len(frames) < 2:
        raise ValueError(f"{sequence_dir.name}: need at least two frames")
    return frames


def probe_geometry(frame_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {frame_path}")
    height, width = image.shape[:2]
    return width, height


def build_manifest(
    davis_root: Path,
    cache_dir: Path,
    split: str,
    min_frames: int,
) -> Path:
    images_root = davis_root / "DAVIS" / "JPEGImages" / "Full-Resolution"
    if not images_root.exists():
        raise SystemExit(f"Missing DAVIS images root: {images_root}")

    split_file = davis_root / "DAVIS" / "ImageSets" / "2017" / f"{split}.txt"
    if split_file.exists():
        names = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sequences = [images_root / name for name in names if (images_root / name).is_dir()]
    else:
        sequences = discover_sequences(images_root)

    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for sequence_dir in sequences:
        frames = list_frames(sequence_dir)
        if len(frames) < min_frames:
            continue
        width, height = probe_geometry(frames[0])
        entries.append(
            {
                "name": sequence_dir.name,
                "source": "davis2017",
                "split": split,
                "width": width,
                "height": height,
                "frame_count": len(frames),
                "frames": [str(path) for path in frames],
            }
        )
        print(
            f"{sequence_dir.name}: frames={len(frames)} "
            f"size={width}x{height}"
        )

    if not entries:
        raise SystemExit("No usable DAVIS sequences found.")

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    citation = cache_dir / "CITATION.txt"
    citation.write_text(
        "DAVIS 2017 TrainVal Full-Resolution\n"
        "Cite: Pont-Tuset et al., arXiv:1704.00675 and Perazzi et al., CVPR 2016.\n"
        "Images/annotations follow the DAVIS project terms; 2017 annotations are CC BY 4.0.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(entries)} sequences to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--davis-root",
        type=Path,
        default=Path(r"D:\denoise\素材\DAVIS2017"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("nafnet_denoise/video_burst_cache_davis"),
    )
    parser.add_argument("--split", choices=("train", "val", "trainval"), default="train")
    parser.add_argument("--min-frames", type=int, default=16)
    args = parser.parse_args()
    split = "train" if args.split == "train" else args.split
    if args.split == "trainval":
        # Merge train+val names if both exist.
        train_manifest = build_manifest(
            args.davis_root,
            args.cache_dir / "train",
            "train",
            args.min_frames,
        )
        val_manifest = build_manifest(
            args.davis_root,
            args.cache_dir / "val",
            "val",
            args.min_frames,
        )
        train_entries = json.loads(train_manifest.read_text(encoding="utf-8"))
        val_entries = json.loads(val_manifest.read_text(encoding="utf-8"))
        merged = train_entries + val_entries
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        merged_path = args.cache_dir / "manifest.json"
        merged_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Merged {len(merged)} sequences to {merged_path}")
        return
    build_manifest(args.davis_root, args.cache_dir, split, args.min_frames)


if __name__ == "__main__":
    main()
