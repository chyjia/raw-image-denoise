"""P2-d runner: fps/edge fuse blend↔FiLM on val_raw."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"D:\denoise\val_raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_blend_film_fuse"),
    )
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nafnet_denoise.compare_blend_film_fuse",
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
