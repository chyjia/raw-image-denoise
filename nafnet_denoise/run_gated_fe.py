"""P0-a: run fps/σ-gated FE compare on frozen split_edge_fid (no training)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    ckpt = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    out = Path("nafnet_denoise/compare_gated_fe")
    if not ckpt.exists():
        raise SystemExit(f"Missing {ckpt}")

    cmd = [
        python, "-u", "-m", "nafnet_denoise.compare_gated_fe",
        "--input-dir", str(material_dir),
        "--output-dir", str(out),
        "--checkpoint-4f", str(ckpt),
    ]
    print(" ".join(cmd), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "compare_stdout.log"
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"Compare failed ({code}); see {log}")
    print(f"Compare complete: {out / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
