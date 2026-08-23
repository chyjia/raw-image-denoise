"""P5-d: leave-burst temporal (N2N-lite) flat FT on low-fps hard scene.

No new captures available → default to temporal teacher (leave-burst mean) with
strong flat σ / noise_gain losses. Resume SOTA DualHead under gated FE.
Goal: improve f0.5 ng inside the network (bilateral remains deploy fallback).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

F05_TOKEN = "20260719153940814"


def run(command: list[str], log_path: Path) -> None:
    print(" ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        code = subprocess.call(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if code != 0:
        raise SystemExit(f"Command failed ({code}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-6)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    resume = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_p5d_n2n_flat")
    compare_dir = Path("nafnet_denoise/compare_p5d_n2n_flat")

    if not resume.exists():
        raise SystemExit(f"Missing resume: {resume}")
    if not hard_manifest.exists():
        raise SystemExit(f"Missing {hard_manifest}")

    if not args.skip_train:
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "dual_head",
            "--use-nonlocal",
            "--nonlocal-count", "1",
            "--align-soft-gate",
            "--align-threshold", "0.03",
            "--align-temperature", "0.015",
            "--align-drop-prob", "0.05",
            "--wiener-front-end",
            "--wiener-merge-frames", "16",
            "--wiener-tile", "32",
            "--wiener-overlap", "16",
            "--wiener-c-factor", "8.0",
            "--wiener-spatial",
            "--wiener-spatial-c-factor", "1.0",
            "--wiener-spatial-adaptive",
            "--wiener-spatial-flat-c-mult", "1.75",
            "--wiener-spatial-edge-c-mult", "0.45",
            "--wiener-spatial-dark-boost", "0.35",
            "--wiener-spatial-mask-harden", "8.0",
            "--wiener-spatial-freq-gamma", "0.15",
            "--wiener-fe-schedule", "fps_sigma",
            # N2N-lite: leave-burst temporal teacher (no BM3D)
            "--teacher-mode", "temporal",
            "--dark-boost", "0.5",
            "--scene-loss-power", "0.5",
            "--scene-loss-ref-fps", "10.0",
            "--scene-loss-max", "3.0",
            "--hard-scene-substr", F05_TOKEN,
            "--hard-scene-sample-mult", "4.0",
            "--hard-scene-loss-mult", "2.5",
            "--des-edge-fid-weight", "0.90",
            "--loss-domain", "dn",
            "--real-manifest", str(hard_manifest),
            "--out-dir", str(out_dir),
            "--resume", str(resume),
            "--fresh-resume",
            "--use-sigma",
            "--input-frames", "4",
            "--width", "32",
            "--patch-size", "256",
            "--micro-batch", "2",
            "--accum-steps", "4",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "1.0",
            "--lr", str(args.lr),
            "--teacher-weight-start", "0.90",
            "--teacher-weight-end", "0.75",
            "--highpass-weight", "0.80",
            "--flat-boost", "4.0",
            "--flat-sigma-weight", "0.90",
            "--des-flat-sigma-weight", "1.20",
            "--des-noise-gain-weight", "1.10",
            "--dual-aux-weight", "0.45",
            "--edge-weight", "0.20",
            "--validation-every", "4",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        run(cmd, out_dir / "train_stdout.log")

    if not args.skip_compare:
        ckpt = out_dir / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_gated_fe",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-4f", str(ckpt),
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)
        print(
            "refs: P5 deploy bilateral ~0.9100 / P4 deploy 0.9059 / "
            "goal: gated single-net or better f0.5 ng without ef collapse",
            flush=True,
        )


if __name__ == "__main__":
    main()
