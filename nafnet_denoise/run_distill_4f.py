"""Orchestrate BM3D-teacher refresh, 4f distillation, then BM3D re-compare."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log_path: Path | None = None) -> None:
    print(" ".join(command), flush=True)
    if log_path is None:
        code = subprocess.call(command, cwd=ROOT)
    else:
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
    parser.add_argument("--skip-teachers", action="store_true")
    parser.add_argument("--overwrite-teachers", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--edge-weight", type=float, default=0.20)
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_ptc_black60/best.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_ptc_black60_bm3d_distill"),
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_adaptive_vs_bm3d_distill"),
    )
    args = parser.parse_args()

    python = sys.executable
    real_manifest = "nafnet_denoise/burst_cache_train/manifest.json"
    synthetic_manifest = "nafnet_denoise/cache/manifest.json"
    out_4f = args.out_dir
    compare_dir = args.compare_dir

    if not args.skip_teachers:
        teacher_cmd = [
            python,
            "-u",
            "-m",
            "nafnet_denoise.build_bm3d_teacher",
            "--manifest",
            real_manifest,
        ]
        if args.overwrite_teachers:
            teacher_cmd.append("--overwrite")
        run(teacher_cmd)

    if not args.skip_train:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train_distill",
                "--synthetic-manifest",
                synthetic_manifest,
                "--real-manifest",
                real_manifest,
                "--out-dir",
                str(out_4f),
                "--resume",
                str(args.resume),
                "--fresh-resume",
                "--input-frames",
                "4",
                "--width",
                "32",
                "--patch-size",
                "256",
                "--micro-batch",
                "2",
                "--accum-steps",
                "4",
                "--epochs",
                str(args.epochs),
                "--patches-per-epoch",
                "256",
                "--real-fraction",
                "0.65",
                "--lr",
                "5e-5",
                "--teacher-weight-start",
                "0.45",
                "--teacher-weight-end",
                "0.20",
                "--highpass-weight",
                "0.25",
                "--edge-weight",
                str(args.edge_weight),
                "--validation-every",
                "5",
                "--validation-frame-index",
                "-1",
                "--validation-dir",
                r"D:\denoise\素材\降噪素材",
                "--num-workers",
                "0",
            ],
            out_4f / "train_stdout.log",
        )

    if not args.skip_compare:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.compare_adaptive_vs_bm3d",
                "--input-dir",
                r"D:\denoise\素材\降噪素材",
                "--output-dir",
                str(compare_dir),
                "--checkpoint-1f",
                "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
                "--checkpoint-4f",
                str(out_4f / "best.pt"),
                "--checkpoint-16f",
                "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
