"""Longer validation train for NAFNet-wide and stacked Restormer 4f + DES compare."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--skip-wide", action="store_true")
    parser.add_argument("--skip-restormer", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patches-per-epoch", type=int, default=320)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_train = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    synthetic = Path("nafnet_denoise/cache/manifest.json")
    baseline = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt")
    if not baseline.exists():
        baseline = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat/best.pt")

    wide_src = Path("nafnet_denoise/checkpoints_4f_nafnet_wide_flat_sigma/best.pt")
    resto_src = Path("nafnet_denoise/checkpoints_4f_stacked_restormer_flat_sigma/best.pt")
    wide_dir = Path("nafnet_denoise/checkpoints_4f_nafnet_wide_flat_sigma_long")
    resto_dir = Path("nafnet_denoise/checkpoints_4f_stacked_restormer_flat_sigma_long")
    compare_dir = Path("nafnet_denoise/compare_wide_restormer_long")

    if not material_train.exists():
        raise SystemExit(f"Missing {material_train}")

    if not args.skip_wide:
        if not wide_src.exists():
            raise SystemExit(f"Missing wide warm-start: {wide_src}")
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "nafnet",
            "--synthetic-manifest", str(synthetic),
            "--real-manifest", str(material_train),
            "--out-dir", str(wide_dir),
            "--resume", str(wide_src),
            "--fresh-resume",
            "--use-sigma",
            "--input-frames", "4",
            "--width", "64",
            "--patch-size", "256",
            "--micro-batch", "2",
            "--accum-steps", "4",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "0.85",
            "--lr", "5e-6",
            "--teacher-weight-start", "0.55",
            "--teacher-weight-end", "0.28",
            "--highpass-weight", "0.50",
            "--flat-boost", "2.0",
            "--flat-sigma-weight", "0.45",
            "--edge-weight", "0.18",
            "--validation-every", "5",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        if not synthetic.exists():
            cmd.remove("--synthetic-manifest")
            cmd.remove(str(synthetic))
            fr = cmd.index("--real-fraction")
            cmd[fr + 1] = "1.0"
        run(cmd, wide_dir / "train_stdout.log")

    if not args.skip_restormer:
        if not resto_src.exists():
            raise SystemExit(f"Missing Restormer warm-start: {resto_src}")
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "stacked_restormer",
            "--synthetic-manifest", str(synthetic),
            "--real-manifest", str(material_train),
            "--out-dir", str(resto_dir),
            "--resume", str(resto_src),
            "--fresh-resume",
            "--use-sigma",
            "--input-frames", "4",
            "--dim", "48",
            "--num-blocks", "2,3,3,4",
            "--num-refinement-blocks", "2",
            "--patch-size", "256",
            "--micro-batch", "1",
            "--accum-steps", "8",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "0.80",
            "--lr", "3e-5",
            "--teacher-weight-start", "0.55",
            "--teacher-weight-end", "0.28",
            "--highpass-weight", "0.50",
            "--flat-boost", "2.0",
            "--flat-sigma-weight", "0.45",
            "--edge-weight", "0.18",
            "--validation-every", "5",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        if not synthetic.exists():
            cmd.remove("--synthetic-manifest")
            cmd.remove(str(synthetic))
            fr = cmd.index("--real-fraction")
            cmd[fr + 1] = "1.0"
        run(cmd, resto_dir / "train_stdout.log")

    if not args.skip_compare:
        wide_ckpt = wide_dir / "best.pt"
        resto_ckpt = resto_dir / "best.pt"
        if not wide_ckpt.exists():
            wide_ckpt = wide_src
        if not resto_ckpt.exists():
            resto_ckpt = resto_src
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_wide_flat_sigma",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-baseline-4f", str(baseline),
                "--checkpoint-wide-4f", str(wide_ckpt),
                "--checkpoint-stacked-restormer", str(resto_ckpt),
                "--checkpoint-1f", "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
                "--checkpoint-16f", "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
