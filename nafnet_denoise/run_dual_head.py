"""Train dual-head NAFNet + flat-σ and compare DES vs baseline / BM3D."""

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
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patches-per-epoch", type=int, default=320)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_train = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    synthetic = Path("nafnet_denoise/cache/manifest.json")
    resume = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt")
    if not resume.exists():
        resume = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_dual_head_flat_sigma")
    compare_dir = Path("nafnet_denoise/compare_dual_head_flat_sigma")

    if not args.skip_train:
        if not resume.exists():
            raise SystemExit(f"Missing resume: {resume}")
        if not material_train.exists():
            raise SystemExit(f"Missing {material_train}")
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "dual_head",
            "--real-manifest", str(material_train),
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
            "--real-fraction", "0.85" if synthetic.exists() else "1.0",
            "--lr", "1e-5",
            "--teacher-weight-start", "0.55",
            "--teacher-weight-end", "0.30",
            "--highpass-weight", "0.50",
            "--flat-boost", "2.0",
            "--flat-sigma-weight", "0.45",
            "--dual-aux-weight", "0.30",
            "--edge-weight", "0.20",
            "--validation-every", "4",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        if synthetic.exists():
            idx = cmd.index("--real-manifest")
            cmd[idx:idx] = ["--synthetic-manifest", str(synthetic)]
        run(cmd, out_dir / "train_stdout.log")

    if not args.skip_compare:
        ckpt = out_dir / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing dual-head checkpoint: {ckpt}")
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_wide_flat_sigma",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-baseline-4f", str(resume),
                "--checkpoint-wide-4f", str(ckpt),
                "--label-wide-4f", "dual_head_4f",
                "--checkpoint-1f", "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
                "--checkpoint-16f", "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
