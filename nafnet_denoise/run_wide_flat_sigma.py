"""Short FT: NAFNet-wide + flat-σ, and 4f stacked Restormer + flat-σ; DES compare."""

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
    parser.add_argument("--skip-nafnet-wide", action="store_true")
    parser.add_argument("--skip-stacked-restormer", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_train = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    synthetic = Path("nafnet_denoise/cache/manifest.json")
    resume_naf = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt")
    if not resume_naf.exists():
        resume_naf = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat/best.pt")
    init_restormer = Path("restormer/checkpoints_v4/best.pt")

    wide_dir = Path("nafnet_denoise/checkpoints_4f_nafnet_wide_flat_sigma")
    resto_dir = Path("nafnet_denoise/checkpoints_4f_stacked_restormer_flat_sigma")
    compare_dir = Path("nafnet_denoise/compare_wide_flat_sigma")

    if not material_train.exists():
        raise SystemExit(f"Missing material train manifest: {material_train}")

    if not args.skip_nafnet_wide:
        if not resume_naf.exists():
            raise SystemExit(f"Missing NAFNet resume: {resume_naf}")
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "nafnet",
            "--real-manifest", str(material_train),
            "--out-dir", str(wide_dir),
            "--resume", str(resume_naf),
            "--fresh-resume",
            "--use-sigma",
            "--input-frames", "4",
            "--width", "64",
            "--patch-size", "256",
            "--micro-batch", "2",
            "--accum-steps", "4",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "0.85" if synthetic.exists() else "1.0",
            "--lr", "8e-6",
            "--teacher-weight-start", "0.55",
            "--teacher-weight-end", "0.30",
            "--highpass-weight", "0.50",
            "--flat-boost", "2.0",
            "--flat-sigma-weight", "0.40",
            "--edge-weight", "0.18",
            "--validation-every", "4",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        if synthetic.exists():
            cmd[cmd.index("--real-manifest"):cmd.index("--real-manifest")] = [
                "--synthetic-manifest", str(synthetic),
            ]
        run(cmd, wide_dir / "train_stdout.log")

    if not args.skip_stacked_restormer:
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "stacked_restormer",
            "--real-manifest", str(material_train),
            "--out-dir", str(resto_dir),
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
            "--real-fraction", "0.80" if synthetic.exists() else "1.0",
            "--lr", "5e-5",
            "--teacher-weight-start", "0.55",
            "--teacher-weight-end", "0.30",
            "--highpass-weight", "0.50",
            "--flat-boost", "2.0",
            "--flat-sigma-weight", "0.40",
            "--edge-weight", "0.18",
            "--validation-every", "4",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        if init_restormer.exists():
            cmd.extend(["--init-restormer", str(init_restormer)])
        if synthetic.exists():
            cmd[cmd.index("--real-manifest"):cmd.index("--real-manifest")] = [
                "--synthetic-manifest", str(synthetic),
            ]
        run(cmd, resto_dir / "train_stdout.log")

    if not args.skip_compare:
        wide_ckpt = wide_dir / "best.pt"
        resto_ckpt = resto_dir / "best.pt"
        cmd = [
            python, "-u", "-m", "nafnet_denoise.compare_wide_flat_sigma",
            "--input-dir", str(material_dir),
            "--output-dir", str(compare_dir),
            "--checkpoint-baseline-4f", str(resume_naf),
            "--checkpoint-1f", "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
            "--checkpoint-16f", "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
        ]
        if wide_ckpt.exists():
            cmd.extend(["--checkpoint-wide-4f", str(wide_ckpt)])
        if resto_ckpt.exists():
            cmd.extend(["--checkpoint-stacked-restormer", str(resto_ckpt)])
        run(cmd, compare_dir / "compare_stdout.log")
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
