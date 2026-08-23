"""Expand flat-boost Dual+NL from 4f to 8f; gated DES compare."""

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patches-per-epoch", type=int, default=320)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_train = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    synthetic = Path("nafnet_denoise/cache/manifest.json")
    resume = Path("nafnet_denoise/checkpoints_4f_flat_boost/best.pt")
    if not resume.exists():
        resume = Path("nafnet_denoise/checkpoints_4f_dn_domain/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_8f_flat_boost")
    compare_dir = Path("nafnet_denoise/compare_8f_flat_boost")

    if not args.skip_train:
        if not resume.exists():
            raise SystemExit(f"Missing resume: {resume}")
        if not material_train.exists():
            raise SystemExit(f"Missing {material_train}")
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "dual_head",
            "--use-nonlocal",
            "--nonlocal-count", "1",
            "--align-soft-gate",
            "--align-threshold", "0.03",
            "--align-temperature", "0.015",
            "--align-drop-prob", "0.10",
            "--loss-domain", "dn",
            "--teacher-mode", "partitioned",
            "--real-manifest", str(material_train),
            "--out-dir", str(out_dir),
            "--resume", str(resume),
            "--fresh-resume",
            "--use-sigma",
            "--input-frames", "8",
            "--width", "32",
            "--patch-size", "256",
            "--micro-batch", "2",
            "--accum-steps", "4",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "0.85" if synthetic.exists() else "1.0",
            "--lr", "5e-6",
            "--teacher-weight-start", "0.70",
            "--teacher-weight-end", "0.45",
            "--highpass-weight", "0.80",
            "--flat-boost", "3.5",
            "--flat-sigma-weight", "0.90",
            "--dual-aux-weight", "0.35",
            "--edge-weight", "0.15",
            "--validation-every", "5",
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
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_gated",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-4f", str(ckpt),
                "--checkpoint-baseline-4f",
                "nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt",
                "--checkpoint-1f", "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
                "--checkpoint-16f",
                "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
