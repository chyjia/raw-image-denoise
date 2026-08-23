"""Stronger DES push: longer sigma-FT, larger residual UNet, flat BM3D L1."""

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
    parser.add_argument("--skip-sigma-ft", action="store_true")
    parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--sigma-epochs", type=int, default=60)
    parser.add_argument("--residual-epochs", type=int, default=50)
    parser.add_argument("--residual-width", type=int, default=48)
    parser.add_argument("--residual-depth", type=int, default=2)
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_des_sigma_flat/best.pt"),
        help="Prefer continuing the previous sigma-conditioned checkpoint.",
    )
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_train = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    real_manifest = Path("nafnet_denoise/burst_cache_train/manifest.json")
    synthetic_manifest = Path("nafnet_denoise/cache/manifest.json")

    if not args.resume.exists():
        fallback = Path("nafnet_denoise/checkpoints_4f_des_material_ft/best.pt")
        if fallback.exists():
            print(f"Resume missing ({args.resume}); falling back to {fallback}", flush=True)
            args.resume = fallback
        else:
            raise SystemExit(f"Missing resume checkpoint: {args.resume}")

    sigma_dir = Path("nafnet_denoise/checkpoints_4f_des_sigma_flat_long")
    residual_dir = Path("nafnet_denoise/checkpoints_detail_residual_unet")
    compare_dir = Path("nafnet_denoise/compare_adaptive_vs_bm3d_sigma_unet")

    if not args.skip_sigma_ft:
        train_manifest = material_train if material_train.exists() else real_manifest
        # Continue from an already sigma-expanded ckpt when possible.
        already_sigma = "sigma" in str(args.resume).lower()
        command = [
            python,
            "-u",
            "-m",
            "nafnet_denoise.train_distill",
            "--synthetic-manifest",
            str(synthetic_manifest),
            "--real-manifest",
            str(train_manifest),
            "--out-dir",
            str(sigma_dir),
            "--resume",
            str(args.resume),
            "--fresh-resume",
            "--use-sigma",
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
            str(args.sigma_epochs),
            "--patches-per-epoch",
            "256",
            "--real-fraction",
            "0.80",
            "--lr",
            "8e-6",
            "--teacher-weight-start",
            "0.70",
            "--teacher-weight-end",
            "0.45",
            "--highpass-weight",
            "0.55",
            "--flat-boost",
            "2.50",
            "--edge-weight",
            "0.20",
            "--validation-every",
            "5",
            "--validation-frame-index",
            "-1",
            "--validation-dir",
            str(material_dir),
            "--num-workers",
            "0",
        ]
        if not synthetic_manifest.exists():
            # Drop synthetic args if cache is missing.
            idx = command.index("--synthetic-manifest")
            del command[idx : idx + 2]
            fr = command.index("--real-fraction")
            command[fr + 1] = "1.0"
        _ = already_sigma  # documented intent only
        run(command, sigma_dir / "train_stdout.log")

    base_ckpt = sigma_dir / "best.pt"
    if not base_ckpt.exists():
        base_ckpt = args.resume
        print(f"Long sigma FT best missing; residual base={base_ckpt}", flush=True)

    residual_manifest = str(material_train if material_train.exists() else real_manifest)
    if not args.skip_residual:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train_detail_residual",
                "--real-manifest",
                residual_manifest,
                "--base-checkpoint",
                str(base_ckpt),
                "--out-dir",
                str(residual_dir),
                "--use-sigma",
                "--input-frames",
                "4",
                "--width",
                "32",
                "--residual-width",
                str(args.residual_width),
                "--residual-depth",
                str(args.residual_depth),
                "--patch-size",
                "256",
                "--micro-batch",
                "2",
                "--accum-steps",
                "4",
                "--epochs",
                str(args.residual_epochs),
                "--patches-per-epoch",
                "320",
                "--lr",
                "1.5e-4",
                "--flat-weight",
                "1.0",
                "--flat-l1-weight",
                "2.5",
                "--edge-suppress",
                "1.0",
                "--highpass-weight",
                "0.65",
                "--validation-every",
                "4",
                "--validation-frame-index",
                "-1",
                "--validation-dir",
                str(material_dir),
                "--num-workers",
                "0",
            ],
            residual_dir / "train_stdout.log",
        )

    if not args.skip_compare:
        residual_ckpt = residual_dir / "best.pt"
        compare_cmd = [
            python,
            "-u",
            "-m",
            "nafnet_denoise.compare_adaptive_vs_bm3d",
            "--input-dir",
            str(material_dir),
            "--output-dir",
            str(compare_dir),
            "--checkpoint-1f",
            "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
            "--checkpoint-4f",
            str(base_ckpt),
            "--checkpoint-16f",
            "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
        ]
        if residual_ckpt.exists():
            compare_cmd.extend(["--checkpoint-residual-4f", str(residual_ckpt)])
        run(compare_cmd, compare_dir / "compare_stdout.log")
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
