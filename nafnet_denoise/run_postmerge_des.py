"""Post-merge distribution alignment + strong DES-loss short FT.

Resume from wiener_hybrid DualHead. Train on the same temporal+spatial Wiener
front-end used at inference for both real and synthetic bursts; up-weight
low-fps/dark scenes and DES-identical flat σ / noise-gain losses.
"""

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
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patches-per-epoch", type=int, default=320)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_train = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    synthetic = Path("nafnet_denoise/cache/manifest.json")
    resume = Path("nafnet_denoise/checkpoints_4f_wiener_hybrid/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_postmerge_des")
    compare_dir = Path("nafnet_denoise/compare_postmerge_des")

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
            "--align-drop-prob", "0.05",
            "--wiener-front-end",
            "--wiener-merge-frames", "16",
            "--wiener-tile", "32",
            "--wiener-overlap", "16",
            "--wiener-c-factor", "8.0",
            "--wiener-spatial",
            "--wiener-spatial-c-factor", "1.0",
            "--dark-boost", "1.0",
            "--dark-target-mean-max", "350.0",
            "--loss-domain", "dn",
            "--teacher-mode", "blend",
            "--bm3d-flat-alpha", "0.70",
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
            "--real-fraction", "0.80" if synthetic.exists() else "1.0",
            "--lr", "3e-6",
            "--teacher-weight-start", "0.65",
            "--teacher-weight-end", "0.40",
            "--highpass-weight", "0.70",
            "--flat-boost", "3.0",
            "--flat-sigma-weight", "0.70",
            "--des-flat-sigma-weight", "1.10",
            "--des-noise-gain-weight", "0.80",
            "--dual-aux-weight", "0.30",
            "--edge-weight", "0.18",
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
                python, "-u", "-m", "nafnet_denoise.compare_wiener_merge",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-4f", str(ckpt),
                "--checkpoint-1f", "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
                "--spatial-wiener",
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
