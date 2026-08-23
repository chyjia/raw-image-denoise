"""P2-b: selective FiLM on flat tower only (fix full-net FiLM mean drop).

Freeze edge head; measured FE-σ + σ jitter; gated FE. Goal: lift f0.5 ng
without giving back f10 ef.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F05_TOKEN = "20260719153940814"
F10_TOKEN = "20260725174824541"
HARD_TOKENS = (F05_TOKEN, F10_TOKEN)


def run(command: list[str], log_path: Path) -> None:
    print(" ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"Command failed ({code}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
        help="Prefer lap_edge best.pt if it won; else SOTA.",
    )
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    sigma_map = Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json")
    out_dir = Path("nafnet_denoise/checkpoints_4f_film_flat")
    compare_dir = Path("nafnet_denoise/compare_film_flat")

    if not args.resume.exists():
        raise SystemExit(f"Missing resume: {args.resume}")

    if not args.skip_train:
        cmd = [
            python, "-u", "-m", "nafnet_denoise.train_distill",
            "--arch", "dual_head", "--use-nonlocal", "--nonlocal-count", "1",
            "--align-soft-gate", "--align-threshold", "0.03",
            "--align-temperature", "0.015", "--align-drop-prob", "0.05",
            "--wiener-front-end", "--wiener-merge-frames", "16",
            "--wiener-tile", "32", "--wiener-overlap", "16", "--wiener-c-factor", "8.0",
            "--wiener-spatial", "--wiener-spatial-c-factor", "1.0",
            "--wiener-spatial-adaptive",
            "--wiener-spatial-flat-c-mult", "1.75",
            "--wiener-spatial-edge-c-mult", "0.45",
            "--wiener-spatial-dark-boost", "0.35",
            "--wiener-spatial-mask-harden", "8.0",
            "--wiener-spatial-freq-gamma", "0.15",
            "--wiener-fe-schedule", "fps_sigma",
            "--use-sigma", "--measured-fe-sigma",
            "--use-sigma-film-flat", "--sigma-film-jitter", "0.25",
            "--freeze-edge-head",
            "--wiener-bm3d-teacher", "--wiener-bm3d-sigma", "0.35",
            "--wiener-bm3d-prob", "1.0",
            "--dark-boost", "0.6", "--scene-loss-power", "0.6",
            "--scene-loss-ref-fps", "10.0", "--scene-loss-max", "3.0",
            "--hard-scene-substr", ",".join(HARD_TOKENS),
            "--hard-scene-sample-mult", "4.0", "--hard-scene-loss-mult", "2.5",
            "--hard-des-sigma-mult", "1.8", "--hard-des-hp-weight", "0.35",
            "--hard-des-use-pure-teacher",
            "--teacher-mode", "partitioned", "--distill-edge-harden", "16.0",
            "--des-edge-fid-weight", "0.80",
            "--edge-focus-substr", F10_TOKEN,
            "--edge-focus-edge-mult", "1.2",
            "--edge-focus-flat-des-mult", "1.0",
            "--edge-focus-des-fid-mult", "1.5",
            "--loss-domain", "dn",
            "--real-manifest", str(hard_manifest),
            "--out-dir", str(out_dir), "--resume", str(args.resume), "--fresh-resume",
            "--input-frames", "4", "--width", "32",
            "--patch-size", "256", "--micro-batch", "2", "--accum-steps", "4",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "1.0", "--lr", "2e-6",
            "--teacher-weight-start", "0.75", "--teacher-weight-end", "0.55",
            "--highpass-weight", "0.75", "--flat-boost", "3.8",
            "--flat-sigma-weight", "0.90", "--des-flat-sigma-weight", "1.30",
            "--des-noise-gain-weight", "1.10",
            "--dual-aux-weight", "0.40", "--edge-weight", "0.20",
            "--validation-every", "4", "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir), "--num-workers", "0",
        ]
        if sigma_map.exists():
            cmd.extend(["--hard-wiener-bm3d-sigma-map", str(sigma_map)])
        run(cmd, out_dir / "train_stdout.log")

    if not args.skip_compare:
        ckpt = out_dir / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing {ckpt}")
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


if __name__ == "__main__":
    main()
