"""P2-a: distill blend_edge_soft (SOTA↔edgeKD) into a single DualHead.

Student resumes split_edge_fid under fps-gated FE. Frozen teachers:
  edge*edgeKD + flat*SOTA (same recipe as P1-a deploy).
Goal: single-net mean DES ≥ 0.9055 without dual forward at infer.
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
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    parser.add_argument("--blend-kd-mix", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-6)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    sigma_map = Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json")
    resume = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edgekd = Path("nafnet_denoise/checkpoints_4f_edge_kd_new_fe/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_blend_distill")
    compare_dir = Path("nafnet_denoise/compare_blend_distill")

    if not resume.exists():
        raise SystemExit(f"Missing resume: {resume}")
    if not edgekd.exists():
        raise SystemExit(f"Missing edgeKD teacher: {edgekd}")
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
            "--blend-kd-sota", str(sota),
            "--blend-kd-edgekd", str(edgekd),
            "--blend-kd-mix", str(args.blend_kd_mix),
            "--blend-kd-temperature", "8.0",
            "--init-edge-from", str(edgekd),
            "--wiener-bm3d-teacher",
            "--wiener-bm3d-sigma", "0.35",
            "--wiener-bm3d-prob", "1.0",
            "--dark-boost", "0.5",
            "--scene-loss-power", "0.5",
            "--scene-loss-ref-fps", "10.0",
            "--scene-loss-max", "3.0",
            "--hard-scene-substr", ",".join(HARD_TOKENS),
            "--hard-scene-sample-mult", "3.0",
            "--hard-scene-loss-mult", "2.0",
            "--hard-des-sigma-mult", "1.3",
            "--hard-des-hp-weight", "0.20",
            "--hard-des-use-pure-teacher",
            "--teacher-mode", "partitioned",
            "--distill-edge-harden", "16.0",
            "--des-edge-fid-weight", "1.10",
            "--edge-focus-substr", F10_TOKEN,
            "--edge-focus-edge-mult", "1.8",
            "--edge-focus-flat-des-mult", "0.50",
            "--edge-focus-des-fid-mult", "3.5",
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
            "--teacher-weight-start", "0.85",
            "--teacher-weight-end", "0.70",
            "--highpass-weight", "0.65",
            "--flat-boost", "3.0",
            "--flat-sigma-weight", "0.70",
            "--des-flat-sigma-weight", "1.00",
            "--des-noise-gain-weight", "0.85",
            "--dual-aux-weight", "0.45",
            "--edge-weight", "0.28",
            "--validation-every", "4",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
        if sigma_map.exists():
            cmd.extend(["--hard-wiener-bm3d-sigma-map", str(sigma_map)])
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
            "refs: blend_edge_soft 0.9055 / gated SOTA 0.9050 / "
            "goal single-net mean>=0.9055",
            flush=True,
        )


if __name__ == "__main__":
    main()
