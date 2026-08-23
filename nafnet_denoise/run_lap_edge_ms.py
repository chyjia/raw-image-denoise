"""P3-a: multi-scale Laplacian edge arm + stronger xy-grad loss.

Resume lap_edge; freeze flat; add half-res lap branch. Then compare alone and
as SOTA↔edge blend vs mean 0.9058.
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
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    sigma_map = Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json")
    resume = Path("nafnet_denoise/checkpoints_4f_lap_edge/best.pt")
    if not resume.exists():
        resume = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edgekd = Path("nafnet_denoise/checkpoints_4f_edge_kd_new_fe/best.pt")
    sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms")
    compare_dir = Path("nafnet_denoise/compare_lap_edge_ms")
    blend_dir = Path("nafnet_denoise/compare_sota_lap_ms_blend")

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
            "--use-lap-edge", "--use-lap-edge-ms", "--freeze-flat-head",
            "--edge-kd-checkpoint", str(edgekd if edgekd.exists() else resume),
            "--edge-kd-mix", "0.80",
            "--wiener-bm3d-teacher", "--wiener-bm3d-sigma", "0.35",
            "--wiener-bm3d-prob", "1.0",
            "--dark-boost", "0.5", "--scene-loss-power", "0.5",
            "--scene-loss-ref-fps", "10.0", "--scene-loss-max", "3.0",
            "--hard-scene-substr", ",".join(HARD_TOKENS),
            "--hard-scene-sample-mult", "3.5", "--hard-scene-loss-mult", "2.0",
            "--hard-des-sigma-mult", "1.1", "--hard-des-hp-weight", "0.12",
            "--hard-des-use-pure-teacher",
            "--teacher-mode", "partitioned", "--distill-edge-harden", "16.0",
            "--des-edge-fid-weight", "1.60",
            "--mixed-grad-xy-weight", "0.40",
            "--edge-focus-substr", F10_TOKEN,
            "--edge-focus-edge-mult", "2.2",
            "--edge-focus-flat-des-mult", "0.40",
            "--edge-focus-des-fid-mult", "4.5",
            "--loss-domain", "dn",
            "--real-manifest", str(hard_manifest),
            "--out-dir", str(out_dir), "--resume", str(resume), "--fresh-resume",
            "--use-sigma", "--input-frames", "4", "--width", "32",
            "--patch-size", "256", "--micro-batch", "2", "--accum-steps", "4",
            "--epochs", str(args.epochs),
            "--patches-per-epoch", str(args.patches_per_epoch),
            "--real-fraction", "1.0", "--lr", "2e-6",
            "--teacher-weight-start", "0.65", "--teacher-weight-end", "0.45",
            "--highpass-weight", "0.50", "--flat-boost", "2.2",
            "--flat-sigma-weight", "0.50", "--des-flat-sigma-weight", "0.70",
            "--des-noise-gain-weight", "0.60",
            "--dual-aux-weight", "0.55", "--edge-weight", "0.36",
            "--grad-weight", "0.40",
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
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_sota_edgekd_blend",
                "--input-dir", str(material_dir),
                "--output-dir", str(blend_dir),
                "--checkpoint-sota", str(sota),
                "--checkpoint-edgekd", str(ckpt),
            ],
            blend_dir / "compare_stdout.log",
        )
        print(f"Compare alone: {compare_dir / 'metrics.csv'}", flush=True)
        print(f"Compare blend: {blend_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
