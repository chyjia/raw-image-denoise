"""P1-b: edge-only FT under gated FE after P1-a blend win.

Resume split_edge_fid (mean SOTA); freeze trunk/flat; train ending_edge with
stronger DES edge_fidelity + KD from edge_kd_new_fe (already in gated FE domain).
Goal: single-net mean ≥ 0.9055 and f10 ef ≥ 0.968 (close remaining BM3D gap).
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
    parser.add_argument("--edge-kd-mix", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=3e-6)
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
    # Prefer gated-domain edgeKD student as edge teacher; fall back to spec_low.
    edge_kd = Path("nafnet_denoise/checkpoints_4f_edge_kd_new_fe/best.pt")
    if not edge_kd.exists():
        edge_kd = Path("nafnet_denoise/checkpoints_4f_spec_low/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_edge_only_ft")
    compare_dir = Path("nafnet_denoise/compare_edge_only_ft")

    if not resume.exists():
        raise SystemExit(f"Missing resume: {resume}")
    if not edge_kd.exists():
        raise SystemExit(f"Missing edge KD teacher: {edge_kd}")
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
            "--train-edge-only",
            "--init-edge-from", str(edge_kd),
            "--edge-kd-checkpoint", str(edge_kd),
            "--edge-kd-mix", str(args.edge_kd_mix),
            "--wiener-bm3d-teacher",
            "--wiener-bm3d-sigma", "0.35",
            "--wiener-bm3d-prob", "1.0",
            "--dark-boost", "0.5",
            "--scene-loss-power", "0.5",
            "--scene-loss-ref-fps", "10.0",
            "--scene-loss-max", "3.0",
            "--hard-scene-substr", ",".join(HARD_TOKENS),
            "--hard-scene-sample-mult", "4.0",
            "--hard-scene-loss-mult", "2.0",
            "--hard-des-sigma-mult", "1.2",
            "--hard-des-hp-weight", "0.20",
            "--hard-des-use-pure-teacher",
            "--teacher-mode", "partitioned",
            "--distill-edge-harden", "16.0",
            "--des-edge-fid-weight", "1.40",
            "--edge-focus-substr", F10_TOKEN,
            "--edge-focus-edge-mult", "2.2",
            "--edge-focus-flat-des-mult", "0.40",
            "--edge-focus-des-fid-mult", "4.5",
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
            "--teacher-weight-start", "0.65",
            "--teacher-weight-end", "0.45",
            "--highpass-weight", "0.60",
            "--flat-boost", "3.0",
            "--flat-sigma-weight", "0.70",
            "--des-flat-sigma-weight", "0.90",
            "--des-noise-gain-weight", "0.75",
            "--dual-aux-weight", "0.50",
            "--edge-weight", "0.32",
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
            "goal mean>=0.9055 & f10 ef>=0.968",
            flush=True,
        )


if __name__ == "__main__":
    main()
