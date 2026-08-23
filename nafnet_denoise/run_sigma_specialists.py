"""Train high/low residual-σ DualHead specialists + routed DES compare.

P0 after hard residual synth failed: stop one net compromising f10 edges while
chasing f0.5 flats. Both specialists resume from split_edge_fid, use measured
FE σ channel + global postmerge_noise.json (not hard-only calib).
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


def base_train_cmd(
    python: str,
    *,
    out_dir: Path,
    resume: Path,
    hard_manifest: Path,
    calib: Path,
    sigma_map: Path,
    material_dir: Path,
    hard_substr: str,
    epochs: int,
    patches: int,
    dark_boost: float,
    des_edge_fid: float,
    edge_focus: str,
    edge_focus_edge_mult: float,
    edge_focus_flat_des_mult: float,
    edge_focus_des_fid_mult: float,
    des_flat_sigma: float,
    des_noise_gain: float,
    hard_des_sigma_mult: float,
    hard_des_hp: float,
    flat_boost: float,
) -> list[str]:
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
        "--postmerge-noise-calib", str(calib),
        "--measured-fe-sigma",
        "--wiener-bm3d-teacher",
        "--wiener-bm3d-sigma", "0.35",
        "--wiener-bm3d-prob", "1.0",
        "--dark-boost", str(dark_boost),
        "--scene-loss-power", "0.5",
        "--scene-loss-ref-fps", "10.0",
        "--scene-loss-max", "3.0",
        "--hard-scene-substr", hard_substr,
        "--hard-scene-sample-mult", "5.0",
        "--hard-scene-loss-mult", "3.0",
        "--hard-des-sigma-mult", str(hard_des_sigma_mult),
        "--hard-des-hp-weight", str(hard_des_hp),
        "--hard-des-use-pure-teacher",
        "--teacher-mode", "partitioned",
        "--distill-edge-harden", "16.0",
        "--des-edge-fid-weight", str(des_edge_fid),
        "--edge-focus-substr", edge_focus,
        "--edge-focus-edge-mult", str(edge_focus_edge_mult),
        "--edge-focus-flat-des-mult", str(edge_focus_flat_des_mult),
        "--edge-focus-des-fid-mult", str(edge_focus_des_fid_mult),
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
        "--epochs", str(epochs),
        "--patches-per-epoch", str(patches),
        "--real-fraction", "1.0",
        "--lr", "2e-6",
        "--teacher-weight-start", "0.70",
        "--teacher-weight-end", "0.50",
        "--highpass-weight", "0.70",
        "--flat-boost", str(flat_boost),
        "--flat-sigma-weight", "0.80",
        "--des-flat-sigma-weight", str(des_flat_sigma),
        "--des-noise-gain-weight", str(des_noise_gain),
        "--dual-aux-weight", "0.30",
        "--edge-weight", "0.18",
        "--validation-every", "4",
        "--validation-frame-index", "-1",
        "--validation-dir", str(material_dir),
        "--num-workers", "0",
    ]
    if sigma_map.exists():
        cmd.extend(["--hard-wiener-bm3d-sigma-map", str(sigma_map)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train-high", action="store_true")
    parser.add_argument("--skip-train-low", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    parser.add_argument("--fe-sigma-threshold", type=float, default=0.50)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    sigma_map = Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json")
    calib = Path("nafnet_denoise/cache/postmerge_noise.json")
    resume = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    out_high = Path("nafnet_denoise/checkpoints_4f_spec_high")
    out_low = Path("nafnet_denoise/checkpoints_4f_spec_low")
    compare_dir = Path("nafnet_denoise/compare_sigma_specialists")

    if not resume.exists():
        raise SystemExit(f"Missing resume: {resume}")
    if not hard_manifest.exists():
        raise SystemExit(f"Missing {hard_manifest}")
    if not calib.exists():
        raise SystemExit(f"Missing global calib: {calib}")

    if not args.skip_train_high:
        # High-σ specialist: f0.5 flats / noise_gain.
        run(
            base_train_cmd(
                python,
                out_dir=out_high,
                resume=resume,
                hard_manifest=hard_manifest,
                calib=calib,
                sigma_map=sigma_map,
                material_dir=material_dir,
                hard_substr=F05_TOKEN,
                epochs=args.epochs,
                patches=args.patches_per_epoch,
                dark_boost=1.0,
                des_edge_fid=0.35,
                edge_focus=F05_TOKEN,
                edge_focus_edge_mult=0.8,
                edge_focus_flat_des_mult=1.4,
                edge_focus_des_fid_mult=0.8,
                des_flat_sigma=1.50,
                des_noise_gain=1.35,
                hard_des_sigma_mult=2.2,
                hard_des_hp=0.45,
                flat_boost=4.0,
            ),
            out_high / "train_stdout.log",
        )

    if not args.skip_train_low:
        # Low-σ specialist: f10 edge_fidelity.
        run(
            base_train_cmd(
                python,
                out_dir=out_low,
                resume=resume,
                hard_manifest=hard_manifest,
                calib=calib,
                sigma_map=sigma_map,
                material_dir=material_dir,
                hard_substr=F10_TOKEN,
                epochs=args.epochs,
                patches=args.patches_per_epoch,
                dark_boost=0.3,
                des_edge_fid=1.10,
                edge_focus=F10_TOKEN,
                edge_focus_edge_mult=1.8,
                edge_focus_flat_des_mult=0.55,
                edge_focus_des_fid_mult=3.5,
                des_flat_sigma=0.90,
                des_noise_gain=0.70,
                hard_des_sigma_mult=1.2,
                hard_des_hp=0.25,
                flat_boost=2.8,
            ),
            out_low / "train_stdout.log",
        )

    if not args.skip_compare:
        ckpt_high = out_high / "best.pt"
        ckpt_low = out_low / "best.pt"
        if not ckpt_high.exists():
            raise SystemExit(f"Missing {ckpt_high}")
        if not ckpt_low.exists():
            raise SystemExit(f"Missing {ckpt_low}")
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_sigma_specialists",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-high", str(ckpt_high),
                "--checkpoint-low", str(ckpt_low),
                "--checkpoint-baseline", str(resume),
                "--fe-sigma-threshold", str(args.fe_sigma_threshold),
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
