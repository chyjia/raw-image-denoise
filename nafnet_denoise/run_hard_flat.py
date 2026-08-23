"""Hard-scene flat-σ FT for the two clips still behind BM3D (f0.5 + f10).

Builds a train manifest that includes the previously held-out f10 clip, then
short-FT from scene_bm3d with name-based sample/loss boost and stronger
Wiener-BM3D teacher on those clips only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARD_TOKENS = (
    "20260719153940814",  # f0.5
    "20260725174824541",  # f10
)


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


def build_hard_manifest(train_path: Path, full_path: Path, out_path: Path) -> Path:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    by_token: dict[str, dict] = {}
    for entry in [*train, *full]:
        name = str(entry.get("name") or Path(entry["path"]).name)
        path = str(entry.get("path", ""))
        for token in HARD_TOKENS:
            if token in name or token in path:
                by_token[token] = entry
    missing = [token for token in HARD_TOKENS if token not in by_token]
    if missing:
        raise SystemExit(f"Hard scenes missing from manifests: {missing}")

    # Start from train; append any hard clip not already present.
    present = set()
    for entry in train:
        name = str(entry.get("name") or Path(entry["path"]).name)
        path = str(entry.get("path", ""))
        for token in HARD_TOKENS:
            if token in name or token in path:
                present.add(token)
    merged = list(train)
    for token in HARD_TOKENS:
        if token not in present:
            merged.append(by_token[token])
            print(f"Appended hard clip {token} -> train manifest", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(merged)} entries)", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    # Prefer ASCII junction to avoid .bat / console encoding corruption.
    ascii_val = Path(r"D:\denoise\val_raw")
    if ascii_val.exists():
        material_dir = ascii_val
    train_manifest = Path("nafnet_denoise/burst_cache_denoise_material/manifest_train.json")
    full_manifest = Path("nafnet_denoise/burst_cache_denoise_material/manifest.json")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    resume = Path("nafnet_denoise/checkpoints_4f_scene_bm3d/best.pt")
    out_dir = Path("nafnet_denoise/checkpoints_4f_hard_flat")
    compare_dir = Path("nafnet_denoise/compare_hard_flat")

    if not full_manifest.exists():
        raise SystemExit(f"Missing {full_manifest}")
    if not train_manifest.exists():
        raise SystemExit(f"Missing {train_manifest}")
    build_hard_manifest(train_manifest, full_manifest, hard_manifest)

    if not args.skip_train:
        if not resume.exists():
            raise SystemExit(f"Missing resume: {resume}")
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
            "--wiener-bm3d-teacher",
            "--wiener-bm3d-sigma", "0.35",
            "--hard-wiener-bm3d-sigma", "0.50",
            "--wiener-bm3d-prob", "1.0",
            "--dark-boost", "0.5",
            "--scene-loss-power", "0.5",
            "--scene-loss-ref-fps", "10.0",
            "--scene-loss-max", "3.0",
            "--hard-scene-substr", ",".join(HARD_TOKENS),
            "--hard-scene-sample-mult", "4.0",
            "--hard-scene-loss-mult", "3.0",
            "--loss-domain", "dn",
            "--teacher-mode", "blend",
            "--bm3d-flat-alpha", "0.92",
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
            "--lr", "2e-6",
            "--teacher-weight-start", "0.70",
            "--teacher-weight-end", "0.50",
            "--highpass-weight", "0.80",
            "--flat-boost", "4.0",
            "--flat-sigma-weight", "0.90",
            "--des-flat-sigma-weight", "1.40",
            "--des-noise-gain-weight", "1.10",
            "--dual-aux-weight", "0.30",
            "--edge-weight", "0.12",
            "--validation-every", "5",
            "--validation-frame-index", "-1",
            "--validation-dir", str(material_dir),
            "--num-workers", "0",
        ]
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
