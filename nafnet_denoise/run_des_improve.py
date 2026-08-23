"""Two-phase DES-oriented fine-tune: partitioned teacher, then 降噪素材 FT."""

from __future__ import annotations

import argparse
import json
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


def split_manifest(src: Path, train_out: Path, holdout_tokens: list[str]) -> None:
    entries = json.loads(src.read_text(encoding="utf-8"))
    train = [
        entry
        for entry in entries
        if not any(token in entry["name"] for token in holdout_tokens)
    ]
    if not train:
        raise SystemExit("No training entries left after holdout split.")
    train_out.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(train)}/{len(entries)} train entries to {train_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build-material", action="store_true")
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--skip-phase-b", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--phase-a-epochs", type=int, default=40)
    parser.add_argument("--phase-b-epochs", type=int, default=15)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_cache = Path("nafnet_denoise/burst_cache_denoise_material")
    material_manifest = material_cache / "manifest.json"
    material_train_manifest = material_cache / "manifest_train.json"
    synthetic_manifest = "nafnet_denoise/cache/manifest.json"
    real_manifest = "nafnet_denoise/burst_cache_train/manifest.json"

    phase_a_dir = Path("nafnet_denoise/checkpoints_4f_des_partition")
    phase_b_dir = Path("nafnet_denoise/checkpoints_4f_des_material_ft")
    compare_dir = Path("nafnet_denoise/compare_adaptive_vs_bm3d_des")

    if not args.skip_build_material:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.build_burst_dataset",
                "--input-dir",
                str(material_dir),
                "--cache-dir",
                str(material_cache),
                "--include-validation",
                "--reference-frames",
                "32",
            ],
            material_cache / "build_stdout.log",
        )
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.build_bm3d_teacher",
                "--manifest",
                str(material_manifest),
            ],
            material_cache / "teacher_stdout.log",
        )
        split_manifest(
            material_manifest,
            material_train_manifest,
            holdout_tokens=["_f10.raw", "_f250.raw"],
        )

    if not args.skip_phase_a:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train_distill",
                "--synthetic-manifest",
                synthetic_manifest,
                "--real-manifest",
                real_manifest,
                "--out-dir",
                str(phase_a_dir),
                "--resume",
                "nafnet_denoise/checkpoints_4f_ptc_black60_bm3d_distill_edge/best.pt",
                "--fresh-resume",
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
                str(args.phase_a_epochs),
                "--patches-per-epoch",
                "256",
                "--real-fraction",
                "0.70",
                "--lr",
                "4e-5",
                "--teacher-weight-start",
                "0.50",
                "--teacher-weight-end",
                "0.25",
                "--highpass-weight",
                "0.30",
                "--edge-weight",
                "0.18",
                "--validation-every",
                "5",
                "--validation-frame-index",
                "-1",
                "--validation-dir",
                str(material_dir),
                "--num-workers",
                "0",
            ],
            phase_a_dir / "train_stdout.log",
        )

    if not args.skip_phase_b:
        resume = phase_a_dir / "best.pt"
        if not resume.exists():
            resume = Path(
                "nafnet_denoise/checkpoints_4f_ptc_black60_bm3d_distill_edge/best.pt"
            )
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train_distill",
                "--real-manifest",
                str(material_train_manifest),
                "--out-dir",
                str(phase_b_dir),
                "--resume",
                str(resume),
                "--fresh-resume",
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
                str(args.phase_b_epochs),
                "--patches-per-epoch",
                "192",
                "--real-fraction",
                "1.0",
                "--lr",
                "1e-5",
                "--teacher-weight-start",
                "0.55",
                "--teacher-weight-end",
                "0.30",
                "--highpass-weight",
                "0.35",
                "--edge-weight",
                "0.15",
                "--validation-every",
                "3",
                "--validation-frame-index",
                "-1",
                "--validation-dir",
                str(material_dir),
                "--num-workers",
                "0",
            ],
            phase_b_dir / "train_stdout.log",
        )

    if not args.skip_compare:
        ckpt = phase_b_dir / "best.pt"
        if not ckpt.exists():
            ckpt = phase_a_dir / "best.pt"
        run(
            [
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
                str(ckpt),
                "--checkpoint-16f",
                "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
            ],
            compare_dir / "compare_stdout.log",
        )
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
