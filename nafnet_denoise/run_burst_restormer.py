"""Train BurstRestormer (mixed + material FT) and compare vs Adaptive / BM3D."""

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
    parser.add_argument("--skip-phase-a", action="store_true")
    parser.add_argument("--skip-phase-b", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--phase-a-epochs", type=int, default=40)
    parser.add_argument("--phase-b-epochs", type=int, default=15)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--input-frames", type=int, default=16)
    parser.add_argument(
        "--init-burst-checkpoint",
        type=Path,
        default=None,
        help="Warm-start from an existing BurstRestormer checkpoint (shape-matched).",
    )
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\素材\降噪素材")
    material_cache = Path("nafnet_denoise/burst_cache_denoise_material")
    material_manifest = material_cache / "manifest.json"
    material_train = material_cache / "manifest_train.json"
    synthetic_manifest = Path("nafnet_denoise/cache/manifest.json")
    video_manifest = Path("nafnet_denoise/video_burst_cache_davis/manifest.json")
    init_restormer = Path("restormer/checkpoints_v4/best.pt")

    tag = f"{args.input_frames}f"
    phase_a_dir = Path(f"nafnet_denoise/checkpoints_burst_restormer_mixed_{tag}")
    phase_b_dir = Path(f"nafnet_denoise/checkpoints_burst_restormer_material_ft_{tag}")
    compare_dir = Path(f"nafnet_denoise/compare_burst_restormer_vs_bm3d_{tag}")
    # Keep legacy 16f paths when using the original defaults.
    if args.input_frames == 16 and args.init_burst_checkpoint is None:
        phase_a_dir = Path("nafnet_denoise/checkpoints_burst_restormer_mixed")
        phase_b_dir = Path("nafnet_denoise/checkpoints_burst_restormer_material_ft")
        compare_dir = Path("nafnet_denoise/compare_burst_restormer_vs_bm3d")

    if material_manifest.exists() and not material_train.exists():
        split_manifest(
            material_manifest,
            material_train,
            holdout_tokens=["_f10.raw", "_f250.raw"],
        )

    if not args.skip_phase_a:
        if not synthetic_manifest.exists():
            raise SystemExit(f"Missing synthetic manifest: {synthetic_manifest}")
        if not video_manifest.exists():
            raise SystemExit(f"Missing video manifest: {video_manifest}")
        command = [
            python,
            "-u",
            "-m",
            "nafnet_denoise.train_burst_restormer",
            "--dataset-mode",
            "mixed",
            "--manifest",
            str(synthetic_manifest),
            "--video-manifest",
            str(video_manifest),
            "--video-fraction",
            "0.45",
            "--out-dir",
            str(phase_a_dir),
            "--input-frames",
            str(args.input_frames),
            "--width",
            "48",
            "--dim",
            "48",
            "--num-blocks",
            "2,3,3,4",
            "--num-refinement-blocks",
            "2",
            "--patch-size",
            "256",
            "--micro-batch",
            "1",
            "--accum-steps",
            "8",
            "--epochs",
            str(args.phase_a_epochs),
            "--samples-per-epoch",
            str(args.samples_per_epoch),
            "--lr",
            "1e-4",
            "--teacher-weight-start",
            "0.0",
            "--teacher-weight-end",
            "0.0",
            "--flat-hp-weight",
            "0.45",
            "--flat-sigma-weight",
            "0.35",
            "--validation-every",
            "5",
            "--validation-frame-index",
            "-1",
            "--validation-dir",
            str(material_dir),
            "--num-workers",
            "0",
        ]
        if args.init_burst_checkpoint is not None:
            command.extend(["--init-burst-checkpoint", str(args.init_burst_checkpoint)])
        elif init_restormer.exists():
            command.extend(["--init-restormer", str(init_restormer)])
        run(command, phase_a_dir / "train_stdout.log")

    if not args.skip_phase_b:
        resume = phase_a_dir / "best.pt"
        if not resume.exists():
            resume = phase_a_dir / "last.pt"
        if not resume.exists():
            raise SystemExit(f"Missing phase-A checkpoint under {phase_a_dir}")
        if not material_train.exists():
            raise SystemExit(
                f"Missing material train manifest: {material_train}. "
                "Build burst_cache_denoise_material first."
            )
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train_burst_restormer",
                "--dataset-mode",
                "real",
                "--manifest",
                str(material_train),
                "--out-dir",
                str(phase_b_dir),
                "--resume",
                str(resume),
                "--fresh-resume",
                "--input-frames",
                str(args.input_frames),
                "--width",
                "48",
                "--dim",
                "48",
                "--num-blocks",
                "2,3,3,4",
                "--num-refinement-blocks",
                "2",
                "--patch-size",
                "256",
                "--micro-batch",
                "1",
                "--accum-steps",
                "8",
                "--epochs",
                str(args.phase_b_epochs),
                "--samples-per-epoch",
                "384",
                "--lr",
                "3e-5",
                "--teacher-weight-start",
                "0.0",
                "--teacher-weight-end",
                "0.0",
                "--flat-hp-weight",
                "0.55",
                "--flat-sigma-weight",
                "0.45",
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
        if not ckpt.exists():
            raise SystemExit("No BurstRestormer checkpoint available for compare.")
        residual = Path("nafnet_denoise/checkpoints_detail_residual_unet/best.pt")
        compare_cmd = [
            python,
            "-u",
            "-m",
            "nafnet_denoise.compare_burst_restormer_vs_bm3d",
            "--input-dir",
            str(material_dir),
            "--output-dir",
            str(compare_dir),
            "--checkpoint-burst-restormer",
            str(ckpt),
            "--checkpoint-1f",
            "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
            "--checkpoint-4f",
            "nafnet_denoise/checkpoints_4f_des_sigma_flat_long/best.pt",
            "--checkpoint-16f",
            "nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt",
        ]
        if residual.exists():
            compare_cmd.extend(["--checkpoint-residual-4f", str(residual)])
        run(compare_cmd, compare_dir / "compare_stdout.log")
        print(f"Compare complete: {compare_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
