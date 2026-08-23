"""Sequential 1f → 4f → 16f retrain under the new PTC / black-level protocol."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = process.wait()
    if code != 0:
        raise SystemExit(f"Command failed ({code}): {' '.join(command)}\nSee {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-1f", action="store_true")
    parser.add_argument("--skip-4f", action="store_true")
    parser.add_argument("--skip-16f", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    manifest = "nafnet_denoise/cache/manifest.json"
    video_manifest = "nafnet_denoise/video_burst_cache_davis/manifest.json"
    validation_manifest = "nafnet_denoise/burst_cache_ptc_new_validation/manifest.json"
    validation_dir = r"D:\denoise\素材\降噪素材"

    out_1f = Path("nafnet_denoise/checkpoints_1f_ptc_black60")
    out_4f = Path("nafnet_denoise/checkpoints_4f_ptc_black60")
    out_16f = Path("nafnet_denoise/checkpoints_16f_ptc_black60_mixed")
    bench_dir = Path("nafnet_denoise/validation_results_adaptive_ptc_black60")

    if not args.skip_1f:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train",
                "--manifest",
                manifest,
                "--out-dir",
                str(out_1f),
                "--input-frames",
                "1",
                "--width",
                "32",
                "--patch-size",
                "256",
                "--micro-batch",
                "4",
                "--accum-steps",
                "2",
                "--epochs",
                "120",
                "--patches-per-epoch",
                "256",
                "--validation-every",
                "10",
                "--validation-frame-index",
                "-1",
                "--validation-dir",
                validation_dir,
                "--lr",
                "2e-4",
                "--exposure-ms-min",
                "10",
                "--exposure-ms-max",
                "2000",
                "--black-level-dn",
                "60",
                "--dark-variance-per-s-min",
                "0",
                "--dark-variance-per-s-max",
                "6",
                "--noise-jitter",
                "0.15",
                "--num-workers",
                "0",
            ],
            out_1f / "train_stdout.log",
        )

    if not args.skip_4f:
        resume_4f = []
        previous = Path("nafnet_denoise/checkpoints_4f_quality/best.pt")
        if previous.exists():
            resume_4f = ["--resume", str(previous)]
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train",
                "--manifest",
                manifest,
                "--out-dir",
                str(out_4f),
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
                "120",
                "--patches-per-epoch",
                "256",
                "--validation-every",
                "10",
                "--validation-frame-index",
                "-1",
                "--validation-dir",
                validation_dir,
                "--lr",
                "1e-4",
                "--exposure-ms-min",
                "10",
                "--exposure-ms-max",
                "2000",
                "--black-level-dn",
                "60",
                "--dark-variance-per-s-min",
                "0",
                "--dark-variance-per-s-max",
                "6",
                "--noise-jitter",
                "0.15",
                "--num-workers",
                "0",
                *resume_4f,
                *(["--fresh-resume"] if resume_4f else []),
            ],
            out_4f / "train_stdout.log",
        )

    if not args.skip_16f:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.train_burst",
                "--dataset-mode",
                "mixed",
                "--manifest",
                manifest,
                "--video-manifest",
                video_manifest,
                "--video-fraction",
                "0.5",
                "--validation-manifest",
                validation_manifest,
                "--out-dir",
                str(out_16f),
                "--resume",
                "nafnet_denoise/checkpoints_16f_ptc_black60_transfer/best.pt",
                "--reset-best",
                "--input-frames",
                "16",
                "--patch-size",
                "256",
                "--width",
                "48",
                "--micro-batch",
                "2",
                "--accum-steps",
                "1",
                "--epochs",
                "555",
                "--samples-per-epoch",
                "128",
                "--validation-every",
                "5",
                "--validation-frame-index",
                "-1",
                "--lr",
                "3e-5",
                "--noise-jitter",
                "0.15",
                "--max-motion-shift",
                "2.0",
                "--max-motion-rotation",
                "0.25",
                "--motion-strength",
                "1.0",
                "--motion-blur-strength",
                "1.0",
                "--exposure-ms-min",
                "10",
                "--exposure-ms-max",
                "2000",
                "--black-level-dn",
                "60",
                "--black-drift-sigma",
                "1.0",
                "--fixed-pattern-sigma",
                "0.5",
                "--row-noise-sigma",
                "0.3",
                "--column-noise-sigma",
                "0.2",
                "--dark-variance-per-s-min",
                "0",
                "--dark-variance-per-s-max",
                "6",
                "--mean-weight",
                "0.5",
                "--num-workers",
                "0",
            ],
            out_16f / "train_stdout.log",
        )

    if not args.skip_benchmark:
        run(
            [
                python,
                "-u",
                "-m",
                "nafnet_denoise.benchmark_multiframe",
                "--input-dir",
                validation_dir,
                "--output-dir",
                str(bench_dir),
                "--frame-index",
                "-1",
                "--checkpoint-1f",
                str(out_1f / "best.pt"),
                "--checkpoint-4f",
                str(out_4f / "best.pt"),
                "--checkpoint-16f",
                str(out_16f / "best.pt"),
            ],
            bench_dir / "benchmark_stdout.log",
        )
        print(f"Benchmark complete: {bench_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
