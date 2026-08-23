"""Run P2-c → P2-b → deploy bake sequentially (no prompts; defaults)."""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"Failed ({code}): {' '.join(cmd)}")


def mean_des(metrics_csv: Path, method: str = "dual_gated_fps_sigma") -> float | None:
    if not metrics_csv.exists():
        return None
    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    vals = [float(r["des"]) for r in rows if r["method"] == method]
    return sum(vals) / len(vals) if vals else None


def hard_ef(metrics_csv: Path, method: str = "dual_gated_fps_sigma") -> float | None:
    if not metrics_csv.exists():
        return None
    for row in csv.DictReader(metrics_csv.open(encoding="utf-8")):
        if row["method"] == method and "74824541" in row["file"]:
            return float(row["edge_fidelity"])
    return None


def main() -> None:
    python = sys.executable
    blend_ref = 0.9055

    # --- P2-c Laplacian edge ---
    print("=== P2-c lap_edge ===", flush=True)
    run([python, "-u", "-m", "nafnet_denoise.run_lap_edge", "--epochs", "14"])
    lap_csv = Path("nafnet_denoise/compare_lap_edge/metrics.csv")
    lap_des = mean_des(lap_csv)
    lap_ef = hard_ef(lap_csv)
    print(f"P2-c result: mean_DES={lap_des} f10_ef={lap_ef}", flush=True)

    # Prefer lap_edge as P2-b resume if it didn't collapse mean too badly.
    resume = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    lap_ckpt = Path("nafnet_denoise/checkpoints_4f_lap_edge/best.pt")
    if lap_ckpt.exists() and lap_des is not None and lap_des >= blend_ref - 0.002:
        resume = lap_ckpt
        print(f"P2-b will resume from lap_edge ({lap_des:.4f})", flush=True)
    else:
        print(f"P2-b will resume from SOTA (lap mean={lap_des})", flush=True)

    # --- P2-b flat-only FiLM ---
    print("=== P2-b film_flat ===", flush=True)
    run(
        [
            python, "-u", "-m", "nafnet_denoise.run_film_flat",
            "--epochs", "12",
            "--resume", str(resume),
        ]
    )
    film_csv = Path("nafnet_denoise/compare_film_flat/metrics.csv")
    film_des = mean_des(film_csv)
    film_ef = hard_ef(film_csv)
    print(f"P2-b result: mean_DES={film_des} f10_ef={film_ef}", flush=True)

    # --- Deploy bake: compare fps-gated sota-only vs full blend on holdout ---
    print("=== Deploy: fps-cheap blend route compare ===", flush=True)
    run(
        [
            python, "-u", "-m", "nafnet_denoise.compare_sota_edgekd_blend",
            "--input-dir", r"D:\denoise\val_raw",
            "--output-dir", "nafnet_denoise/compare_deploy_blend_route",
        ]
    )

    # Summary
    print("=== LADDER SUMMARY ===", flush=True)
    print(f"blend_edge_soft ref: {blend_ref:.4f}", flush=True)
    print(f"P2-c lap_edge: mean={lap_des} f10_ef={lap_ef}", flush=True)
    print(f"P2-b film_flat: mean={film_des} f10_ef={film_ef}", flush=True)
    deploy_csv = Path("nafnet_denoise/compare_deploy_blend_route/metrics.csv")
    if deploy_csv.exists():
        by: dict[str, list[float]] = defaultdict(list)
        for row in csv.DictReader(deploy_csv.open(encoding="utf-8")):
            if row["method"] in ("input", "temporal_reference"):
                continue
            by[row["method"]].append(float(row["des"]))
        for method, vals in sorted(by.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            print(f"deploy {method}: {sum(vals)/len(vals):.4f}", flush=True)
    print(
        "Deploy API: nafnet_denoise.infer_blend_deploy "
        "(fps<=1 → SOTA only; else blend_edge_soft)",
        flush=True,
    )


if __name__ == "__main__":
    main()
