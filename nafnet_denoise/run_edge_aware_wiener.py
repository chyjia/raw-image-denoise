"""P1-a: edge-aware Wiener γ FE sweep on frozen split_edge_fid (+ optional FT)."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

F05 = "20260719153940814"
F10 = "20260725174824541"
HARD = (F05, F10)

# Map compare method suffix -> train flags.
TRAIN_FROM_DUAL = {
    "dual_ea_mild": dict(
        flat=1.75, edge=0.45, dark=0.35, harden=8.0, gamma=0.0
    ),
    "dual_ea_mild_gamma": dict(
        flat=1.75, edge=0.45, dark=0.35, harden=8.0, gamma=0.15
    ),
    "dual_ea_mid": dict(
        flat=2.0, edge=0.40, dark=0.40, harden=10.0, gamma=0.10
    ),
    "dual_ea_protect": dict(
        flat=2.25, edge=0.30, dark=0.50, harden=12.0, gamma=0.12
    ),
}


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


def summarize(metrics: Path) -> tuple[str | None, float, dict[str, float]]:
    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    by: dict[str, list[float]] = defaultdict(list)
    hard: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        method = row["method"]
        if not (method.startswith("dual_") or method == "vst_bm3d"):
            continue
        des = float(row["des"])
        by[method].append(des)
        name = row["file"]
        if F05 in name:
            hard["f0.5"][method] = des
        if F10 in name:
            hard["f10"][method] = des
    means = {m: sum(v) / len(v) for m, v in by.items()}
    # Prefer edge-aware configs that beat baseline_spatial; else best overall dual_*.
    baseline = means.get("dual_baseline_spatial", 0.0)
    candidates = [
        (m, d)
        for m, d in means.items()
        if m.startswith("dual_ea_")
    ]
    best = None
    best_des = -1.0
    for name, des in candidates:
        f10_ok = hard.get("f10", {}).get(name, 0.0) >= hard.get("f10", {}).get(
            "dual_baseline_spatial", 0.0
        ) - 0.003
        if des >= baseline - 1e-6 and f10_ok and des > best_des:
            best, best_des = name, des
    if best is None and candidates:
        best, best_des = max(candidates, key=lambda item: item[1])
    print("--- FE sweep mean DES ---", flush=True)
    for method, des in sorted(means.items(), key=lambda item: -item[1]):
        print(f"{method}: {des:.4f}", flush=True)
    if best is not None:
        print(
            f"Selected for FT: {best} DES={best_des:.4f} "
            f"(baseline={baseline:.4f})",
            flush=True,
        )
    return best, best_des, means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--force-train-config",
        type=str,
        default="",
        help="Force FT config key e.g. dual_ea_mid (skip auto-select).",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patches-per-epoch", type=int, default=256)
    args = parser.parse_args()

    python = sys.executable
    material_dir = Path(r"D:\denoise\val_raw")
    if not material_dir.exists():
        material_dir = Path(r"D:\denoise\素材\降噪素材")
    ckpt = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    compare_dir = Path("nafnet_denoise/compare_edge_aware_wiener")
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    sigma_map = Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json")
    out_dir = Path("nafnet_denoise/checkpoints_4f_edge_aware_wiener")
    ft_compare = Path("nafnet_denoise/compare_edge_aware_wiener_ft")

    if not ckpt.exists():
        raise SystemExit(f"Missing {ckpt}")

    selected = args.force_train_config.strip() or None
    means: dict[str, float] = {}
    if not args.skip_compare:
        run(
            [
                python, "-u", "-m", "nafnet_denoise.compare_edge_aware_wiener",
                "--input-dir", str(material_dir),
                "--output-dir", str(compare_dir),
                "--checkpoint-4f", str(ckpt),
            ],
            compare_dir / "compare_stdout.log",
        )
        selected_auto, _des, means = summarize(compare_dir / "metrics.csv")
        if not selected:
            selected = selected_auto

    if args.skip_train:
        return
    if not selected or selected not in TRAIN_FROM_DUAL:
        print(
            f"No edge-aware config selected for FT (selected={selected}); skip train.",
            flush=True,
        )
        return
    baseline = means.get("dual_baseline_spatial", 0.9039)
    sel_des = means.get(selected, 0.0)
    if means and sel_des + 1e-6 < baseline:
        print(
            f"Selected {selected} DES={sel_des:.4f} < baseline {baseline:.4f}; "
            "skip FT (FE-only already worse).",
            flush=True,
        )
        return

    cfg = TRAIN_FROM_DUAL[selected]
    if not hard_manifest.exists():
        raise SystemExit(f"Missing {hard_manifest}")
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
        "--wiener-spatial-flat-c-mult", str(cfg["flat"]),
        "--wiener-spatial-edge-c-mult", str(cfg["edge"]),
        "--wiener-spatial-dark-boost", str(cfg["dark"]),
        "--wiener-spatial-mask-harden", str(cfg["harden"]),
        "--wiener-spatial-freq-gamma", str(cfg["gamma"]),
        "--wiener-bm3d-teacher",
        "--wiener-bm3d-sigma", "0.35",
        "--wiener-bm3d-prob", "1.0",
        "--dark-boost", "0.5",
        "--scene-loss-power", "0.5",
        "--scene-loss-ref-fps", "10.0",
        "--scene-loss-max", "3.0",
        "--hard-scene-substr", ",".join(HARD),
        "--hard-scene-sample-mult", "4.0",
        "--hard-scene-loss-mult", "2.5",
        "--hard-des-sigma-mult", "1.8",
        "--hard-des-hp-weight", "0.35",
        "--hard-des-use-pure-teacher",
        "--teacher-mode", "partitioned",
        "--distill-edge-harden", "16.0",
        "--des-edge-fid-weight", "0.80",
        "--edge-focus-substr", F10,
        "--edge-focus-edge-mult", "1.5",
        "--edge-focus-flat-des-mult", "0.70",
        "--edge-focus-des-fid-mult", "3.0",
        "--loss-domain", "dn",
        "--real-manifest", str(hard_manifest),
        "--out-dir", str(out_dir),
        "--resume", str(ckpt),
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
        "--highpass-weight", "0.70",
        "--flat-boost", "3.5",
        "--flat-sigma-weight", "0.80",
        "--des-flat-sigma-weight", "1.20",
        "--des-noise-gain-weight", "1.00",
        "--dual-aux-weight", "0.30",
        "--edge-weight", "0.18",
        "--validation-every", "4",
        "--validation-frame-index", "-1",
        "--validation-dir", str(material_dir),
        "--num-workers", "0",
    ]
    if sigma_map.exists():
        cmd.extend(["--hard-wiener-bm3d-sigma-map", str(sigma_map)])
    (out_dir / "selected_config.txt").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_config.txt").write_text(
        f"{selected}\t{cfg}\n", encoding="utf-8"
    )
    run(cmd, out_dir / "train_stdout.log")

    ft_ckpt = out_dir / "best.pt"
    if not ft_ckpt.exists():
        raise SystemExit(f"Missing {ft_ckpt}")
    run(
        [
            python, "-u", "-m", "nafnet_denoise.compare_wiener_merge",
            "--input-dir", str(material_dir),
            "--output-dir", str(ft_compare),
            "--checkpoint-4f", str(ft_ckpt),
            "--checkpoint-1f", "nafnet_denoise/checkpoints_1f_ptc_black60/best.pt",
            "--spatial-wiener",
        ],
        ft_compare / "compare_stdout.log",
    )
    print(f"FT compare: {ft_compare / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
