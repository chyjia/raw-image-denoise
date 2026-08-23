"""P15 meta: Young/FitNet multi-scale edge FT on P13 D4 deploy (skipped by P13 early-stop).

Baseline: d4_r90_flip DES ≈ 0.9252
Priority:
  1) ms-edge feature FT (train_distill --ms-edge-weight)
  2) mild mask + ms-edge combos
  3) optional output blends of FT winners

Weight-soup / plain D4 catalog already saturated in P13/P14.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .compare_sota_edgekd_blend import _spatial_on_merged
from .infer import load_model
from .p8_fusion import deploy_p7_base
from .p13_tta import geom_forward, geom_inverse
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9252
D4 = ("id", "r90", "r270", "r90_lr", "r270_lr")
SOTA_CKPT = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
EDGE_CKPT = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")
EDGE_FALLBACK = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")


@dataclass
class Recipe:
    name: str
    family: str
    augs: tuple[str, ...] = D4
    target: str = "edge"
    epochs: int = 4
    patches: int = 128
    lr: float = 1.5e-6
    ms_edge_weight: float = 0.35
    ms_edge_scales: int = 3
    highpass_weight: float = 0.45
    des_edge_fid: float = 1.4
    mask_ratio: float = 0.0
    mixed_grad_xy: float = 0.0


def edge_path() -> Path:
    return EDGE_CKPT if EDGE_CKPT.exists() else EDGE_FALLBACK


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4_flip", family="baseline")]
    configs = [
        (0.25, 3, 1.5e-6, 4, 0.0, 0.0),
        (0.35, 3, 1.5e-6, 4, 0.0, 0.0),
        (0.50, 3, 1.2e-6, 4, 0.0, 0.0),
        (0.35, 4, 1.5e-6, 5, 0.0, 0.0),
        (0.40, 3, 1e-6, 6, 0.0, 0.0),
        (0.35, 3, 1.5e-6, 4, 0.25, 0.0),
        (0.45, 3, 2e-6, 4, 0.0, 0.0),
        (0.30, 2, 1.5e-6, 4, 0.0, 0.0),
        (0.55, 3, 1e-6, 5, 0.0, 0.0),
        (0.35, 3, 8e-7, 8, 0.0, 0.0),
        (0.40, 4, 1.5e-6, 4, 0.35, 0.0),
        (0.25, 3, 2e-6, 3, 0.0, 0.0),
        (0.35, 3, 1.5e-6, 4, 0.0, 0.25),
        (0.40, 3, 1.5e-6, 5, 0.20, 0.15),
        (0.60, 3, 1e-6, 4, 0.0, 0.0),
        (0.35, 5, 1.2e-6, 4, 0.0, 0.0),
        (0.28, 3, 1.8e-6, 4, 0.30, 0.0),
        (0.42, 3, 1.5e-6, 6, 0.0, 0.20),
        (0.33, 4, 1e-6, 5, 0.15, 0.10),
        (0.50, 2, 1.5e-6, 4, 0.0, 0.0),
    ]
    for i, (w, sc, lr, ep, mr, mg) in enumerate(configs):
        recipes.append(
            Recipe(
                name=f"feat_{i}_w{w:g}_s{sc}",
                family="feat_ft",
                ms_edge_weight=w,
                ms_edge_scales=sc,
                lr=lr,
                epochs=ep,
                mask_ratio=mr,
                mixed_grad_xy=mg,
            )
        )
    # sota-arm FT (usually worse but try a few)
    for i, w in enumerate([0.35, 0.45, 0.30]):
        recipes.append(
            Recipe(
                name=f"feat_sota_{i}_w{w:g}",
                family="feat_ft",
                target="sota",
                ms_edge_weight=w,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_feat_{i}",
                family="feat_ft",
                ms_edge_weight=0.2 + 0.03 * (i % 12),
                ms_edge_scales=2 + (i % 3),
                lr=1e-6 * (1.0 + 0.15 * (i % 6)),
                epochs=3 + (i % 5),
                mask_ratio=0.0 if i % 3 else 0.25,
            )
        )
    return recipes[:100]


def summarize(des_list, f10, f05, f1) -> dict:
    return {
        "mean_des": float(sum(des_list) / max(len(des_list), 1)),
        "f10_ef": float(f10["edge_fidelity"]) if f10 else float("nan"),
        "f05_ef": float(f05["edge_fidelity"]) if f05 else float("nan"),
        "f05_ng": float(f05["noise_gain"]) if f05 else float("nan"),
        "f1_ef": float(f1["edge_fidelity"]) if f1 else float("nan"),
        "edge_safe": edge_safe(
            float(f10["edge_fidelity"]) if f10 else 0.0,
            float(f05["edge_fidelity"]) if f05 else 0.0,
            float(f1["edge_fidelity"]) if f1 else 0.0,
        ),
    }


def spatial_geom(model, fe, exposure_ms, device, n_in, tile, augs):
    outs = []
    for aug in augs:
        fe_t, h, w = geom_forward(fe, aug)
        dn = _spatial_on_merged(model, fe_t, exposure_ms, device, n_in, tile)
        outs.append(geom_inverse(dn, aug, h, w))
    return np.mean(np.stack(outs, 0), 0).astype(np.float32)


def score_baseline(files, metas, cache_dir) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(edge_path(), None, device)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4)
        edge = spatial_geom(edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4)
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, "p13_d4_flip", out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def train_feat(recipe: Recipe, out_root: Path) -> Path | None:
    out_dir = out_root / f"ckpts_{recipe.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / "best.pt"
    if existing.exists():
        print(f"Reuse existing {existing}", flush=True)
        return existing
    resume = edge_path() if recipe.target == "edge" else SOTA_CKPT
    hard = Path("nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json")
    material = Path(r"D:\denoise\val_raw")
    cmd = [
        sys.executable, "-u", "-m", "nafnet_denoise.train_distill",
        "--arch", "dual_head", "--use-nonlocal", "--nonlocal-count", "1",
        "--align-soft-gate", "--wiener-front-end", "--wiener-merge-frames", "16",
        "--wiener-tile", "32", "--wiener-overlap", "16", "--wiener-c-factor", "8.0",
        "--wiener-spatial", "--wiener-spatial-adaptive",
        "--wiener-fe-schedule", "fps_sigma",
        "--teacher-mode", "partitioned",
        "--wiener-bm3d-teacher", "--wiener-bm3d-sigma", "0.35",
        "--wiener-bm3d-prob", "1.0", "--distill-edge-harden", "16.0",
        "--loss-domain", "dn",
        "--real-manifest", str(hard),
        "--out-dir", str(out_dir),
        "--resume", str(resume), "--fresh-resume",
        "--use-sigma", "--input-frames", "4", "--width", "32",
        "--patch-size", "256", "--micro-batch", "2", "--accum-steps", "4",
        "--epochs", str(recipe.epochs),
        "--patches-per-epoch", str(recipe.patches),
        "--real-fraction", "1.0", "--lr", str(recipe.lr),
        "--highpass-weight", str(recipe.highpass_weight),
        "--des-edge-fid-weight", str(recipe.des_edge_fid),
        "--edge-weight", "0.40", "--grad-weight", "0.40",
        "--ms-edge-weight", str(recipe.ms_edge_weight),
        "--ms-edge-scales", str(recipe.ms_edge_scales),
        "--mixed-grad-xy-weight", str(recipe.mixed_grad_xy),
        "--input-mask-ratio", str(recipe.mask_ratio),
        "--input-mask-patch", "16",
        "--blend-kd-sota", str(SOTA_CKPT),
        "--blend-kd-edgekd", str(EDGE_FALLBACK),
        "--blend-kd-mix", "0.85",
        "--blend-kd-temperature", "8.0",
        "--blend-kd-harden", "16.0",
        "--validation-every", "99",
        "--validation-dir", str(material),
        "--num-workers", "0",
    ]
    print(
        f"FEAT-FT {recipe.name} ms_w={recipe.ms_edge_weight} s={recipe.ms_edge_scales} "
        f"mask={recipe.mask_ratio}",
        flush=True,
    )
    proc = subprocess.run(cmd, cwd=str(Path.cwd()))
    if proc.returncode != 0 or not existing.exists():
        print(f"FAIL train {recipe.name} rc={proc.returncode}", flush=True)
        return None
    return existing


def score_feat(files, metas, cache_dir, recipe: Recipe, ckpt: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(ckpt if recipe.target == "edge" else edge_path(), None, device)
    if recipe.target == "sota":
        sota_m, n_in, _ = load_model(ckpt, None, device)
        edge_m, _, _ = load_model(edge_path(), None, device)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4)
        edge = spatial_geom(edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4)
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, recipe.name, out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout"))
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p15"))
    parser.add_argument("--holdout-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    args = parser.parse_args()

    files = sorted(args.holdout_dir.rglob("*_pMono10_f*.raw"))
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict | None = None
    stagnant = 0
    hist_path = args.output_dir / "history.json"
    if int(args.start) > 0 and hist_path.exists():
        prev = json.loads(hist_path.read_text(encoding="utf-8"))
        history = list(prev.get("history") or [])
        best_safe = prev.get("best_safe")
        stagnant = 0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        print(f"\n===== P15 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        try:
            if recipe.family == "baseline":
                sc = score_baseline(files, metas, args.cache_dir)
                rec = {"iter": idx, "name": recipe.name, "family": recipe.family, "recipe": asdict(recipe), **sc}
            else:
                ckpt = train_feat(recipe, args.output_dir)
                if ckpt is None:
                    stagnant += 1
                    history.append({"iter": idx, "name": recipe.name, "failed": True, "mean_des": 0.0, "edge_safe": False})
                    if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                        break
                    continue
                sc = score_feat(files, metas, args.cache_dir, recipe, ckpt)
                rec = {
                    "iter": idx,
                    "name": recipe.name,
                    "family": recipe.family,
                    "ckpt": str(ckpt),
                    "recipe": asdict(recipe),
                    **sc,
                }
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {recipe.name}: {exc}", flush=True)
            stagnant += 1
            history.append({"iter": idx, "name": recipe.name, "failed": True, "error": str(exc), "mean_des": 0.0, "edge_safe": False})
            if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                break
            continue

        history.append(rec)
        mean_des = sc["mean_des"]
        ok = sc["edge_safe"]
        if ok and (best_safe is None or mean_des > best_safe["mean_des"] + 1e-4):
            best_safe = rec
            stagnant = 0
            (args.output_dir / "global_best.json").write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
            print(f"NEW SAFE BEST {recipe.name} DES={mean_des:.4f}", flush=True)
        else:
            if recipe.family != "baseline":
                stagnant += 1
            print(f"{recipe.name}: DES={mean_des:.4f} ({'ok' if ok else 'unsafe'}) stagnant={stagnant}", flush=True)

        (args.output_dir / "history.json").write_text(
            json.dumps({"history": history, "best_safe": best_safe}, indent=2), encoding="utf-8"
        )
        with (args.output_dir / "metrics_by_recipe.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["iter", "name", "family", "mean_des", "f10_ef", "f05_ef", "f05_ng", "f1_ef", "edge_safe"],
            )
            writer.writeheader()
            for h in history:
                writer.writerow({k: h.get(k) for k in writer.fieldnames})

        if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience) and idx > 0:
            print(f"Early stop: stagnant={stagnant}", flush=True)
            break

    print("--- P15 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4 and best_safe.get("family") == "feat_ft":
        Path("nafnet_denoise/deploy_p15_hook.json").write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
        dest = Path("nafnet_denoise/checkpoints_4f_p15_deploy_edge")
        dest.mkdir(parents=True, exist_ok=True)
        ckpt = Path(best_safe.get("ckpt") or args.output_dir / f"ckpts_{best_safe['name']}" / "best.pt")
        if ckpt.exists():
            shutil.copy2(ckpt, dest / "best.pt")
            dep = Path("nafnet_denoise/infer_blend_deploy.py")
            text = dep.read_text(encoding="utf-8")
            text = text.replace(
                "nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt",
                "nafnet_denoise/checkpoints_4f_p15_deploy_edge/best.pt",
            )
            import re

            text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best_safe['mean_des']:.4f}", text, count=1)
            dep.write_text(text, encoding="utf-8")
            print(f"BAKE {best_safe['name']} → {dest}", flush=True)
    else:
        print(f"No bake (best={None if best_safe is None else best_safe['mean_des']:.4f})", flush=True)


if __name__ == "__main__":
    main()
