"""P13 meta: pad-D4 TTA + model soup + Young/FitNet multi-scale edge FT.

Baseline: P12 deploy (mask_0 edge) mean DES ≈ 0.9240
Lit leftovers (after mask-FT / flip-TTA / bilat / unsharp dead-ends):
  1) pad-to-square 90° rotations (D4 geometric self-ensemble)
  2) output / weight soup of edge-safe mask ckpts
  3) multi-scale Sobel feature match FT (Young Feature Matching / FitNet-style)
  4) mild shift ensembles on P12 deploy stack

Defaults: early-stop when stagnant; no mid-loop questions.
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

BASELINE = 0.9240
TTA4 = ("id", "lr", "ud", "udlr")
SOTA_CKPT = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
EDGE_CKPT = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")
EDGE_FALLBACK = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")


@dataclass
class Recipe:
    name: str
    family: str  # baseline | d4 | soup | shift | feat_ft
    augs: tuple[str, ...] = TTA4
    shifts: tuple[tuple[int, int], ...] = ()
    soup_ckpts: tuple[str, ...] = ()
    soup_mode: str = "output"  # output | weight
    target: str = "edge"
    epochs: int = 4
    patches: int = 128
    lr: float = 1.5e-6
    ms_edge_weight: float = 0.35
    ms_edge_scales: int = 3
    highpass_weight: float = 0.45
    des_edge_fid: float = 1.4
    mask_ratio: float = 0.0


def edge_path() -> Path:
    return EDGE_CKPT if EDGE_CKPT.exists() else EDGE_FALLBACK


def build_100_recipes() -> list[Recipe]:
    recipes: list[Recipe] = [
        Recipe(name="p12_deploy", family="baseline"),
    ]

    # --- pad-D4 geometric (zero-train) ---
    d4_sets = [
        ("d4_r90", ("id", "lr", "ud", "udlr", "r90", "r270")),
        ("d4_full", ("id", "lr", "ud", "udlr", "r90", "r180", "r270")),
        ("d4_rot_only", ("id", "r90", "r180", "r270")),
        ("d4_r90_flip", ("id", "r90", "r270", "r90_lr", "r270_lr")),
        ("d4_min", ("id", "r90", "r270")),
        ("d4_edge_r90", ("id", "lr", "ud", "udlr", "r90")),
        ("d4_sota_r90", ("id", "lr", "ud", "udlr", "r90")),
    ]
    for name, augs in d4_sets:
        recipes.append(Recipe(name=name, family="d4", augs=augs))

    # more d4 variants
    for extra in [
        ("r90", "r270"),
        ("r90", "r180", "r270"),
        ("r90_lr", "r270_lr"),
        ("r90", "r270", "r180_lr"),
        ("r90", "ud", "lr"),
    ]:
        augs = tuple(dict.fromkeys(("id", "lr", "ud", "udlr") + extra))
        recipes.append(
            Recipe(
                name=f"d4_{'_'.join(extra)}",
                family="d4",
                augs=augs,
            )
        )

    # --- model soup (output avg of edge-safe mask ckpts) ---
    soup_roots = Path("nafnet_denoise/compare_meta100_p12")
    candidates = []
    for p in sorted(soup_roots.glob("ckpts_mask_*/best.pt")):
        candidates.append(str(p))
    # Prefer known good + near-ties
    preferred = [
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_0_r0.25_p8_edge/best.pt",
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_8_r0.35_p16_edge/best.pt",
        str(EDGE_CKPT),
        str(EDGE_FALLBACK),
    ]
    preferred = [p for p in preferred if Path(p).exists()]
    for i, pair in enumerate(
        [
            preferred[:2],
            preferred[:3],
            preferred,
            candidates[:2],
            candidates[:3],
            candidates[0:1] + candidates[2:3] if len(candidates) >= 3 else preferred[:2],
        ]
    ):
        pair = tuple(dict.fromkeys(p for p in pair if Path(p).exists()))
        if len(pair) < 2:
            continue
        recipes.append(
            Recipe(
                name=f"soup_{i}_n{len(pair)}",
                family="soup",
                soup_ckpts=pair,
                soup_mode="output",
            )
        )

    # --- shifts on P12 stack ---
    for shifts in [
        ((0, 0), (8, 0)),
        ((0, 0), (0, 8)),
        ((0, 0), (8, 8)),
        ((0, 0), (-8, 0), (8, 0)),
        ((0, 0), (0, -8), (0, 8)),
        ((0, 0), (8, 0), (0, 8), (8, 8)),
        ((0, 0), (4, 4), (-4, -4)),
        ((0, 0), (16, 0), (-16, 0)),
    ]:
        tag = "_".join(f"{dy}x{dx}" for dy, dx in shifts)
        recipes.append(Recipe(name=f"shift_{tag}", family="shift", shifts=shifts))

    # --- feature-match FT ---
    for i, (w, sc, lr, ep, mr) in enumerate(
        [
            (0.25, 3, 1.5e-6, 4, 0.0),
            (0.35, 3, 1.5e-6, 4, 0.0),
            (0.50, 3, 1.2e-6, 4, 0.0),
            (0.35, 4, 1.5e-6, 5, 0.0),
            (0.40, 3, 1e-6, 6, 0.0),
            (0.35, 3, 1.5e-6, 4, 0.25),  # + mild mask
            (0.45, 3, 2e-6, 4, 0.0),
            (0.30, 2, 1.5e-6, 4, 0.0),
            (0.55, 3, 1e-6, 5, 0.0),
            (0.35, 3, 8e-7, 8, 0.0),
            (0.40, 4, 1.5e-6, 4, 0.35),
            (0.25, 3, 2e-6, 3, 0.0),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"feat_{i}_w{w:g}_s{sc}",
                family="feat_ft",
                target="edge",
                ms_edge_weight=w,
                ms_edge_scales=sc,
                lr=lr,
                epochs=ep,
                mask_ratio=mr,
            )
        )

    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_feat_{i}",
                family="feat_ft",
                ms_edge_weight=0.2 + 0.03 * (i % 10),
                ms_edge_scales=2 + (i % 3),
                lr=1e-6 * (1.0 + 0.2 * (i % 5)),
                epochs=3 + (i % 4),
            )
        )
    return recipes[:100]


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


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


def spatial_geom(
    model, fe, exposure_ms, device, n_in, tile, augs: tuple[str, ...]
) -> np.ndarray:
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
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, TTA4
        )
        edge = spatial_geom(
            edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, TTA4
        )
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, "p12_deploy", out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def score_d4(files, metas, cache_dir, recipe: Recipe) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(edge_path(), None, device)
    # Optional: d4 only on one arm
    sota_augs = recipe.augs
    edge_augs = recipe.augs
    if recipe.name.endswith("_sota_r90") or "sota_r90" in recipe.name:
        edge_augs = TTA4
    if recipe.name.endswith("_edge_r90") or recipe.name == "d4_edge_r90":
        sota_augs = TTA4
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, sota_augs
        )
        edge = spatial_geom(
            edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, edge_augs
        )
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


def score_soup(files, metas, cache_dir, recipe: Recipe) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_models = []
    for ck in recipe.soup_ckpts:
        m, _, _ = load_model(Path(ck), None, device)
        edge_models.append(m)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, TTA4
        )
        edges = [
            spatial_geom(m, fe, float(meta["exposure_ms"]), device, n_in, 256, TTA4)
            for m in edge_models
        ]
        edge = np.mean(np.stack(edges, 0), 0).astype(np.float32)
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


def score_shift(files, metas, cache_dir, recipe: Recipe) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(edge_path(), None, device)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        outs = []
        for dy, dx in recipe.shifts:
            fe_s = shift_image(fe, dy, dx)
            sota = spatial_geom(
                sota_m, fe_s, float(meta["exposure_ms"]), device, n_in, 256, TTA4
            )
            edge = spatial_geom(
                edge_m, fe_s, float(meta["exposure_ms"]), device, n_in, 256, TTA4
            )
            out = deploy_p7_base(sota, edge, float(meta["fps"]))
            if dy or dx:
                out = shift_image(out, -dy, -dx)
            outs.append(out)
        final = np.mean(np.stack(outs, 0), 0).astype(np.float32)
        sc = score_image(meta, cache_dir, recipe.name, final)
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
    resume = edge_path()
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
    print(f"FEAT-FT {recipe.name} ms_w={recipe.ms_edge_weight} s={recipe.ms_edge_scales}", flush=True)
    proc = subprocess.run(cmd, cwd=str(Path.cwd()))
    if proc.returncode != 0 or not existing.exists():
        print(f"FAIL train {recipe.name} rc={proc.returncode}", flush=True)
        return None
    return existing


def score_feat(files, metas, cache_dir, recipe: Recipe, ckpt: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(ckpt, None, device)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, TTA4
        )
        edge = spatial_geom(
            edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, TTA4
        )
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
    parser.add_argument("--min-eval", type=int, default=15)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("nafnet_denoise/cache_dual_holdout"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_meta100_p13"),
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path(r"D:\denoise\val_raw"),
    )
    args = parser.parse_args()

    files = sorted(args.holdout_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.holdout_dir}")
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    for p in files:
        if not (args.cache_dir / f"{p.stem}_fe.npy").exists():
            raise SystemExit(f"Missing FE cache for {p.stem}")

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
        print(
            f"Resuming: {len(history)} rows, best="
            f"{None if best_safe is None else best_safe.get('mean_des')}",
            flush=True,
        )

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        print(f"\n===== P13 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        try:
            if recipe.family == "baseline":
                sc = score_baseline(files, metas, args.cache_dir)
            elif recipe.family == "d4":
                sc = score_d4(files, metas, args.cache_dir, recipe)
            elif recipe.family == "soup":
                sc = score_soup(files, metas, args.cache_dir, recipe)
            elif recipe.family == "shift":
                sc = score_shift(files, metas, args.cache_dir, recipe)
            elif recipe.family == "feat_ft":
                ckpt = train_feat(recipe, args.output_dir)
                if ckpt is None:
                    stagnant += 1
                    history.append(
                        {
                            "iter": idx,
                            "name": recipe.name,
                            "failed": True,
                            "mean_des": 0.0,
                            "edge_safe": False,
                        }
                    )
                    if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                        break
                    continue
                sc = score_feat(files, metas, args.cache_dir, recipe, ckpt)
            else:
                raise ValueError(recipe.family)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {recipe.name}: {exc}", flush=True)
            stagnant += 1
            history.append(
                {
                    "iter": idx,
                    "name": recipe.name,
                    "failed": True,
                    "error": str(exc),
                    "mean_des": 0.0,
                    "edge_safe": False,
                }
            )
            if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                break
            continue

        rec = {
            "iter": idx,
            "name": recipe.name,
            "family": recipe.family,
            "recipe": asdict(recipe),
            **sc,
        }
        history.append(rec)
        mean_des = sc["mean_des"]
        ok = sc["edge_safe"]
        if ok and (best_safe is None or mean_des > best_safe["mean_des"] + 1e-4):
            best_safe = rec
            stagnant = 0
            (args.output_dir / "global_best.json").write_text(
                json.dumps(best_safe, indent=2), encoding="utf-8"
            )
            print(f"NEW SAFE BEST {recipe.name} DES={mean_des:.4f}", flush=True)
        else:
            if recipe.family != "baseline":
                stagnant += 1
            print(
                f"{recipe.name}: DES={mean_des:.4f} "
                f"({'ok' if ok else 'unsafe'}) stagnant={stagnant}",
                flush=True,
            )

        (args.output_dir / "history.json").write_text(
            json.dumps({"history": history, "best_safe": best_safe}, indent=2),
            encoding="utf-8",
        )
        with (args.output_dir / "metrics_by_recipe.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iter",
                    "name",
                    "family",
                    "mean_des",
                    "f10_ef",
                    "f05_ef",
                    "f05_ng",
                    "f1_ef",
                    "edge_safe",
                ],
            )
            writer.writeheader()
            for h in history:
                writer.writerow({k: h.get(k) for k in writer.fieldnames})

        if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience) and idx > 0:
            print(
                f"Early stop: stagnant={stagnant} best="
                f"{None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P13 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        (args.output_dir / "bake_recipe.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        Path("nafnet_denoise/deploy_p13_hook.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        if best_safe.get("family") == "feat_ft":
            dest = Path("nafnet_denoise/checkpoints_4f_p13_deploy_edge")
            dest.mkdir(parents=True, exist_ok=True)
            ckpt = args.output_dir / f"ckpts_{best_safe['name']}" / "best.pt"
            if ckpt.exists():
                shutil.copy2(ckpt, dest / "best.pt")
                dep = Path("nafnet_denoise/infer_blend_deploy.py")
                text = dep.read_text(encoding="utf-8")
                text = text.replace(
                    "nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt",
                    "nafnet_denoise/checkpoints_4f_p13_deploy_edge/best.pt",
                )
                import re

                text = re.sub(
                    r"mean DES ~0\.\d+",
                    f"mean DES ~{best_safe['mean_des']:.4f}",
                    text,
                    count=1,
                )
                dep.write_text(text, encoding="utf-8")
                print(f"Patched deploy → {dest}", flush=True)
        elif best_safe.get("family") in ("d4", "soup", "shift"):
            # Inference-only bake: record hook; deploy code may read augs from hook.
            print(
                f"Inference bake recorded in deploy_p13_hook.json "
                f"(family={best_safe['family']})",
                flush=True,
            )
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
