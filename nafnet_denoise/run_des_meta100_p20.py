"""P20 meta: fps/noise-adaptive unsharp + bilat schedule on D4 deploy.

Baseline: P13 d4_r90 DES ≈ 0.9252
Lit leftovers: exposure-aware ISP; schedule post by fps (already Gen2) —
  refine unsharp/bilat strengths per fps band without new forwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p7_fusion import edge_unsharp
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9253


@dataclass
class Recipe:
    name: str
    family: str  # baseline | sched
    low_fps: float = 1.0
    mid_fps: float = 5.0
    low_bilat: float = 0.75
    mid_bilat: float = 0.7
    bilat_harden: float = 40.0
    u_low: float = 0.15  # fps<=low
    u_mid: float = 0.15
    u_high: float = 0.15
    u_sig: float = 1.4
    blend_T: float = 8.0
    blend_harden: float = 16.0


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4", family="baseline")]
    grids = []
    for ul, um, uh in [
        (0.12, 0.15, 0.18),
        (0.10, 0.15, 0.20),
        (0.15, 0.15, 0.12),
        (0.18, 0.15, 0.12),
        (0.12, 0.12, 0.15),
        (0.15, 0.18, 0.15),
        (0.08, 0.15, 0.18),
        (0.15, 0.10, 0.15),
        (0.20, 0.15, 0.10),
        (0.14, 0.16, 0.14),
        (0.16, 0.14, 0.16),
        (0.12, 0.18, 0.12),
    ]:
        for lb, mb in [(0.75, 0.7), (0.8, 0.75), (0.7, 0.65), (0.75, 0.65), (0.85, 0.7)]:
            grids.append((ul, um, uh, lb, mb))
    for i, (ul, um, uh, lb, mb) in enumerate(grids[:80]):
        recipes.append(
            Recipe(
                name=f"sch_u{ul:g}_{um:g}_{uh:g}_b{lb:g}_{mb:g}",
                family="sched",
                u_low=ul,
                u_mid=um,
                u_high=uh,
                low_bilat=lb,
                mid_bilat=mb,
            )
        )
    for sig in [1.2, 1.4, 1.6, 1.8]:
        recipes.append(
            Recipe(
                name=f"sch_sig{sig:g}",
                family="sched",
                u_sig=sig,
                u_low=0.15,
                u_mid=0.15,
                u_high=0.15,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_sch_{i}",
                family="sched",
                u_low=0.1 + (i % 5) * 0.02,
                u_mid=0.12 + (i % 4) * 0.02,
                u_high=0.14 + (i % 3) * 0.02,
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        out.append(r)
    return out[:100]


def deploy_sched(sota, edge, fps, recipe: Recipe) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= recipe.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(
            out, guide=out, flat_strength=recipe.low_bilat, harden=recipe.bilat_harden
        )
        u = recipe.u_low
    elif float(fps) <= recipe.mid_fps:
        out, _ = blend_sota_edgekd(
            sota,
            edge,
            guide_dn=guide,
            temperature=recipe.blend_T,
            harden=recipe.blend_harden,
            edge_weight=1.0,
        )
        out = flat_bilateral_boost(
            out, guide=out, flat_strength=recipe.mid_bilat, harden=recipe.bilat_harden
        )
        u = recipe.u_mid
    else:
        out, _ = blend_sota_edgekd(
            sota,
            edge,
            guide_dn=guide,
            temperature=recipe.blend_T,
            harden=recipe.blend_harden,
            edge_weight=1.0,
        )
        u = recipe.u_high
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=recipe.u_sig, harden=16.0)
    return out.astype(np.float32)


P19_TAG = "id_r90_r180_r270_r90_lr_r270_lr"


def load_arms(cache_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Prefer P19 d4+r180 caches; fall back to P16 D4."""
    sp = cache_dir / f"{stem}_sota_g_{P19_TAG}.npy"
    ep = cache_dir / f"{stem}_edge_g_{P19_TAG}.npy"
    if sp.exists() and ep.exists():
        return np.load(sp), np.load(ep)
    return (
        np.load(cache_dir / f"{stem}_sota_d4.npy"),
        np.load(cache_dir / f"{stem}_edge_d4.npy"),
    )


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float) -> np.ndarray:
    sota, edge = load_arms(cache_dir, stem)
    if recipe.family == "baseline":
        from .p8_fusion import deploy_p7_base

        return deploy_p7_base(sota, edge, fps)
    return deploy_sched(sota, edge, fps, recipe)


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


def bake_deploy(best: dict) -> None:
    recipe = best.get("recipe") or {}
    hook = Path("nafnet_denoise/deploy_p20_hook.json")
    hook.write_text(json.dumps(best, indent=2), encoding="utf-8")
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"mean DES ~0\.\d+",
        f"mean DES ~{best['mean_des']:.4f}",
        text,
        count=1,
    )
    # Bake constant unsharp if uniform; else leave hook for fps-varying
    if (
        abs(float(recipe.get("u_low", 0.15)) - float(recipe.get("u_mid", 0.15))) < 1e-6
        and abs(float(recipe.get("u_mid", 0.15)) - float(recipe.get("u_high", 0.15)))
        < 1e-6
    ):
        text = re.sub(
            r"unsharp_amount: float = [0-9.]+",
            f"unsharp_amount: float = {float(recipe['u_low'])}",
            text,
            count=1,
        )
    text = re.sub(
        r"bilateral_strength: float = [0-9.]+",
        f"bilateral_strength: float = {float(recipe.get('low_bilat', 0.75))}",
        text,
        count=1,
    )
    text = re.sub(
        r"mid_bilateral_strength: float = [0-9.]+",
        f"mid_bilateral_strength: float = {float(recipe.get('mid_bilat', 0.7))}",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    print(f"Baked schedule into deploy ({best['name']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p20")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict = {
        "iter": -1,
        "name": "baseline_seed",
        "family": "baseline",
        "mean_des": BASELINE,
        "edge_safe": True,
    }
    stagnant = 0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        print(f"\n===== P20 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        des_list, f10, f05, f1 = [], None, None, None
        for path, meta in zip(files, metas):
            out = apply_recipe(recipe, args.cache_dir, path.stem, float(meta["fps"]))
            sc = score_image(meta, args.cache_dir, recipe.name, out)
            des_list.append(sc["des"])
            if F10 in path.name:
                f10 = sc
            if F05 in path.name:
                f05 = sc
            if F1 in path.name:
                f1 = sc
        sc = summarize(des_list, f10, f05, f1)
        rec = {
            "iter": idx,
            "name": recipe.name,
            "family": recipe.family,
            "recipe": asdict(recipe),
            **sc,
        }
        history.append(rec)
        if sc["edge_safe"] and sc["mean_des"] > best_safe["mean_des"] + 1e-4:
            best_safe = rec
            stagnant = 0
            (args.output_dir / "global_best.json").write_text(
                json.dumps(best_safe, indent=2), encoding="utf-8"
            )
            print(f"NEW SAFE BEST {recipe.name} DES={sc['mean_des']:.4f}", flush=True)
        else:
            if recipe.family != "baseline":
                stagnant += 1
            print(
                f"{recipe.name}: DES={sc['mean_des']:.4f} "
                f"({'ok' if sc['edge_safe'] else 'unsafe'}) stagnant={stagnant}",
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
                f"Early stop: stagnant={stagnant} best={best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P20 done ---", flush=True)
    if best_safe["mean_des"] >= BASELINE + 1e-4 and best_safe.get("name") != "baseline_seed":
        (args.output_dir / "bake_recipe.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        bake_deploy(best_safe)
    else:
        print(f"No bake (best={best_safe['mean_des']:.4f})", flush=True)


if __name__ == "__main__":
    main()
