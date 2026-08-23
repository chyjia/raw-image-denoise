"""P21 meta: refine P20 fps schedule + mild clip-unsharp / MS-unsharp on P19 arms.

Baseline: P20 DES ≈ 0.9271 (bilat 0.8/0.75, unsharp 0.18/0.15/0.12)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .p8_fusion import clipped_unsharp, deploy_p7_base, multi_scale_unsharp
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image
from .run_des_meta100_p20 import deploy_sched, load_arms, Recipe as SchedRecipe

BASELINE = 0.9271


@dataclass
class Recipe:
    name: str
    family: str  # baseline | sched | clip | msu
    u_low: float = 0.18
    u_mid: float = 0.15
    u_high: float = 0.12
    low_bilat: float = 0.8
    mid_bilat: float = 0.75
    u_sig: float = 1.4
    clip_pct: float = 98.0
    amounts: tuple[float, ...] = (0.1, 0.08, 0.05)
    sigmas: tuple[float, ...] = (0.8, 1.4, 2.5)


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p20", family="baseline")]
    for ul, um, uh in [
        (0.18, 0.15, 0.12),
        (0.20, 0.15, 0.12),
        (0.18, 0.16, 0.12),
        (0.18, 0.15, 0.10),
        (0.16, 0.15, 0.12),
        (0.19, 0.15, 0.11),
        (0.18, 0.14, 0.12),
        (0.17, 0.15, 0.13),
        (0.22, 0.15, 0.10),
        (0.18, 0.18, 0.12),
        (0.15, 0.15, 0.12),
        (0.20, 0.16, 0.12),
    ]:
        for lb, mb in [(0.8, 0.75), (0.85, 0.75), (0.8, 0.8), (0.85, 0.8), (0.75, 0.7)]:
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
    for amt, pct in [
        (0.18, 98.0),
        (0.15, 97.0),
        (0.12, 99.0),
        (0.16, 98.0),
        (0.14, 96.0),
        (0.18, 95.0),
    ]:
        recipes.append(
            Recipe(name=f"clip_a{amt:g}_p{pct:g}", family="clip", u_low=amt, clip_pct=pct)
        )
    for amts in [
        (0.06, 0.05, 0.03),
        (0.08, 0.05, 0.03),
        (0.05, 0.05, 0.04),
        (0.1, 0.04, 0.02),
    ]:
        recipes.append(
            Recipe(name=f"msu_{'_'.join(f'{a:g}' for a in amts)}", family="msu", amounts=amts)
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="sched",
                u_low=0.16 + (i % 5) * 0.01,
                u_mid=0.14 + (i % 4) * 0.01,
                u_high=0.10 + (i % 3) * 0.01,
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


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float) -> np.ndarray:
    sota, edge = load_arms(cache_dir, stem)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    if recipe.family == "sched":
        sr = SchedRecipe(
            name=recipe.name,
            family="sched",
            u_low=recipe.u_low,
            u_mid=recipe.u_mid,
            u_high=recipe.u_high,
            low_bilat=recipe.low_bilat,
            mid_bilat=recipe.mid_bilat,
            u_sig=recipe.u_sig,
        )
        return deploy_sched(sota, edge, fps, sr)
    if recipe.family == "clip":
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0,
                              unsharp_amount_low=0.0, unsharp_amount_mid=0.0,
                              unsharp_amount_high=0.0)
        return clipped_unsharp(base, amount=recipe.u_low, sigma=recipe.u_sig,
                               clip_pct=recipe.clip_pct)
    if recipe.family == "msu":
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0,
                              unsharp_amount_low=0.0, unsharp_amount_mid=0.0,
                              unsharp_amount_high=0.0)
        return multi_scale_unsharp(base, amounts=recipe.amounts, sigmas=recipe.sigmas)
    return deploy_p7_base(sota, edge, fps)


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
    Path("nafnet_denoise/deploy_p21_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best['mean_des']:.4f}", text, count=1)
    if recipe.get("family") == "sched":
        text = re.sub(
            r"unsharp_amount_low: float = [0-9.]+",
            f"unsharp_amount_low: float = {float(recipe['u_low'])}",
            text,
            count=1,
        )
        text = re.sub(
            r"unsharp_amount_mid: float = [0-9.]+",
            f"unsharp_amount_mid: float = {float(recipe['u_mid'])}",
            text,
            count=1,
        )
        text = re.sub(
            r"unsharp_amount_high: float = [0-9.]+",
            f"unsharp_amount_high: float = {float(recipe['u_high'])}",
            text,
            count=1,
        )
        text = re.sub(
            r"bilateral_strength: float = [0-9.]+",
            f"bilateral_strength: float = {float(recipe['low_bilat'])}",
            text,
            count=1,
        )
        text = re.sub(
            r"mid_bilateral_strength: float = [0-9.]+",
            f"mid_bilateral_strength: float = {float(recipe['mid_bilat'])}",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"Baked P21 {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p21")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
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
        print(f"\n===== P21 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
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

    print("--- P21 done ---", flush=True)
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
