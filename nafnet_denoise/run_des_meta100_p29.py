"""P29: DualEx/Sobel blend hyperparams (T, harden, edge_weight, residual_mix).

Baseline: P22 DES ≈ 0.9290 — keep bilat/unsharp fixed; retune blend gate only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p7_fusion import edge_unsharp
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | blend
    temperature: float = 8.0
    harden: float = 16.0
    edge_weight: float = 1.0
    residual_mix: float = 0.0
    residual_power: float = 1.0
    edge_power: float = 1.0


def deploy_custom(sota, edge, fps, recipe: Recipe) -> np.ndarray:
    low_fps, mid_fps = 1.0, 5.0
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=0.9, harden=40.0)
        u = 0.20
    else:
        out, _ = blend_sota_edgekd(
            sota,
            edge,
            guide_dn=guide,
            temperature=recipe.temperature,
            harden=recipe.harden,
            edge_weight=recipe.edge_weight,
            residual_mix=recipe.residual_mix,
            residual_power=recipe.residual_power,
            edge_power=recipe.edge_power,
        )
        if float(fps) <= mid_fps:
            out = flat_bilateral_boost(out, guide=out, flat_strength=0.85, harden=40.0)
            u = 0.16
        else:
            u = 0.12
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p22", family="baseline")]
    for T in [6.0, 8.0, 10.0, 12.0, 16.0]:
        for h in [8.0, 12.0, 16.0, 24.0, 32.0]:
            recipes.append(
                Recipe(
                    name=f"blend_T{T:g}_h{h:g}",
                    family="blend",
                    temperature=T,
                    harden=h,
                )
            )
    for ew in [0.7, 0.85, 0.95, 1.0, 1.1]:
        recipes.append(
            Recipe(
                name=f"blend_ew{ew:g}",
                family="blend",
                edge_weight=min(ew, 1.0),
            )
        )
    for rm, rp in [
        (0.15, 1.0),
        (0.25, 1.0),
        (0.35, 0.8),
        (0.25, 1.5),
        (0.4, 1.0),
        (0.2, 2.0),
        (0.5, 1.0),
        (0.3, 0.7),
    ]:
        recipes.append(
            Recipe(
                name=f"blend_rm{rm:g}_rp{rp:g}",
                family="blend",
                residual_mix=rm,
                residual_power=rp,
            )
        )
    for ep in [0.7, 0.85, 1.2, 1.5]:
        recipes.append(
            Recipe(name=f"blend_ep{ep:g}", family="blend", edge_power=ep)
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_blend_{i}",
                family="blend",
                temperature=6.0 + (i % 5) * 2.0,
                harden=8.0 + (i % 6) * 4.0,
                residual_mix=(i % 5) * 0.1,
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(
                name=name,
                family=r.family,
                temperature=r.temperature,
                harden=r.harden,
                edge_weight=r.edge_weight,
                residual_mix=r.residual_mix,
                residual_power=r.residual_power,
                edge_power=r.edge_power,
            )
        seen.add(name)
        out.append(r)
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    return deploy_custom(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    recipe = best.get("recipe") or {}
    Path("nafnet_denoise/deploy_p29_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best['mean_des']:.4f}", text, count=1)
    # bake temperature / harden defaults if present in denoise_blend_deploy signature
    if "temperature" in recipe:
        text = re.sub(
            r"edge_temperature: float = [0-9.]+",
            f"edge_temperature: float = {float(recipe['temperature'])}",
            text,
            count=1,
        )
    if "harden" in recipe:
        text = re.sub(
            r"edge_harden: float = [0-9.]+",
            f"edge_harden: float = {float(recipe['harden'])}",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"Baked P29 blend params {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p29")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P29",
        recipes=build_recipes(),
        apply_fn=apply_recipe,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        baseline=float(args.baseline),
        patience=int(args.patience),
        min_eval=int(args.min_eval),
        bake_fn=bake_fn,
    )


if __name__ == "__main__":
    main()
