"""P37: GTTA-style inverse-variance weighted multi-geom ensemble.

Baseline: P31 DES ≈ 0.9292
Lit: GTTA / certainty weighting — downweight high-variance TTA candidates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import (
    BASELINE,
    P19_TAG,
    available_geom_tags,
    load_arm_tag,
    load_arms,
    run_meta_loop,
    stamp_deploy_des,
)
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str
    tags: tuple[str, ...] = (P19_TAG,)
    mode: str = "invvar"  # invvar | invvar_soft | mean
    eps: float = 1e-3
    power: float = 1.0


def invvar_ensemble(stack: np.ndarray, eps: float, power: float) -> np.ndarray:
    # stack: [K,H,W]
    var = np.var(stack, axis=0) + float(eps)
    w = np.power(1.0 / var, float(power))
    w = w / (np.sum(w) + 1e-12)  # wrong — need per-candidate weights
    # Per-candidate: weight = 1 / mean_var_of_that_candidate_vs_mean
    mean = np.mean(stack, axis=0, keepdims=True)
    dist2 = np.mean((stack - mean) ** 2, axis=(1, 2)) + float(eps)  # [K]
    ww = np.power(1.0 / dist2, float(power))
    ww = ww / ww.sum()
    return np.tensordot(ww.astype(np.float32), stack, axes=(0, 0)).astype(np.float32)


def soft_invvar(stack: np.ndarray, eps: float, power: float) -> np.ndarray:
    """Pixel-wise: mix mean with median weighted by local disagreement."""
    mean = np.mean(stack, axis=0)
    med = np.median(stack, axis=0)
    std = np.std(stack, axis=0)
    a = 1.0 / (1.0 + np.power(std / (float(eps) + 1e-6), float(power)))
    return (a * mean + (1.0 - a) * med).astype(np.float32)


def ens_arms(cache_dir, stem, tags, mode, eps, power):
    sotas, edges = [], []
    for tag in tags:
        s = load_arm_tag(cache_dir, stem, "sota", tag)
        e = load_arm_tag(cache_dir, stem, "edge", tag)
        if s is None or e is None:
            continue
        sotas.append(s)
        edges.append(e)
    if not sotas:
        return load_arms(cache_dir, stem)
    S, E = np.stack(sotas, 0), np.stack(edges, 0)
    if mode == "mean":
        return np.mean(S, 0).astype(np.float32), np.mean(E, 0).astype(np.float32)
    if mode == "invvar_soft":
        return soft_invvar(S, eps, power), soft_invvar(E, eps, power)
    return invvar_ensemble(S, eps, power), invvar_ensemble(E, eps, power)


def build_recipes(have: set[str]) -> list[Recipe]:
    preferred = [t for t in [
        P19_TAG,
        "id_r90_r180_r270",
        "id_lr_ud_udlr",
        "id_r90_r180_lr_ud",
        "id_r90_r270_lr_ud_udlr",
        "id_lr_ud_udlr_r90_lr_r270_lr",
        "id_lr_r90_r270_r90_lr_r270_lr",
        "id_ud_r90_r270_r90_lr_r270_lr",
    ] if t in have]
    recipes = [Recipe(name="p31", family="baseline", tags=(P19_TAG,))]
    tag_sets = [
        tuple(preferred[:2]),
        tuple(preferred[:3]),
        tuple(preferred[:4]),
        tuple(preferred[:5]),
        tuple(preferred[:6]),
        (P19_TAG, "id_lr_ud_udlr", "id_r90_r180_r270")
        if all(t in have for t in (P19_TAG, "id_lr_ud_udlr", "id_r90_r180_r270"))
        else tuple(preferred[:3]),
    ]
    for tags in tag_sets:
        if len(tags) < 2:
            continue
        for mode in ("invvar", "invvar_soft", "mean"):
            for power in ([1.0, 1.5, 2.0] if mode != "mean" else [1.0]):
                recipes.append(
                    Recipe(
                        name=f"{mode}_k{len(tags)}_p{power:g}",
                        family="ens",
                        tags=tags,
                        mode=mode,
                        power=power,
                    )
                )
    while len(recipes) < 80:
        i = len(recipes)
        tags = tuple(preferred[: 2 + (i % max(1, len(preferred) - 1))])
        if len(tags) < 2:
            break
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="ens",
                tags=tags,
                mode="invvar_soft" if i % 2 else "invvar",
                power=1.0 + (i % 3) * 0.5,
                eps=1e-3 * (1 + i % 4),
            )
        )
    seen = set()
    out = []
    for r in recipes:
        n = r.name if r.name not in seen else f"{r.name}_x{len(out)}"
        seen.add(n)
        out.append(Recipe(name=n, family=r.family, tags=r.tags, mode=r.mode, eps=r.eps, power=r.power))
    return out[:80]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    if recipe.family == "baseline":
        sota, edge = load_arms(cache_dir, stem)
    else:
        sota, edge = ens_arms(
            cache_dir, stem, recipe.tags, recipe.mode, recipe.eps, recipe.power
        )
    return deploy_p7_base(sota, edge, fps)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p37_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"P37 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout"))
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p37"))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=16)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    have = set(available_geom_tags(args.cache_dir)) | {P19_TAG}
    run_meta_loop(
        cycle="P37",
        recipes=build_recipes(have),
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
