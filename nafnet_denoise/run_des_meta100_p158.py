"""P158: BiShrink + NeighLevel residual cascade (MAP_NBShrink lite) + SURE-k.

Baseline: P150 DES ≈ 0.9589
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p150, deploy_p158


@dataclass
class Recipe:
    name: str
    family: str
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2)
    n_iters: int = 5
    thr_decay: float = 0.7
    sigma: float = 1.8
    residual_scale: float = 0.3
    neighlevel_alpha: float = 0.5
    neigh_win: int = 3


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p150", family="baseline")]
    grids = [
        (0.8, 1.0, 1.2),
        (0.7, 1.0, 1.3),
    ]
    for ks in grids:
        for alpha in [0.3, 0.5, 0.7]:
            for nw in [3, 5]:
                for nit in [4, 5, 6]:
                    for decay in [0.6, 0.7, 0.85]:
                        for sig in [1.6, 1.8, 2.0]:
                            for rs in [0.28, 0.3, 0.35]:
                                recipes.append(
                                    Recipe(
                                        name=(
                                            f"bn_{'_'.join(f'{k:g}' for k in ks)}"
                                            f"_a{alpha:g}_w{nw}_n{nit}"
                                        ),
                                        family="bineigh",
                                        k_list=ks,
                                        n_iters=nit,
                                        thr_decay=decay,
                                        sigma=sig,
                                        residual_scale=rs,
                                        neighlevel_alpha=alpha,
                                        neigh_win=nw,
                                    )
                                )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        out.append(r)
        if len(out) >= 80:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p150(sota, edge, fps)
    return deploy_p158(
        sota,
        edge,
        fps,
        k_list=recipe.k_list,
        neighlevel_alpha=recipe.neighlevel_alpha,
        neigh_win=recipe.neigh_win,
        n_iters=recipe.n_iters,
        thr_decay=recipe.thr_decay,
        ans_sigma=recipe.sigma,
        residual_scale=recipe.residual_scale,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p158_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P158 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p158")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P158",
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
