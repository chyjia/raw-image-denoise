"""P153: wide-window NeighShrink (Chen 5×5 / 7×5 NeighSure).

Baseline: P142 DES ≈ 0.9580
Larger neighborhood energy than P148 default 3×3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p142, deploy_p149


@dataclass
class Recipe:
    name: str
    family: str
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2)
    neigh_win: int = 5
    n_iters: int = 4
    thr_decay: float = 0.7
    sigma: float = 2.0
    residual_scale: float = 0.35


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p142", family="baseline")]
    for win in [5, 7, 9]:
        for ks in [(0.8, 1.0, 1.2), (0.6, 0.9, 1.2), (1.0, 1.2, 1.4)]:
            for nit in [3, 4, 5]:
                for decay in [0.5, 0.7, 0.85]:
                    for sig in [1.8, 2.0, 2.4]:
                        recipes.append(
                            Recipe(
                                name=f"nsw_w{win}_n{nit}_d{decay:g}_sig{sig:g}",
                                family="neighw",
                                k_list=ks,
                                neigh_win=win,
                                n_iters=nit,
                                thr_decay=decay,
                                sigma=sig,
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
        return deploy_p142(sota, edge, fps)
    return deploy_p149(
        sota,
        edge,
        fps,
        k_list=recipe.k_list,
        neigh_win=recipe.neigh_win,
        n_iters=recipe.n_iters,
        thr_decay=recipe.thr_decay,
        ans_sigma=recipe.sigma,
        residual_scale=recipe.residual_scale,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p153_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P149 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p153")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P153",
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
