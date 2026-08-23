"""P132: RPG-VST σz gate algebraic vs EUI recursive spin.

Baseline: P126 DES ≈ 0.9565
Pick branch whose stabilized-domain flat MAD is closer to target.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p126, deploy_p132


@dataclass
class Recipe:
    name: str
    family: str
    target_sz: float = 0.02
    n_iters: int = 3
    thr_decay: float = 0.7
    k_mad: float = 1.0
    sigma: float = 2.4
    residual_scale: float = 0.4


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p126", family="baseline")]
    for tgt in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]:
        for nit in [2, 3, 4]:
            for decay in [0.5, 0.7]:
                for k in [0.9, 1.0, 1.2]:
                    for sig in [2.0, 2.4, 2.8]:
                        recipes.append(
                            Recipe(
                                name=f"sz_t{tgt:g}_n{nit}_d{decay:g}_k{k:g}_sig{sig:g}",
                                family="szgate",
                                target_sz=tgt,
                                n_iters=nit,
                                thr_decay=decay,
                                k_mad=k,
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
        if len(out) >= 100:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p126(sota, edge, fps)
    return deploy_p132(
        sota,
        edge,
        fps,
        target_sz=recipe.target_sz,
        n_iters=recipe.n_iters,
        thr_decay=recipe.thr_decay,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
        residual_scale=recipe.residual_scale,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p132_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P132 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p132")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P132",
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
