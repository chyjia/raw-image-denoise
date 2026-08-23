"""P135: DualDn-lite residual-domain Anscombe after EUI (ECCV'24 DualDn).

Baseline: P126 DES ≈ 0.9565
Fuse intensity-domain EUI with residual-domain flat shrink.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p126, deploy_p135


@dataclass
class Recipe:
    name: str
    family: str
    dual_strength: float = 0.35
    dual_k: float = 1.0
    dual_sigma: float = 2.4
    n_iters: int = 3
    thr_decay: float = 0.7
    k_mad: float = 1.0
    sigma: float = 2.4


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p126", family="baseline")]
    for ds in [0.2, 0.35, 0.5, 0.65]:
        for dk in [0.75, 1.0, 1.25]:
            for dsig in [2.0, 2.4, 2.8]:
                for nit in [2, 3]:
                    for decay in [0.5, 0.7]:
                        for k in [1.0, 1.2]:
                            recipes.append(
                                Recipe(
                                    name=f"dual_s{ds:g}_k{dk:g}_sig{dsig:g}_n{nit}",
                                    family="dual",
                                    dual_strength=ds,
                                    dual_k=dk,
                                    dual_sigma=dsig,
                                    n_iters=nit,
                                    thr_decay=decay,
                                    k_mad=k,
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
    return deploy_p135(
        sota,
        edge,
        fps,
        dual_strength=recipe.dual_strength,
        dual_k=recipe.dual_k,
        dual_sigma=recipe.dual_sigma,
        n_iters=recipe.n_iters,
        thr_decay=recipe.thr_decay,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p135_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P135 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p135")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P135",
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
