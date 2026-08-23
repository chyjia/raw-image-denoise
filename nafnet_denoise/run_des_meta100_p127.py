"""P127: interscale Haar shrink after P120 (Luisier Signal Processing'09).

Baseline: P120 DES ≈ 0.9556
Parent wavelet band estimates child soft-threshold on flats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p120, deploy_p127


@dataclass
class Recipe:
    name: str
    family: str
    is_strength: float = 0.15
    is_k_mad: float = 1.0
    is_flat_pct: float = 50.0
    n_iters: int = 2
    thr_decay: float = 0.5
    k_mad: float = 1.2
    sigma: float = 2.8


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p120", family="baseline")]
    for ist in [0.08, 0.12, 0.15, 0.2, 0.25]:
        for ik in [0.75, 1.0, 1.25, 1.5]:
            for fp in [40.0, 50.0, 60.0]:
                for nit in [2, 3]:
                    for decay in [0.5, 0.7]:
                        for k in [1.0, 1.2]:
                            recipes.append(
                                Recipe(
                                    name=f"is_s{ist:g}_k{ik:g}_f{fp:g}_n{nit}_d{decay:g}",
                                    family="interscale",
                                    is_strength=ist,
                                    is_k_mad=ik,
                                    is_flat_pct=fp,
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
        return deploy_p120(sota, edge, fps)
    return deploy_p127(
        sota,
        edge,
        fps,
        is_strength=recipe.is_strength,
        is_k_mad=recipe.is_k_mad,
        is_flat_pct=recipe.is_flat_pct,
        n_iters=recipe.n_iters,
        thr_decay=recipe.thr_decay,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p127_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P127 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p127")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P127",
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
