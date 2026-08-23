"""P82: median of MAD/std/p75 noise-gated outputs (EnsIR-style estimator bank).

Baseline: P74 DES ≈ 0.9442
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise


@dataclass
class Recipe:
    name: str
    family: str
    modes: str = "mad"  # comma-separated
    reduce: str = "median"  # median|mean


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    banks = [
        "mad",
        "mad,std",
        "mad,p75",
        "mad,p90",
        "mad,std,p75",
        "mad,std,p90",
        "mad,p75,p90",
        "std,p75,p90",
        "mad,std,p75,p90",
    ]
    for bank in banks:
        for red in ["median", "mean"]:
            recipes.append(
                Recipe(name=f"bank_{bank.replace(',', '-')}_{red}", family="bank", modes=bank, reduce=red)
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="bank",
                modes=banks[i % len(banks)],
                reduce="median" if i % 2 == 0 else "mean",
            )
        )
    return recipes[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p68_noise(sota, edge, fps)
    outs = []
    for mode in recipe.modes.split(","):
        mode = mode.strip()
        kwargs = {"noise_mode": mode}
        if mode == "mad":
            pass  # P74 defaults
        elif mode == "std":
            kwargs.update(
                dict(noise_lo=0.003, noise_hi=0.015, bilat_lo=0.7, bilat_hi=0.98, flat_pct=40.0)
            )
        else:
            kwargs.update(dict(noise_lo=0.002, noise_hi=0.012, bilat_lo=0.72, bilat_hi=1.0, flat_pct=30.0))
        outs.append(deploy_p68_noise(sota, edge, fps, **kwargs))
    stack = np.stack(outs, 0)
    if recipe.reduce == "mean":
        return np.mean(stack, 0).astype(np.float32)
    return np.median(stack, 0).astype(np.float32)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p82_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P82 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p82")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P82",
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
