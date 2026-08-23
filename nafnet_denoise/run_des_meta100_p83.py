"""P83: ensemble MAD+std noise gates (multi-estimator AdaptiveISP).

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
    w_mad: float = 1.0
    # std arm params (P71-ish)
    std_lo: float = 0.003
    std_hi: float = 0.015
    std_blo: float = 0.70
    std_bhi: float = 0.98


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline", w_mad=1.0)]
    for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for slo, shi in [(0.003, 0.015), (0.002, 0.012), (0.004, 0.018)]:
            for blo, bhi in [(0.70, 0.98), (0.72, 1.0), (0.68, 0.96)]:
                recipes.append(
                    Recipe(
                        name=f"ens_w{w:g}_n{slo:g}_{shi:g}_b{blo:g}",
                        family="ens",
                        w_mad=w,
                        std_lo=slo,
                        std_hi=shi,
                        std_blo=blo,
                        std_bhi=bhi,
                    )
                )
    return recipes[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    mad = deploy_p68_noise(sota, edge, fps)  # defaults = P74 MAD
    if recipe.family == "baseline" or recipe.w_mad >= 0.999:
        return mad
    std = deploy_p68_noise(
        sota,
        edge,
        fps,
        noise_lo=recipe.std_lo,
        noise_hi=recipe.std_hi,
        bilat_lo=recipe.std_blo,
        bilat_hi=recipe.std_bhi,
        noise_mode="std",
        flat_pct=40.0,
    )
    w = float(recipe.w_mad)
    return (w * mad + (1.0 - w) * std).astype(np.float32)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p83_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P83 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p83")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P83",
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
