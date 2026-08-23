"""P72: mix P68 noise-gate with P64 exposure arm (AdaptiveISP dual-cue).

Baseline: P68 DES ≈ 0.9438
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p64_exposure, deploy_p68_noise


@dataclass
class Recipe:
    name: str
    family: str
    w_exp: float = 0.0
    low_fps: float = 1.67


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p68", family="baseline")]
    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for lf in [1.5, 1.67, 2.0]:
            recipes.append(
                Recipe(name=f"mix_e{w:g}_f{lf:g}", family="mix", w_exp=w, low_fps=lf)
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="mix",
                w_exp=0.05 + (i % 10) * 0.05,
                low_fps=1.5 + (i % 3) * 0.25,
            )
        )
    return recipes[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    noise = deploy_p68_noise(sota, edge, fps, low_fps=recipe.low_fps)
    if recipe.family == "baseline" or recipe.w_exp <= 0:
        return noise
    exp = deploy_p64_exposure(
        sota, edge, fps, exposure_of(cache_dir, stem), low_fps=recipe.low_fps
    )
    w = float(recipe.w_exp)
    return ((1.0 - w) * noise + w * exp).astype(np.float32)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p72_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P72 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p72")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P72",
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
