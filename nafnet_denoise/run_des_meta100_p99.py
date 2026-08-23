"""P99: true weight-space TTT-MIM on flat arm then P74 MAD fuse (ECCV'24).

Baseline: P74 DES ≈ 0.9442
Few GD steps of masked reconstruction on FE; cache adapted sota per hyperparams.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise
from .ttt_mim_weight import ttt_sota_cached


@dataclass
class Recipe:
    name: str
    family: str
    niters: int = 0
    lr: float = 1e-5
    mask_ratio: float = 0.1
    patch: int = 8
    crop: int = 256
    seed: int = 0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for n in [4, 8, 12]:
        for lr in [1e-6, 3e-6, 1e-5, 3e-5]:
            for r in [0.05, 0.1, 0.2]:
                for p in [4, 8, 16]:
                    for c in [192, 256]:
                        recipes.append(
                            Recipe(
                                name=f"ttt_n{n}_lr{lr:g}_r{r:g}_p{p}_c{c}",
                                family="ttt_w",
                                niters=n,
                                lr=lr,
                                mask_ratio=r,
                                patch=p,
                                crop=c,
                            )
                        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(**{**r.__dict__, "name": name})
        seen.add(name)
        out.append(r)
        if len(out) >= 40:  # weight TTT is expensive; smaller grid
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    _, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline" or recipe.niters <= 0:
        sota, edge = load_arms(cache_dir, stem, P19_TAG)
        return deploy_p68_noise(sota, edge, fps)
    sota = ttt_sota_cached(
        cache_dir,
        stem,
        niters=recipe.niters,
        lr=recipe.lr,
        mask_ratio=recipe.mask_ratio,
        patch=recipe.patch,
        crop=recipe.crop,
        seed=recipe.seed,
    )
    return deploy_p68_noise(sota, edge, fps)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p99_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P99 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p99")
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=12)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P99",
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
