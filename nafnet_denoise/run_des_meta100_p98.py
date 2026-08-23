"""P98: GALOSH/RPG-VST lite — Anscombe flat soft-threshold on P74 deploy.

Baseline: P74 DES ≈ 0.9442
Stabilize flats with soft-shrink of HP; leave edges untouched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import anscombe_flat_shrink, deploy_p68_noise, deploy_p98


@dataclass
class Recipe:
    name: str
    family: str
    strength: float = 0.0
    k_mad: float = 1.0
    sigma: float = 1.2
    flat_pct: float = 40.0
    harden: float = 40.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for s in [0.2, 0.35, 0.5, 0.65, 0.8]:
        for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for sig in [0.8, 1.2, 1.6, 2.0]:
                for fp in [30.0, 40.0, 50.0]:
                    recipes.append(
                        Recipe(
                            name=f"ans_s{s:g}_k{k:g}_sig{sig:g}_f{fp:g}",
                            family="anscombe",
                            strength=s,
                            k_mad=k,
                            sigma=sig,
                            flat_pct=fp,
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
        if len(out) >= 100:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    base = deploy_p68_noise(sota, edge, fps)
    if recipe.family == "baseline" or recipe.strength <= 0:
        return base
    return anscombe_flat_shrink(
        base,
        strength=recipe.strength,
        k_mad=recipe.k_mad,
        sigma=recipe.sigma,
        flat_pct=recipe.flat_pct,
        harden=recipe.harden,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p98_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P98 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p98")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P98",
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
