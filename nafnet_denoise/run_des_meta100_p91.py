"""P91: TTAD pixel-bank flat HF pull on P74 deploy (ICCV'25).

Baseline: P74 DES ≈ 0.9442
Lit: TTAD self-similarity pixel bank — coarse patch-similarity HF denoise on flats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise
from .ttad_lite import ttad_flat_hf_pull


@dataclass
class Recipe:
    name: str
    family: str
    strength: float = 0.0
    patch: int = 5
    search: int = 7
    topk: int = 8
    scale: float = 0.25
    stride: int = 2


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for s in [0.15, 0.25, 0.35, 0.5, 0.65]:
        for p in [3, 5, 7]:
            for sr in [5, 7, 9]:
                for tk in [4, 8, 12]:
                    for sc in [0.2, 0.25, 0.35]:
                        recipes.append(
                            Recipe(
                                name=f"ttad_s{s:g}_p{p}_sr{sr}_k{tk}_sc{sc:g}",
                                family="ttad",
                                strength=s,
                                patch=p,
                                search=sr,
                                topk=tk,
                                scale=sc,
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
    guide = (0.5 * sota + 0.5 * edge).astype("float32")
    return ttad_flat_hf_pull(
        base,
        guide,
        strength=recipe.strength,
        patch=recipe.patch,
        search=recipe.search,
        topk=recipe.topk,
        scale=recipe.scale,
        stride=recipe.stride,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p91_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P91 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p91")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P91",
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
