"""P79: Neighbor2Neighbor-style half-split fuse of dual arms (zero-train).

Baseline: P40 DES ≈ 0.9294
Lit: Neighbor2Neighbor / TTAD self-similarity — split views, fuse for consistency.
Here: checkerboard / stripe mix of sota↔edge then P40 deploy schedule on fused.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | n2n
    mode: str = "checker"  # checker | hstripe | vstripe | avg_split
    period: int = 2
    # mix fused with original soft path
    alpha: float = 1.0


def half_split_fuse(sota, edge, mode: str, period: int) -> np.ndarray:
    s = np.ascontiguousarray(sota, dtype=np.float32)
    e = np.ascontiguousarray(edge, dtype=np.float32)
    h, w = s.shape
    yy, xx = np.ogrid[:h, :w]
    p = max(int(period), 1)
    if mode == "checker":
        mask = ((yy // p + xx // p) % 2 == 0)
    elif mode == "hstripe":
        mask = ((yy // p) % 2 == 0)
    elif mode == "vstripe":
        mask = ((xx // p) % 2 == 0)
    else:  # avg_split: average two complementary checkerboards
        m1 = ((yy // p + xx // p) % 2 == 0)
        a = np.where(m1, s, e)
        b = np.where(m1, e, s)
        return (0.5 * a + 0.5 * b).astype(np.float32)
    return np.where(mask, s, e).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for mode in ["checker", "hstripe", "vstripe", "avg_split"]:
        for period in [1, 2, 4, 8]:
            for alpha in [0.5, 0.75, 1.0]:
                recipes.append(
                    Recipe(
                        name=f"n2n_{mode}_p{period}_a{alpha:g}",
                        family="n2n",
                        mode=mode,
                        period=period,
                        alpha=alpha,
                    )
                )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="n2n",
                mode=["checker", "hstripe", "vstripe"][i % 3],
                period=1 + (i % 5),
                alpha=0.4 + (i % 4) * 0.2,
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
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    fused = half_split_fuse(sota, edge, recipe.mode, recipe.period)
    # treat fused as both arms for schedule, blend with standard deploy
    out_f = deploy_p7_base(fused, fused, fps)
    out_b = deploy_p7_base(sota, edge, fps)
    a = float(np.clip(recipe.alpha, 0.0, 1.0))
    return ((1.0 - a) * out_b + a * out_f).astype(np.float32)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p79_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p79/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n{best.get('recipe')}\n",
        encoding="utf-8",
    )
    print(f"Wrote P79 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p79")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P79",
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
