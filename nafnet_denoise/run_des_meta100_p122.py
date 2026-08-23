"""P122: multi-k Anscombe soft-median ensemble (NTIRE self-ensemble lite).

Baseline: P114 DES ≈ 0.9528
Run P114 with several k_mad, median-fuse on flats only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .multiband_fuse import _edge_flat_maps
from .p8_fusion import deploy_p114


@dataclass
class Recipe:
    name: str
    family: str
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2)
    flat_pct: float = 50.0
    residual_scale: float = 0.4
    sigma: float = 2.8
    max_shift: int = 1


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p114", family="baseline")]
    grids = [
        (0.8, 1.0, 1.2),
        (0.9, 1.0, 1.1),
        (0.7, 1.0, 1.3),
        (0.85, 1.0, 1.15, 1.3),
        (0.75, 0.9, 1.05, 1.2),
        (1.0, 1.1, 1.2),
    ]
    for ks in grids:
        for rs in [0.35, 0.4, 0.45]:
            for sig in [2.4, 2.8, 3.2]:
                for fp in [40.0, 50.0, 60.0]:
                    recipes.append(
                        Recipe(
                            name=f"mk_{'_'.join(f'{k:g}' for k in ks)}_r{rs:g}_sig{sig:g}_f{fp:g}",
                            family="multik",
                            k_list=ks,
                            residual_scale=rs,
                            sigma=sig,
                            flat_pct=fp,
                        )
                    )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        out.append(r)
        if len(out) >= 80:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p114(sota, edge, fps)
    outs = [
        deploy_p114(
            sota,
            edge,
            fps,
            max_shift=recipe.max_shift,
            residual_scale=recipe.residual_scale,
            ans_k_mad=k,
            ans_sigma=recipe.sigma,
        )
        for k in recipe.k_list
    ]
    stack = np.stack(outs, 0)
    med = np.median(stack, 0).astype(np.float32)
    base = outs[len(outs) // 2]
    _, flat = _edge_flat_maps(base, temperature=8.0, harden=40.0)
    if recipe.flat_pct > 0:
        sob = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            base, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = ((flat > 0.5) & (sob < np.percentile(sob, recipe.flat_pct))).astype(
            np.float32
        )
    return (base * (1.0 - flat) + med * flat).astype(np.float32)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p122_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P122 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p122")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P122",
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
