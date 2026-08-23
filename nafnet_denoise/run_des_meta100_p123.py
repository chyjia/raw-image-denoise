"""P123: fps-gated residual_scale on P114 (AdaptiveISP fps × CNE).

Baseline: P114 DES ≈ 0.9528
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p114


@dataclass
class Recipe:
    name: str
    family: str
    low_fps: float = 1.5
    mid_fps: float = 5.0
    rs_low: float = 0.45
    rs_mid: float = 0.4
    rs_high: float = 0.35
    k_mad: float = 1.0
    sigma: float = 2.8
    s_hi: float = 0.4


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p114", family="baseline")]
    for low in [1.0, 1.5, 1.67]:
        for mid in [5.0, 10.0]:
            for rl, rm, rh in [(0.5, 0.4, 0.3), (0.45, 0.4, 0.35), (0.55, 0.4, 0.28)]:
                for k in [0.9, 1.0, 1.1]:
                    for sig in [2.4, 2.8, 3.2]:
                        recipes.append(
                            Recipe(
                                name=f"fps_l{low:g}_m{mid:g}_r{rl:g}_{rm:g}_{rh:g}_k{k:g}",
                                family="fps_cne",
                                low_fps=low,
                                mid_fps=mid,
                                rs_low=rl,
                                rs_mid=rm,
                                rs_high=rh,
                                k_mad=k,
                                sigma=sig,
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
        return deploy_p114(sota, edge, fps)
    if float(fps) <= recipe.low_fps:
        rs = recipe.rs_low
    elif float(fps) <= recipe.mid_fps:
        rs = recipe.rs_mid
    else:
        rs = recipe.rs_high
    return deploy_p114(
        sota,
        edge,
        fps,
        residual_scale=rs,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
        ans_s_hi=recipe.s_hi,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p123_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P123 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p123")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P123",
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
