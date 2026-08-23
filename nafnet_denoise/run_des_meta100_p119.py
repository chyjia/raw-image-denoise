"""P119: YOND CNE two-pass — residual MAD after P74 gates Anscombe.

Baseline: P103 DES ≈ 0.9506
Coarse denoise then estimate noise from residual (CNE), not from blend guide.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p103, deploy_p103_cne


@dataclass
class Recipe:
    name: str
    family: str
    residual_scale: float = 1.0
    s_lo: float = 0.1
    s_hi: float = 0.3
    noise_lo: float = 0.001
    noise_hi: float = 0.008
    k_mad: float = 0.75
    sigma: float = 2.8


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p103", family="baseline")]
    for rs in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for slo, shi in [(0.1, 0.3), (0.08, 0.32), (0.12, 0.4)]:
            for lo, hi in [(0.0008, 0.006), (0.001, 0.008), (0.002, 0.012)]:
                for k in [0.6, 0.75, 0.9]:
                    for sig in [2.4, 2.8, 3.2]:
                        recipes.append(
                            Recipe(
                                name=f"cne_r{rs:g}_s{slo:g}_{shi:g}_n{lo:g}_{hi:g}_k{k:g}",
                                family="cne",
                                residual_scale=rs,
                                s_lo=slo,
                                s_hi=shi,
                                noise_lo=lo,
                                noise_hi=hi,
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
        return deploy_p103(sota, edge, fps)
    return deploy_p103_cne(
        sota,
        edge,
        fps,
        residual_scale=recipe.residual_scale,
        ans_s_lo=recipe.s_lo,
        ans_s_hi=recipe.s_hi,
        ans_noise_lo=recipe.noise_lo,
        ans_noise_hi=recipe.noise_hi,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p119_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P111 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p119")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P119",
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
