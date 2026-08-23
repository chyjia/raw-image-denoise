"""P125: exposure-conditioned CNE-spin (AdaptiveISP × YOND CNE × Coifman).

Baseline: P114 DES ≈ 0.9528
Scale residual_scale / Anscombe strength by exposure_ms bands.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p114


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


@dataclass
class Recipe:
    name: str
    family: str
    t_dark: float = 3.0
    t_bright: float = 15.0
    dark_rs: float = 0.5
    mid_rs: float = 0.4
    bright_rs: float = 0.3
    dark_s_hi: float = 0.45
    mid_s_hi: float = 0.4
    bright_s_hi: float = 0.3
    k_mad: float = 1.0
    sigma: float = 2.8
    max_shift: int = 1


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p114", family="baseline")]
    for td, tb in [(3.0, 15.0), (5.0, 20.0), (2.0, 10.0)]:
        for drs, mrs, brs in [(0.5, 0.4, 0.3), (0.55, 0.4, 0.25), (0.45, 0.4, 0.35)]:
            for dshi, mshi, bshi in [(0.45, 0.4, 0.3), (0.5, 0.4, 0.28), (0.4, 0.35, 0.25)]:
                for k in [0.9, 1.0, 1.1]:
                    recipes.append(
                        Recipe(
                            name=f"xe_td{td:g}_dr{drs:g}_mr{mrs:g}_br{brs:g}_k{k:g}",
                            family="exp_cne",
                            t_dark=td,
                            t_bright=tb,
                            dark_rs=drs,
                            mid_rs=mrs,
                            bright_rs=brs,
                            dark_s_hi=dshi,
                            mid_s_hi=mshi,
                            bright_s_hi=bshi,
                            k_mad=k,
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
    exp = exposure_of(cache_dir, stem)
    if exp <= recipe.t_dark:
        rs, shi = recipe.dark_rs, recipe.dark_s_hi
    elif exp >= recipe.t_bright:
        rs, shi = recipe.bright_rs, recipe.bright_s_hi
    else:
        rs, shi = recipe.mid_rs, recipe.mid_s_hi
    return deploy_p114(
        sota,
        edge,
        fps,
        max_shift=recipe.max_shift,
        residual_scale=rs,
        ans_s_lo=0.12,
        ans_s_hi=shi,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p125_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P121 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p125")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P125",
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
