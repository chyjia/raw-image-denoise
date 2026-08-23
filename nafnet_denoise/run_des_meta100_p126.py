"""P126: EUI inverse on recursive CNE-spin (Makitalo–Foi TIP'12).

Baseline: P120 DES ≈ 0.9556
Replace algebraic GAT inverse with exact unbiased closed-form.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p120, deploy_p126


@dataclass
class Recipe:
    name: str
    family: str
    read_sigma: float = 0.02
    n_iters: int = 2
    thr_decay: float = 0.5
    max_shift: int = 1
    residual_scale: float = 0.4
    k_mad: float = 1.2
    sigma: float = 2.8
    s_lo: float = 0.12
    s_hi: float = 0.4


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p120", family="baseline")]
    for rs in [0.0, 0.01, 0.02, 0.03, 0.05]:
        for nit in [2, 3]:
            for decay in [0.5, 0.7]:
                for k in [1.0, 1.2, 1.4]:
                    for sig in [2.4, 2.8, 3.2]:
                        recipes.append(
                            Recipe(
                                name=f"eui_rs{rs:g}_n{nit}_d{decay:g}_k{k:g}_sig{sig:g}",
                                family="eui",
                                read_sigma=rs,
                                n_iters=nit,
                                thr_decay=decay,
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
        return deploy_p120(sota, edge, fps)
    return deploy_p126(
        sota,
        edge,
        fps,
        read_sigma=recipe.read_sigma,
        n_iters=recipe.n_iters,
        thr_decay=recipe.thr_decay,
        max_shift=recipe.max_shift,
        residual_scale=recipe.residual_scale,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
        ans_s_lo=recipe.s_lo,
        ans_s_hi=recipe.s_hi,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p126_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P126 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p126")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P126",
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
