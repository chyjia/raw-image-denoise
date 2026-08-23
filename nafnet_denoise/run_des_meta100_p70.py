"""P70: softmix P64 with stronger/weaker exposure arms (EnsIR-style intensity bank).

Baseline: P64 DES ≈ 0.9382
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p64_exposure


@dataclass
class Recipe:
    name: str
    family: str
    w_strong: float = 0.0
    w_weak: float = 0.0
    # strong/weak arms
    bilat_d_s: float = 0.98
    bilat_m_s: float = 0.92
    bilat_b_s: float = 0.85
    u_d_s: float = 0.25
    u_m_s: float = 0.18
    u_b_s: float = 0.16
    bilat_d_w: float = 0.90
    bilat_m_w: float = 0.82
    bilat_b_w: float = 0.70
    u_d_w: float = 0.18
    u_m_w: float = 0.12
    u_b_w: float = 0.10


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    exp = exposure_of(cache_dir, stem)
    base = deploy_p64_exposure(sota, edge, fps, exp)
    if recipe.family == "baseline" or (recipe.w_strong <= 0 and recipe.w_weak <= 0):
        return base
    strong = deploy_p64_exposure(
        sota,
        edge,
        fps,
        exp,
        bilat_d=recipe.bilat_d_s,
        bilat_m=recipe.bilat_m_s,
        bilat_b=recipe.bilat_b_s,
        u_d=recipe.u_d_s,
        u_m=recipe.u_m_s,
        u_b=recipe.u_b_s,
    )
    weak = deploy_p64_exposure(
        sota,
        edge,
        fps,
        exp,
        bilat_d=recipe.bilat_d_w,
        bilat_m=recipe.bilat_m_w,
        bilat_b=recipe.bilat_b_w,
        u_d=recipe.u_d_w,
        u_m=recipe.u_m_w,
        u_b=recipe.u_b_w,
    )
    ws, ww = float(recipe.w_strong), float(recipe.w_weak)
    w0 = max(0.0, 1.0 - ws - ww)
    out = (w0 * base + ws * strong + ww * weak).astype(np.float32)
    return out


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p64", family="baseline")]
    for ws in [0.0, 0.15, 0.25, 0.35, 0.5]:
        for ww in [0.0, 0.15, 0.25, 0.35]:
            if ws + ww > 0.85:
                continue
            if ws == 0 and ww == 0:
                continue
            recipes.append(
                Recipe(name=f"mix_s{ws:g}_w{ww:g}", family="mix", w_strong=ws, w_weak=ww)
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="mix",
                w_strong=0.1 + (i % 5) * 0.08,
                w_weak=0.05 + (i % 4) * 0.05,
            )
        )
    return recipes[:100]


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p70_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P70 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p70")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P70",
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
