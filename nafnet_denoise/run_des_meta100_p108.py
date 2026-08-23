"""P108: Anscombe + MAD noise-gate joint refine (YOND × GALOSH).

Baseline: P98 DES ≈ 0.9472
Sweep noise_lo/hi and bilat/u together with anscombe strength.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p98


@dataclass
class Recipe:
    name: str
    family: str
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    bilat_lo: float = 0.72
    bilat_hi: float = 1.0
    u_lo: float = 0.1
    u_hi: float = 0.22
    ans_s: float = 0.2
    ans_k: float = 0.5
    ans_sig: float = 2.0
    ans_fp: float = 50.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p98", family="baseline")]
    for lo, hi in [(0.002, 0.012), (0.0015, 0.011), (0.0025, 0.014), (0.003, 0.015)]:
        for blo, bhi in [(0.72, 1.0), (0.68, 1.0), (0.75, 0.98)]:
            for ulo, uhi in [(0.1, 0.22), (0.08, 0.24), (0.12, 0.2)]:
                for s in [0.15, 0.2, 0.25]:
                    for sig in [1.8, 2.0, 2.4]:
                        recipes.append(
                            Recipe(
                                name=f"j_n{lo:g}_{hi:g}_b{blo:g}_s{s:g}_sig{sig:g}",
                                family="joint",
                                noise_lo=lo,
                                noise_hi=hi,
                                bilat_lo=blo,
                                bilat_hi=bhi,
                                u_lo=ulo,
                                u_hi=uhi,
                                ans_s=s,
                                ans_sig=sig,
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
    return deploy_p98(
        sota,
        edge,
        fps,
        ans_strength=recipe.ans_s,
        ans_k_mad=recipe.ans_k,
        ans_sigma=recipe.ans_sig,
        ans_flat_pct=recipe.ans_fp,
        noise_lo=recipe.noise_lo,
        noise_hi=recipe.noise_hi,
        bilat_lo=recipe.bilat_lo,
        bilat_hi=recipe.bilat_hi,
        u_lo=recipe.u_lo,
        u_hi=recipe.u_hi,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p108_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P108 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p108")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P108",
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
