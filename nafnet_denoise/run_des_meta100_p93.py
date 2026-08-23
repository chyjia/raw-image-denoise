"""P93: noise-gated UMGF amount (MAD proxy × structure transfer).

Baseline: P74 DES ≈ 0.9442
Lit: AdaptiveISP module strength from scene stats + UMGF detail recovery.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise, flat_noise_proxy, umgf_fuse


@dataclass
class Recipe:
    name: str
    family: str
    amt_lo: float = 0.0
    amt_hi: float = 0.3
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    sigma: float = 1.4
    harden: float = 16.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for lo, hi in [(0.002, 0.012), (0.003, 0.015)]:
        for alo, ahi in [(0.05, 0.25), (0.08, 0.35), (0.1, 0.45), (0.12, 0.5)]:
            for sig in [1.0, 1.4, 1.8, 2.2]:
                for h in [8.0, 16.0, 24.0]:
                    recipes.append(
                        Recipe(
                            name=f"ng_umgf_n{lo:g}_{hi:g}_a{alo:g}_{ahi:g}_s{sig:g}",
                            family="ng_umgf",
                            amt_lo=alo,
                            amt_hi=ahi,
                            noise_lo=lo,
                            noise_hi=hi,
                            sigma=sig,
                            harden=h,
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
    if recipe.family == "baseline" or recipe.amt_hi <= 0:
        return base
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide, mode="mad", flat_pct=30.0)
    t = (n - recipe.noise_lo) / max(recipe.noise_hi - recipe.noise_lo, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    amt = recipe.amt_lo + t * (recipe.amt_hi - recipe.amt_lo)
    return umgf_fuse(
        base, edge, amount=amt, sigma=recipe.sigma, harden=recipe.harden
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p93_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P93 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p93")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P93",
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
