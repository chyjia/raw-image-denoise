"""P106: noise-gated Anscombe strength (RPG-VST σz × GALOSH).

Baseline: P98 DES ≈ 0.9472
Scale anscombe strength by flat MAD proxy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise, anscombe_flat_shrink, flat_noise_proxy


@dataclass
class Recipe:
    name: str
    family: str
    s_lo: float = 0.1
    s_hi: float = 0.35
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    k_mad: float = 0.5
    sigma: float = 2.0
    flat_pct: float = 50.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p98", family="baseline")]
    for slo, shi in [(0.1, 0.3), (0.12, 0.35), (0.15, 0.4), (0.08, 0.25), (0.2, 0.2)]:
        for lo, hi in [(0.002, 0.012), (0.003, 0.015), (0.0015, 0.01)]:
            for k in [0.4, 0.5, 0.6, 0.75]:
                for sig in [1.8, 2.0, 2.4, 2.8]:
                    recipes.append(
                        Recipe(
                            name=f"ngans_s{slo:g}_{shi:g}_n{lo:g}_{hi:g}_k{k:g}_sig{sig:g}",
                            family="ng_ans",
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
    if recipe.family == "baseline":
        from .p8_fusion import deploy_p103

        return deploy_p103(sota, edge, fps)
    base = deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide, mode="mad", flat_pct=30.0)
    t = (n - recipe.noise_lo) / max(recipe.noise_hi - recipe.noise_lo, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = recipe.s_lo + t * (recipe.s_hi - recipe.s_lo)
    return anscombe_flat_shrink(
        base,
        strength=s,
        k_mad=recipe.k_mad,
        sigma=recipe.sigma,
        flat_pct=recipe.flat_pct,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p106_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P106 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p106")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P106",
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
