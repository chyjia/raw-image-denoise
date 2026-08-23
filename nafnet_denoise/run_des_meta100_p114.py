"""P114: CNE residual-MAD + cycle-spin Anscombe (YOND CNE × Coifman).

Baseline: P111 DES ≈ 0.9522
Combine P111 CNE gate with P110 2×2 cycle-spin.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import (
    cycle_spin_anscombe,
    deploy_p68_noise,
    deploy_p103_cne,
    flat_noise_proxy,
)


@dataclass
class Recipe:
    name: str
    family: str
    max_shift: int = 1
    residual_scale: float = 0.5
    s_lo: float = 0.12
    s_hi: float = 0.4
    noise_lo: float = 0.0008
    noise_hi: float = 0.006
    k_mad: float = 0.9
    sigma: float = 2.4


def deploy_cne_spin(sota, edge, fps, r: Recipe):
    if r.family == "baseline":
        return deploy_p103_cne(sota, edge, fps)
    out = deploy_p68_noise(sota, edge, fps)
    resid = abs(out - sota).astype("float32")
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(r.residual_scale)
    t = (n - r.noise_lo) / max(r.noise_hi - r.noise_lo, 1e-6)
    t = float(max(0.0, min(1.0, t)))
    s = r.s_lo + t * (r.s_hi - r.s_lo)
    return cycle_spin_anscombe(
        out,
        max_shift=r.max_shift,
        strength=s,
        k_mad=r.k_mad,
        sigma=r.sigma,
    )


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p111", family="baseline")]
    for ms in [1, 2]:
        for rs in [0.4, 0.5, 0.6]:
            for slo, shi in [(0.12, 0.4), (0.1, 0.35), (0.15, 0.45)]:
                for k in [0.75, 0.9, 1.0]:
                    for sig in [2.2, 2.4, 2.8]:
                        recipes.append(
                            Recipe(
                                name=f"cnes_m{ms}_r{rs:g}_s{slo:g}_{shi:g}_k{k:g}_sig{sig:g}",
                                family="cne_spin",
                                max_shift=ms,
                                residual_scale=rs,
                                s_lo=slo,
                                s_hi=shi,
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
    return deploy_cne_spin(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p114_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P114 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p114")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P114",
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
