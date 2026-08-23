"""P120: recursive cycle-spin Anscombe (EURASIP'25 modified recursive CS).

Baseline: P114 DES ≈ 0.9528
Iterate CNE-gated Anscombe with decaying threshold scale.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import (
    cycle_spin_anscombe,
    deploy_p114,
    deploy_p68_noise,
    flat_noise_proxy,
)


@dataclass
class Recipe:
    name: str
    family: str
    n_iters: int = 2
    thr_decay: float = 0.7
    max_shift: int = 1
    residual_scale: float = 0.4
    s_lo: float = 0.12
    s_hi: float = 0.4
    noise_lo: float = 0.0008
    noise_hi: float = 0.006
    k_mad: float = 1.0
    sigma: float = 2.8


def recursive_cne_spin(sota, edge, fps, r: Recipe):
    if r.family == "baseline":
        return deploy_p114(sota, edge, fps)
    out = deploy_p68_noise(sota, edge, fps)
    resid = np.abs(out - sota).astype(np.float32)
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(r.residual_scale)
    t = (n - r.noise_lo) / max(r.noise_hi - r.noise_lo, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = r.s_lo + t * (r.s_hi - r.s_lo)
    k = float(r.k_mad)
    x = out
    for _ in range(int(r.n_iters)):
        x = cycle_spin_anscombe(
            x,
            max_shift=r.max_shift,
            strength=s,
            k_mad=k,
            sigma=r.sigma,
        )
        k *= float(r.thr_decay)
    return x.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p114", family="baseline")]
    for nit in [2, 3, 4]:
        for decay in [0.5, 0.7, 0.85]:
            for k in [0.9, 1.0, 1.2]:
                for sig in [2.4, 2.8, 3.2]:
                    for slo, shi in [(0.12, 0.4), (0.1, 0.35)]:
                        recipes.append(
                            Recipe(
                                name=f"rcs_n{nit}_d{decay:g}_k{k:g}_sig{sig:g}_s{slo:g}",
                                family="recurse",
                                n_iters=nit,
                                thr_decay=decay,
                                k_mad=k,
                                sigma=sig,
                                s_lo=slo,
                                s_hi=shi,
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
    return recursive_cne_spin(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p120_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P120 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p120")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P120",
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
