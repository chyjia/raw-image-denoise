"""P95: RPG-VST σ_z dual-fit gate — choose mad vs std per image (arXiv'25).

Baseline: P74 DES ≈ 0.9442
If mad/std reliability ratio is out of band, fall back to std gate; else MAD.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise, flat_noise_proxy


@dataclass
class Recipe:
    name: str
    family: str
    ratio_lo: float = 0.55
    ratio_hi: float = 1.35
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    bilat_lo: float = 0.72
    bilat_hi: float = 1.0
    u_lo: float = 0.1
    u_hi: float = 0.22
    flat_pct: float = 30.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for rlo, rhi in [(0.5, 1.4), (0.55, 1.35), (0.6, 1.3), (0.7, 1.2), (0.45, 1.5)]:
        for lo, hi in [(0.002, 0.012), (0.003, 0.015), (0.002, 0.014)]:
            for blo, bhi in [(0.72, 1.0), (0.68, 1.0), (0.75, 0.98)]:
                for ulo, uhi in [(0.1, 0.22), (0.08, 0.24), (0.12, 0.2)]:
                    for fp in [25.0, 30.0, 35.0]:
                        recipes.append(
                            Recipe(
                                name=f"sz_r{rlo:g}_{rhi:g}_n{lo:g}_{hi:g}_b{blo:g}_f{fp:g}",
                                family="sigmaz",
                                ratio_lo=rlo,
                                ratio_hi=rhi,
                                noise_lo=lo,
                                noise_hi=hi,
                                bilat_lo=blo,
                                bilat_hi=bhi,
                                u_lo=ulo,
                                u_hi=uhi,
                                flat_pct=fp,
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
        return deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n_mad = flat_noise_proxy(guide, mode="mad", flat_pct=recipe.flat_pct)
    n_std = flat_noise_proxy(guide, mode="std", flat_pct=recipe.flat_pct)
    ratio = n_mad / max(n_std, 1e-8)
    mode = "mad" if recipe.ratio_lo <= ratio <= recipe.ratio_hi else "std"
    return deploy_p68_noise(
        sota,
        edge,
        fps,
        noise_lo=recipe.noise_lo,
        noise_hi=recipe.noise_hi,
        bilat_lo=recipe.bilat_lo,
        bilat_hi=recipe.bilat_hi,
        u_lo=recipe.u_lo,
        u_hi=recipe.u_hi,
        noise_mode=mode,
        flat_pct=recipe.flat_pct,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p95_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P95 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p95")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P95",
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
