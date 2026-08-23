"""P64: exposure_ms-conditioned schedule (PTC / ISP exposure priors).

Baseline: P40 DES ≈ 0.9294
Lit: exposure-aware denoising; brightness priors in Mono10 pipeline.
Uses per-clip exposure_ms from cache JSON to pick bilat/unsharp.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p7_fusion import edge_unsharp
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str
    # exposure thresholds (ms)
    t_dark: float = 5.0
    t_bright: float = 20.0
    # dark / mid / bright params (override fps schedule)
    bilat_d: float = 0.92
    bilat_m: float = 0.85
    bilat_b: float = 0.75
    u_d: float = 0.22
    u_m: float = 0.16
    u_b: float = 0.12
    # still respect low_fps sota-only?
    low_fps: float = 1.67
    use_fps_gate: bool = True


def deploy_exp(sota, edge, fps, exposure_ms, r: Recipe) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    exp = float(exposure_ms)
    if exp <= r.t_dark:
        bilat, u = r.bilat_d, r.u_d
    elif exp >= r.t_bright:
        bilat, u = r.bilat_b, r.u_b
    else:
        bilat, u = r.bilat_m, r.u_m

    if r.use_fps_gate and float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(sota, edge, guide_dn=guide, temperature=8.0, harden=16.0)
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for td, tb in [
        (3.0, 15.0),
        (5.0, 20.0),
        (8.0, 25.0),
        (4.0, 18.0),
        (6.0, 30.0),
        (2.0, 10.0),
        (10.0, 40.0),
        (5.0, 15.0),
    ]:
        for bd, bm, bb in [
            (0.92, 0.85, 0.75),
            (0.95, 0.88, 0.80),
            (0.90, 0.85, 0.70),
            (0.93, 0.87, 0.78),
        ]:
            for ud, um, ub in [
                (0.22, 0.16, 0.12),
                (0.24, 0.18, 0.12),
                (0.20, 0.16, 0.10),
                (0.22, 0.15, 0.14),
            ]:
                recipes.append(
                    Recipe(
                        name=f"exp_{td:g}_{tb:g}_b{bd:g}_{bm:g}_{bb:g}_u{ud:g}",
                        family="exp",
                        t_dark=td,
                        t_bright=tb,
                        bilat_d=bd,
                        bilat_m=bm,
                        bilat_b=bb,
                        u_d=ud,
                        u_m=um,
                        u_b=ub,
                    )
                )
    recipes = recipes[:100]
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(**{**r.__dict__, "name": name})
        seen.add(name)
        out.append(r)
    return out[:100]


_EXP_CACHE: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP_CACHE:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP_CACHE[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP_CACHE[stem]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    return deploy_exp(sota, edge, fps, exposure_of(cache_dir, stem), recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p64_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p64/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n"
        f"Exposure-conditioned schedule — see hook for params.\n",
        encoding="utf-8",
    )
    print(f"Wrote P64 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p64")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P64",
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
