"""P57: fps-conditional blend temperature/harden (ISP-aware gate schedule).

Baseline: P40 DES ≈ 0.9294 — bilat/unsharp fixed; retune Sobel blend per fps band.
Lit: DualEx adaptive fusion; exposure-aware ISP.
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
    family: str  # baseline | blend_sched
    low_fps: float = 1.67
    mid_fps: float = 5.0
    # mid band
    T_mid: float = 8.0
    h_mid: float = 16.0
    ew_mid: float = 1.0
    # high band
    T_high: float = 8.0
    h_high: float = 16.0
    ew_high: float = 1.0
    low_bilat: float = 0.9
    mid_bilat: float = 0.85
    u_low: float = 0.20
    u_mid: float = 0.16
    u_high: float = 0.12


def deploy_blend_sched(sota, edge, fps, r: Recipe) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=r.low_bilat, harden=40.0)
        u = r.u_low
    elif float(fps) <= r.mid_fps:
        out, _ = blend_sota_edgekd(
            sota,
            edge,
            guide_dn=guide,
            temperature=r.T_mid,
            harden=r.h_mid,
            edge_weight=r.ew_mid,
        )
        out = flat_bilateral_boost(
            out, guide=out, flat_strength=r.mid_bilat, harden=40.0
        )
        u = r.u_mid
    else:
        out, _ = blend_sota_edgekd(
            sota,
            edge,
            guide_dn=guide,
            temperature=r.T_high,
            harden=r.h_high,
            edge_weight=r.ew_high,
        )
        u = r.u_high
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for Tm, hm in [(6, 12), (6, 16), (8, 12), (8, 24), (10, 16), (12, 16), (10, 24)]:
        for Th, hh in [(6, 12), (8, 16), (10, 16), (12, 24), (8, 32)]:
            recipes.append(
                Recipe(
                    name=f"bs_Tm{Tm}_hm{hm}_Th{Th}_hh{hh}",
                    family="blend_sched",
                    T_mid=float(Tm),
                    h_mid=float(hm),
                    T_high=float(Th),
                    h_high=float(hh),
                )
            )
    for ew_m, ew_h in [(0.85, 1.0), (0.95, 1.0), (1.0, 0.9), (0.9, 0.95), (0.8, 1.0)]:
        recipes.append(
            Recipe(
                name=f"bs_ew{ew_m:g}_{ew_h:g}",
                family="blend_sched",
                ew_mid=ew_m,
                ew_high=ew_h,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_bs_{i}",
                family="blend_sched",
                T_mid=6.0 + (i % 5) * 1.5,
                h_mid=8.0 + (i % 6) * 4.0,
                T_high=6.0 + (i % 4) * 2.0,
                h_high=12.0 + (i % 5) * 4.0,
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        n = r.name if r.name not in seen else f"{r.name}_x{len(out)}"
        seen.add(n)
        out.append(Recipe(**{**r.__dict__, "name": n}))
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    return deploy_blend_sched(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    recipe = best.get("recipe") or {}
    Path("nafnet_denoise/deploy_p57_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best['mean_des']:.4f}", text, count=1)
    # Prefer mid-band params as defaults (most clips); high-band in hook
    if "T_mid" in recipe:
        text = re.sub(
            r"edge_temperature: float = [0-9.]+",
            f"edge_temperature: float = {float(recipe['T_mid'])}",
            text,
            count=1,
        )
    if "h_mid" in recipe:
        text = re.sub(
            r"edge_harden: float = [0-9.]+",
            f"edge_harden: float = {float(recipe['h_mid'])}",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"Baked P57 {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p57")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P57",
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
