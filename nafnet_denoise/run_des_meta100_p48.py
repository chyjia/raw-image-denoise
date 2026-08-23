"""P48: fine fps-band + unsharp/bilat neighborhood around P40 (1.67/5).

Baseline: P40 DES ≈ 0.9294
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, load_arms, run_meta_loop, stamp_deploy_des
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p7_fusion import edge_unsharp
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str
    low_fps: float = 1.67
    mid_fps: float = 5.0
    low_bilat: float = 0.9
    mid_bilat: float = 0.85
    bilat_harden: float = 40.0
    u_low: float = 0.20
    u_mid: float = 0.16
    u_high: float = 0.12
    u_sig: float = 1.4


def deploy_sched(sota, edge, fps, r: Recipe) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(
            out, guide=out, flat_strength=r.low_bilat, harden=r.bilat_harden
        )
        u = r.u_low
    elif float(fps) <= r.mid_fps:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
        out = flat_bilateral_boost(
            out, guide=out, flat_strength=r.mid_bilat, harden=r.bilat_harden
        )
        u = r.u_mid
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
        u = r.u_high
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=r.u_sig, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for low in [1.5, 1.6, 1.67, 1.75, 1.8, 2.0, 2.25, 2.5]:
        for mid in [4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
            recipes.append(
                Recipe(
                    name=f"fps_{low:g}_{mid:g}",
                    family="sched",
                    low_fps=low,
                    mid_fps=mid,
                )
            )
    for ul, um, uh in [
        (0.20, 0.16, 0.12),
        (0.21, 0.16, 0.12),
        (0.20, 0.17, 0.12),
        (0.20, 0.16, 0.11),
        (0.19, 0.16, 0.12),
        (0.22, 0.17, 0.13),
    ]:
        for lb, mb in [(0.9, 0.85), (0.91, 0.86), (0.92, 0.88), (0.88, 0.84)]:
            recipes.append(
                Recipe(
                    name=f"sch_u{ul:g}_{um:g}_{uh:g}_b{lb:g}_{mb:g}",
                    family="sched",
                    u_low=ul,
                    u_mid=um,
                    u_high=uh,
                    low_bilat=lb,
                    mid_bilat=mb,
                )
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="sched",
                low_fps=1.5 + (i % 8) * 0.1,
                mid_fps=4.0 + (i % 6) * 0.5,
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
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    return deploy_sched(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    recipe = best.get("recipe") or {}
    Path("nafnet_denoise/deploy_p48_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best['mean_des']:.4f}", text, count=1)
    for key, pat in [
        ("low_fps", r"low_fps: float = [0-9.]+"),
        ("mid_fps", r"mid_fps: float = [0-9.]+"),
        ("low_bilat", r"bilateral_strength: float = [0-9.]+"),
        ("mid_bilat", r"mid_bilateral_strength: float = [0-9.]+"),
        ("bilat_harden", r"bilateral_harden: float = [0-9.]+"),
        ("u_low", r"unsharp_amount_low: float = [0-9.]+"),
        ("u_mid", r"unsharp_amount_mid: float = [0-9.]+"),
        ("u_high", r"unsharp_amount_high: float = [0-9.]+"),
        ("u_sig", r"unsharp_sigma: float = [0-9.]+"),
    ]:
        if key in recipe:
            pname = pat.split(":")[0]
            text = re.sub(pat, f"{pname}: float = {float(recipe[key])}", text, count=1)
    # docstring fps line
    if "low_fps" in recipe:
        text = re.sub(
            r"fps <= [0-9.]+ →",
            f"fps <= {float(recipe['low_fps']):g} →",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"Baked P48 {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p48")
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P48",
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
