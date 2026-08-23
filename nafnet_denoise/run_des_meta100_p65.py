"""P65: mid_fps / low_fps threshold refine (ISP band boundaries).

Baseline: P40 DES ≈ 0.9294 — low_fps=1.67 already helped; retune mid_fps & fine low_fps.
Lit: exposure/fps-aware ISP scheduling.
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
    low_fps: float = 1.67
    mid_fps: float = 5.0
    low_bilat: float = 0.9
    mid_bilat: float = 0.85
    u_low: float = 0.20
    u_mid: float = 0.16
    u_high: float = 0.12
    u_sig: float = 1.4


def deploy_sched(sota, edge, fps, r: Recipe) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=r.low_bilat, harden=40.0)
        u = r.u_low
    elif float(fps) <= r.mid_fps:
        out, _ = blend_sota_edgekd(sota, edge, guide_dn=guide, temperature=8.0, harden=16.0)
        out = flat_bilateral_boost(out, guide=out, flat_strength=r.mid_bilat, harden=40.0)
        u = r.u_mid
    else:
        out, _ = blend_sota_edgekd(sota, edge, guide_dn=guide, temperature=8.0, harden=16.0)
        u = r.u_high
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=r.u_sig, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for low in [1.5, 1.6, 1.67, 1.75, 1.8, 2.0, 2.2, 2.5]:
        for mid in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]:
            if mid <= low:
                continue
            recipes.append(
                Recipe(name=f"fps_{low:g}_{mid:g}", family="sched", low_fps=low, mid_fps=mid)
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="sched",
                low_fps=1.4 + (i % 8) * 0.1,
                mid_fps=4.0 + (i % 7) * 0.5,
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
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    return deploy_sched(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    recipe = best.get("recipe") or {}
    Path("nafnet_denoise/deploy_p65_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best['mean_des']:.4f}", text, count=1)
    if "low_fps" in recipe:
        text = re.sub(
            r"low_fps: float = [0-9.]+",
            f"low_fps: float = {float(recipe['low_fps'])}",
            text,
            count=1,
        )
    if "mid_fps" in recipe:
        text = re.sub(
            r"mid_fps: float = [0-9.]+",
            f"mid_fps: float = {float(recipe['mid_fps'])}",
            text,
            count=1,
        )
    # refresh docstring band lines lightly
    text = re.sub(
        r"Defaults \(P\d+[^)]*mean DES ~[0-9.]+\)",
        f"Defaults (P65 fps bands, mean DES ~{best['mean_des']:.4f})",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    print(f"Baked P65 {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p65")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P65",
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
