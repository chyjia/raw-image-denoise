"""P47: guided-filter flat boost (He GIF / 2025 soft-erosion GIF lit) on P40.

Baseline: P40 DES ≈ 0.9294 — replace or stack flat bilat with guided filter.
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
from .p7_fusion import edge_unsharp, flat_guided_boost
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | gif | stack
    low_fps: float = 1.67
    mid_fps: float = 5.0
    strength: float = 0.9
    radius: int = 4
    eps: float = 1e-3
    harden: float = 40.0
    # stack: bilat then gif
    bilat: float = 0.5
    u_low: float = 0.20
    u_mid: float = 0.16
    u_high: float = 0.12


def deploy_gif(sota, edge, fps, r: Recipe, stack: bool) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= r.low_fps:
        out = sota.copy()
        if stack and r.bilat > 0:
            out = flat_bilateral_boost(
                out, guide=out, flat_strength=r.bilat, harden=r.harden
            )
        out = flat_guided_boost(
            out,
            guide=out,
            flat_strength=r.strength,
            radius=r.radius,
            eps=r.eps,
            harden=r.harden,
        )
        u = r.u_low
    elif float(fps) <= r.mid_fps:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
        if stack and r.bilat > 0:
            out = flat_bilateral_boost(
                out, guide=out, flat_strength=r.bilat, harden=r.harden
            )
        out = flat_guided_boost(
            out,
            guide=out,
            flat_strength=r.strength,
            radius=r.radius,
            eps=r.eps,
            harden=r.harden,
        )
        u = r.u_mid
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
        u = r.u_high
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for s in [0.7, 0.85, 0.9, 0.95]:
        for rad in [2, 3, 4, 6, 8]:
            for eps in [1e-4, 5e-4, 1e-3, 5e-3]:
                recipes.append(
                    Recipe(
                        name=f"gif_s{s:g}_r{rad}_e{eps:g}",
                        family="gif",
                        strength=s,
                        radius=rad,
                        eps=eps,
                    )
                )
    for s, b in [(0.6, 0.4), (0.7, 0.5), (0.5, 0.6), (0.8, 0.3)]:
        for rad in [3, 4, 6]:
            recipes.append(
                Recipe(
                    name=f"stack_s{s:g}_b{b:g}_r{rad}",
                    family="stack",
                    strength=s,
                    bilat=b,
                    radius=rad,
                )
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_gif_{i}",
                family="gif",
                strength=0.7 + (i % 4) * 0.05,
                radius=2 + (i % 5),
                eps=1e-4 * (1 + i % 5),
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
    return deploy_gif(sota, edge, fps, recipe, stack=(recipe.family == "stack"))


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p47_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p47/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\nGuided-filter flat — see hook.\n",
        encoding="utf-8",
    )
    print(f"Wrote P47 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p47")
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=16)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P47",
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
