"""P31: fps-threshold / bilat-harden / unsharp-sigma schedule tweaks (zero-train).

Baseline: P22 DES ≈ 0.9290
Lit: exposure/fps-aware ISP scheduling; Gen2 bilat harden was fixed at 40.
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
    family: str  # baseline | sched
    low_fps: float = 1.0
    mid_fps: float = 5.0
    low_bilat: float = 0.9
    mid_bilat: float = 0.85
    bilat_harden: float = 40.0
    u_low: float = 0.20
    u_mid: float = 0.16
    u_high: float = 0.12
    u_sig: float = 1.4
    blend_T: float = 8.0
    blend_h: float = 16.0


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
            sota, edge, guide_dn=guide, temperature=r.blend_T, harden=r.blend_h
        )
        out = flat_bilateral_boost(
            out, guide=out, flat_strength=r.mid_bilat, harden=r.bilat_harden
        )
        u = r.u_mid
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=r.blend_T, harden=r.blend_h
        )
        u = r.u_high
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=r.u_sig, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p22", family="baseline")]
    for low, mid in [
        (0.8, 4.0),
        (1.0, 4.0),
        (1.0, 6.0),
        (1.2, 5.0),
        (0.5, 5.0),
        (1.5, 5.0),
        (1.0, 8.0),
        (0.75, 3.0),
    ]:
        recipes.append(
            Recipe(name=f"fps_{low:g}_{mid:g}", family="sched", low_fps=low, mid_fps=mid)
        )
    for h in [24.0, 32.0, 40.0, 48.0, 56.0, 64.0]:
        recipes.append(
            Recipe(name=f"bh_{h:g}", family="sched", bilat_harden=h)
        )
    for sig in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        recipes.append(Recipe(name=f"usig_{sig:g}", family="sched", u_sig=sig))
    for ul, um, uh in [
        (0.20, 0.16, 0.12),
        (0.22, 0.16, 0.12),
        (0.20, 0.18, 0.12),
        (0.20, 0.16, 0.14),
        (0.18, 0.16, 0.12),
        (0.22, 0.18, 0.10),
        (0.24, 0.16, 0.12),
        (0.20, 0.14, 0.10),
    ]:
        for lb, mb in [(0.9, 0.85), (0.92, 0.85), (0.9, 0.88), (0.95, 0.9)]:
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
                bilat_harden=30.0 + (i % 5) * 8.0,
                u_sig=1.1 + (i % 5) * 0.2,
                low_fps=0.5 + (i % 4) * 0.25,
                mid_fps=3.0 + (i % 5) * 1.0,
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
    Path("nafnet_denoise/deploy_p31_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best['mean_des']:.4f}", text, count=1)
    mapping = [
        ("low_fps", r"low_fps: float = [0-9.]+", "low_fps"),
        ("mid_fps", r"mid_fps: float = [0-9.]+", "mid_fps"),
        ("low_bilat", r"bilateral_strength: float = [0-9.]+", "bilateral_strength"),
        ("mid_bilat", r"mid_bilateral_strength: float = [0-9.]+", "mid_bilateral_strength"),
        ("bilat_harden", r"bilateral_harden: float = [0-9.]+", "bilateral_harden"),
        ("u_low", r"unsharp_amount_low: float = [0-9.]+", "unsharp_amount_low"),
        ("u_mid", r"unsharp_amount_mid: float = [0-9.]+", "unsharp_amount_mid"),
        ("u_high", r"unsharp_amount_high: float = [0-9.]+", "unsharp_amount_high"),
        ("u_sig", r"unsharp_sigma: float = [0-9.]+", "unsharp_sigma"),
    ]
    for key, pat, _ in mapping:
        if key in recipe:
            # extract param name from pat
            pname = pat.split(":")[0]
            text = re.sub(pat, f"{pname}: float = {float(recipe[key])}", text, count=1)
    path.write_text(text, encoding="utf-8")
    print(f"Baked P31 {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p31")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P31",
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
