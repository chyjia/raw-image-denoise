"""P49: anti-halo clipped / multi-scale unsharp on P40 arms (TIP GIF soft-erosion cue).

Baseline: P40 DES ≈ 0.9294
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, load_arms, run_meta_loop, stamp_deploy_des
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p8_fusion import clipped_unsharp, deploy_p7_base, multi_scale_unsharp


@dataclass
class Recipe:
    name: str
    family: str  # baseline | clip | msu | fps_clip
    low_fps: float = 1.67
    mid_fps: float = 5.0
    u_low: float = 0.20
    u_mid: float = 0.16
    u_high: float = 0.12
    u_sig: float = 1.4
    clip_pct: float = 98.0
    amounts: tuple[float, ...] = (0.08, 0.06, 0.04)
    sigmas: tuple[float, ...] = (0.8, 1.4, 2.5)


def deploy_base_no_u(sota, edge, fps, low_fps=1.67, mid_fps=5.0):
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=0.9, harden=40.0)
    elif float(fps) <= mid_fps:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
        out = flat_bilateral_boost(out, guide=out, flat_strength=0.85, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for amt, sig, pct in [
        (0.18, 1.4, 98.0),
        (0.20, 1.4, 97.0),
        (0.20, 1.4, 99.0),
        (0.22, 1.2, 98.0),
        (0.16, 1.6, 98.0),
        (0.20, 1.5, 96.0),
        (0.18, 1.3, 97.0),
        (0.24, 1.4, 98.0),
    ]:
        recipes.append(
            Recipe(
                name=f"clip_a{amt:g}_s{sig:g}_p{pct:g}",
                family="clip",
                u_low=amt,
                u_sig=sig,
                clip_pct=pct,
            )
        )
    for amts in [
        (0.06, 0.05, 0.03),
        (0.08, 0.05, 0.03),
        (0.1, 0.06, 0.04),
        (0.08, 0.08, 0.04),
        (0.05, 0.05, 0.05),
        (0.12, 0.04, 0.02),
    ]:
        recipes.append(
            Recipe(
                name=f"msu_{'_'.join(f'{a:g}' for a in amts)}",
                family="msu",
                amounts=amts,
                sigmas=(0.8, 1.4, 2.5)[: len(amts)],
            )
        )
    for ul, um, uh in [
        (0.20, 0.16, 0.12),
        (0.22, 0.16, 0.12),
        (0.20, 0.18, 0.12),
        (0.18, 0.16, 0.10),
    ]:
        for pct in [96.0, 98.0, 99.0]:
            recipes.append(
                Recipe(
                    name=f"fclip_{ul:g}_{um:g}_{uh:g}_p{pct:g}",
                    family="fps_clip",
                    u_low=ul,
                    u_mid=um,
                    u_high=uh,
                    clip_pct=pct,
                )
            )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_clip_{i}",
                family="clip",
                u_low=0.15 + (i % 6) * 0.02,
                u_sig=1.1 + (i % 5) * 0.15,
                clip_pct=95.0 + (i % 5),
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
    base = deploy_base_no_u(sota, edge, fps, recipe.low_fps, recipe.mid_fps)
    if recipe.family == "clip":
        return clipped_unsharp(
            base, amount=recipe.u_low, sigma=recipe.u_sig, clip_pct=recipe.clip_pct
        )
    if recipe.family == "msu":
        return multi_scale_unsharp(base, amounts=recipe.amounts, sigmas=recipe.sigmas)
    # fps_clip
    if float(fps) <= recipe.low_fps:
        amt = recipe.u_low
    elif float(fps) <= recipe.mid_fps:
        amt = recipe.u_mid
    else:
        amt = recipe.u_high
    return clipped_unsharp(
        base, amount=amt, sigma=recipe.u_sig, clip_pct=recipe.clip_pct
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p49_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p49/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\nClipped/MS unsharp — see hook.\n",
        encoding="utf-8",
    )
    print(f"Wrote P49 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p49")
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=16)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P49",
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
