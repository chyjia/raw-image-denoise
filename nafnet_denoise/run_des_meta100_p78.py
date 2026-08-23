"""P78: intensity-LUT unsharp (DnLUT/TinyLUT-style local indexing on DN).

Baseline: P40 DES ≈ 0.9294
Lit: DnLUT / TinyLUT — index residual strength by local intensity bins.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .multiband_fuse import _edge_flat_maps
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | lut_u
    bins: int = 8
    amt_lo: float = 0.10
    amt_hi: float = 0.22
    sigma: float = 1.4
    edge_only: bool = True
    harden: float = 16.0
    # ramp direction: dark→more sharpen or bright→more
    invert: bool = False


def lut_unsharp(img: np.ndarray, r: Recipe) -> np.ndarray:
    x = np.ascontiguousarray(img, dtype=np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=float(r.sigma))
    hp = x - blur
    # local mean for binning
    local = cv2.GaussianBlur(x, (0, 0), sigmaX=2.0)
    lo, hi = float(local.min()), float(local.max())
    norm = (local - lo) / (hi - lo + 1e-6)
    if r.invert:
        norm = 1.0 - norm
    # piecewise linear amount
    amt = float(r.amt_lo) + (float(r.amt_hi) - float(r.amt_lo)) * norm
    if r.edge_only:
        emap, _ = _edge_flat_maps(x, temperature=8.0, harden=r.harden)
        out = x + (amt * emap) * hp
    else:
        out = x + amt * hp
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for lo, hi in [
        (0.08, 0.20),
        (0.10, 0.22),
        (0.12, 0.24),
        (0.10, 0.18),
        (0.14, 0.22),
        (0.06, 0.22),
        (0.12, 0.20),
        (0.10, 0.26),
    ]:
        for sig in [1.2, 1.4, 1.6]:
            for inv in [False, True]:
                recipes.append(
                    Recipe(
                        name=f"lut_a{lo:g}_{hi:g}_s{sig:g}{'_inv' if inv else ''}",
                        family="lut_u",
                        amt_lo=lo,
                        amt_hi=hi,
                        sigma=sig,
                        invert=inv,
                    )
                )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_{i}",
                family="lut_u",
                amt_lo=0.05 + (i % 6) * 0.02,
                amt_hi=0.16 + (i % 5) * 0.02,
                sigma=1.1 + (i % 4) * 0.2,
                invert=(i % 2 == 0),
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
    base = deploy_p7_base(sota, edge, fps, unsharp_amount_low=0.0, unsharp_amount_mid=0.0, unsharp_amount_high=0.0)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    return lut_unsharp(base, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p78_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p78/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n{best.get('recipe')}\n",
        encoding="utf-8",
    )
    print(f"Wrote P78 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p78")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P78",
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
