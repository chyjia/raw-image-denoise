"""P56: DualEx-style HF residual compensation after P40 deploy.

Baseline: P40 DES ≈ 0.9294
Lit: DualExNet — fuse dual predictions then add gated high-frequency residual
from edge−sota (structure compensation without replacing the blend).
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
    family: str  # baseline | hf_res
    amount: float = 0.15
    sigma: float = 1.4
    edge_only: bool = True
    harden: float = 16.0
    # apply only above low_fps (keep SOTA path untouched)
    min_fps: float = 1.67


def hf_residual(sota, edge, amount, sigma, edge_only, harden) -> np.ndarray:
    s = np.ascontiguousarray(sota, dtype=np.float32)
    e = np.ascontiguousarray(edge, dtype=np.float32)
    s_low = cv2.GaussianBlur(s, (0, 0), sigmaX=float(sigma))
    e_low = cv2.GaussianBlur(e, (0, 0), sigmaX=float(sigma))
    resid = (e - e_low) - (s - s_low)
    if edge_only:
        emap, _ = _edge_flat_maps(0.5 * s + 0.5 * e, temperature=8.0, harden=harden)
        return (float(amount) * emap * resid).astype(np.float32)
    return (float(amount) * resid).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for amt in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28]:
        for sig in [1.0, 1.2, 1.4, 1.8]:
            for eo in [True, False]:
                recipes.append(
                    Recipe(
                        name=f"hf_a{amt:g}_s{sig:g}_{'e' if eo else 'f'}",
                        family="hf_res",
                        amount=amt,
                        sigma=sig,
                        edge_only=eo,
                    )
                )
    for amt, mf in [(0.1, 1.67), (0.15, 5.0), (0.12, 2.5), (0.2, 5.0)]:
        recipes.append(
            Recipe(
                name=f"hf_a{amt:g}_mf{mf:g}",
                family="hf_res",
                amount=amt,
                min_fps=mf,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_hf_{i}",
                family="hf_res",
                amount=0.05 + (i % 8) * 0.03,
                sigma=1.0 + (i % 5) * 0.2,
                edge_only=(i % 2 == 0),
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
    base = deploy_p7_base(sota, edge, fps)
    if recipe.family == "baseline":
        return base
    if float(fps) < float(recipe.min_fps):
        return base
    return (base + hf_residual(
        sota, edge, recipe.amount, recipe.sigma, recipe.edge_only, recipe.harden
    )).astype(np.float32)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p56_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p56/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n"
        f"Apply HF residual post deploy; see recipe.\n{best.get('recipe')}\n",
        encoding="utf-8",
    )
    print(f"Wrote P56 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p56")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P56",
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
