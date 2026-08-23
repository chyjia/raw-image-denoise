"""P59: SS-TTA-lite — re-noise deploy DN and pull toward self-consistency (output).

Baseline: P40 DES ≈ 0.9294
Lit: SS-TTA — synthesize mild noise on denoised image; consistency without GD.
Cheap: average deploy with Gaussian-smoothed noisy copies (no re-forward).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | sstta
    n: int = 3
    noise_sigma: float = 0.4
    blur_sigma: float = 0.6
    alpha: float = 0.35
    # only apply on mid/high fps
    min_fps: float = 0.0


def sstta_avg(base: np.ndarray, r: Recipe) -> np.ndarray:
    rng = np.random.default_rng(0)
    outs = [base]
    for _ in range(max(1, int(r.n))):
        noisy = base + rng.normal(0, r.noise_sigma, base.shape).astype(np.float32)
        outs.append(
            cv2.GaussianBlur(noisy, (0, 0), sigmaX=float(r.blur_sigma)).astype(np.float32)
        )
    ens = np.mean(np.stack(outs, 0), 0).astype(np.float32)
    a = float(np.clip(r.alpha, 0.0, 1.0))
    return ((1.0 - a) * base + a * ens).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for n in [2, 3, 4, 5]:
        for ns in [0.2, 0.35, 0.5, 0.7]:
            for bs in [0.4, 0.6, 0.9]:
                for a in [0.2, 0.35, 0.5]:
                    recipes.append(
                        Recipe(
                            name=f"ss_n{n}_ns{ns:g}_bs{bs:g}_a{a:g}",
                            family="sstta",
                            n=n,
                            noise_sigma=ns,
                            blur_sigma=bs,
                            alpha=a,
                        )
                    )
    for mf in [1.67, 5.0]:
        recipes.append(
            Recipe(
                name=f"ss_mid_mf{mf:g}",
                family="sstta",
                n=3,
                noise_sigma=0.4,
                alpha=0.3,
                min_fps=mf,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_ss_{i}",
                family="sstta",
                n=2 + (i % 4),
                noise_sigma=0.2 + (i % 5) * 0.1,
                blur_sigma=0.4 + (i % 4) * 0.2,
                alpha=0.15 + (i % 5) * 0.08,
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
    return sstta_avg(base, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p59_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p59/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n{best.get('recipe')}\n",
        encoding="utf-8",
    )
    print(f"Wrote P59 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p59")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P59",
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
