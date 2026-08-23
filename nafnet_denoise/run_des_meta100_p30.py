"""P30: DualEx-style disagreement pixel gate + SS-TTA-lite output consistency.

Baseline: P22 DES ≈ 0.9290
Lit leftovers:
  DualEx / DEU — pixel gate from |sota−edge| disagreement
  SS-TTA / TTT — consistency average of mild noise-perturbed deploy (output-level,
  no weight update; cheap proxy of test-time consistency)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p7_fusion import edge_unsharp
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | disagree | consist
    # disagree gate
    mix: float = 0.35
    k: float = 8.0
    bias: float = 0.02
    # consistency
    n_noise: int = 3
    noise_sigma: float = 0.5  # DN
    alpha: float = 0.5  # weight of noisy-average vs clean


def disagree_blend(sota, edge, mix: float, k: float, bias: float) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    soft, _emap = blend_sota_edgekd(
        sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
    )
    diff = np.abs(sota.astype(np.float32) - edge.astype(np.float32))
    local = cv2.blur(diff, (5, 5))
    scale = float(np.percentile(local, 90)) + 1e-6
    gate = 1.0 / (1.0 + np.exp(-float(k) * (local / scale - float(bias))))
    disagree = ((1.0 - gate) * sota + gate * edge).astype(np.float32)
    m = float(np.clip(mix, 0.0, 1.0))
    return ((1.0 - m) * soft + m * disagree).astype(np.float32)


def deploy_disagree(sota, edge, fps, recipe: Recipe) -> np.ndarray:
    low_fps, mid_fps = 1.0, 5.0
    if float(fps) <= low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=0.9, harden=40.0)
        u = 0.20
    else:
        out = disagree_blend(sota, edge, recipe.mix, recipe.k, recipe.bias)
        if float(fps) <= mid_fps:
            out = flat_bilateral_boost(out, guide=out, flat_strength=0.85, harden=40.0)
            u = 0.16
        else:
            u = 0.12
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def consistency_avg(base: np.ndarray, n: int, sigma: float, alpha: float) -> np.ndarray:
    """Average base with mild Gaussian-noise→blur proxies (output-level TTA lite)."""
    rng = np.random.default_rng(0)
    outs = [base]
    for _ in range(max(int(n), 1)):
        noise = rng.normal(0.0, float(sigma), size=base.shape).astype(np.float32)
        noisy = base + noise
        # small Gaussian as "re-denoise" proxy without network
        smooth = cv2.GaussianBlur(noisy, (0, 0), sigmaX=0.6)
        outs.append(smooth.astype(np.float32))
    ens = np.mean(np.stack(outs, 0), 0).astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - a) * base + a * ens).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p22", family="baseline")]
    for mix in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for k in [6.0, 8.0, 12.0]:
            for bias in [0.01, 0.02, 0.05]:
                recipes.append(
                    Recipe(
                        name=f"dis_m{mix:g}_k{k:g}_b{bias:g}",
                        family="disagree",
                        mix=mix,
                        k=k,
                        bias=bias,
                    )
                )
    for n, sig, a in [
        (2, 0.3, 0.3),
        (3, 0.5, 0.4),
        (4, 0.5, 0.5),
        (3, 0.8, 0.35),
        (5, 0.4, 0.45),
        (2, 0.6, 0.5),
        (3, 0.35, 0.25),
        (4, 0.7, 0.4),
    ]:
        recipes.append(
            Recipe(
                name=f"con_n{n}_s{sig:g}_a{a:g}",
                family="consist",
                n_noise=n,
                noise_sigma=sig,
                alpha=a,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_dis_{i}",
                family="disagree",
                mix=0.2 + (i % 5) * 0.1,
                k=5.0 + (i % 4) * 2.0,
                bias=0.01 + (i % 3) * 0.02,
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(
                name=name,
                family=r.family,
                mix=r.mix,
                k=r.k,
                bias=r.bias,
                n_noise=r.n_noise,
                noise_sigma=r.noise_sigma,
                alpha=r.alpha,
            )
        seen.add(name)
        out.append(r)
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    if recipe.family == "disagree":
        return deploy_disagree(sota, edge, fps, recipe)
    # consist: start from P22 deploy then consistency avg
    base = deploy_p7_base(sota, edge, fps)
    return consistency_avg(base, recipe.n_noise, recipe.noise_sigma, recipe.alpha)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p30_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p30/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n{best.get('recipe')}\n",
        encoding="utf-8",
    )
    print(f"Wrote P30 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p30")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P30",
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
