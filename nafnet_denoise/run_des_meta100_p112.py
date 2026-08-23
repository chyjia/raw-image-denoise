"""P112: YOND SNR-map — local MAD tiles gate Anscombe strength spatially.

Baseline: P103 DES ≈ 0.9506
P97 MAD-map on bilat failed; here the map drives the winning Anscombe lever.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import anscombe_flat_shrink_map, deploy_p68_noise, deploy_p103_cycspin


@dataclass
class Recipe:
    name: str
    family: str
    tile: int = 64
    blur: float = 8.0
    s_lo: float = 0.1
    s_hi: float = 0.3
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    k_mad: float = 0.75
    sigma: float = 2.8


def local_mad_map(img: np.ndarray, tile: int) -> np.ndarray:
    g = cv2.GaussianBlur(img, (0, 0), 1.2)
    hp = np.abs(img - g).astype(np.float32)
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    t = max(8, int(tile))
    for y in range(0, h, t):
        for x in range(0, w, t):
            patch = hp[y : y + t, x : x + t]
            med = float(np.median(patch))
            out[y : y + t, x : x + t] = float(np.median(np.abs(patch - med)) * 1.4826)
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p103_cycspin(sota, edge, fps)
    base = deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    nmap = local_mad_map(guide, recipe.tile)
    nmap = cv2.GaussianBlur(nmap, (0, 0), recipe.blur)
    t = (nmap - recipe.noise_lo) / max(recipe.noise_hi - recipe.noise_lo, 1e-6)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    smap = recipe.s_lo + t * (recipe.s_hi - recipe.s_lo)
    return anscombe_flat_shrink_map(
        base, smap, k_mad=recipe.k_mad, sigma=recipe.sigma
    )


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p103", family="baseline")]
    for tile in [32, 48, 64, 96]:
        for blur in [4.0, 8.0, 12.0]:
            for slo, shi in [(0.1, 0.3), (0.08, 0.35), (0.12, 0.4)]:
                for k in [0.6, 0.75, 0.9]:
                    for sig in [2.4, 2.8, 3.2]:
                        recipes.append(
                            Recipe(
                                name=f"snr_t{tile}_b{blur:g}_s{slo:g}_{shi:g}_k{k:g}",
                                family="snr_map",
                                tile=tile,
                                blur=blur,
                                s_lo=slo,
                                s_hi=shi,
                                k_mad=k,
                                sigma=sig,
                            )
                        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        out.append(r)
        if len(out) >= 100:
            break
    return out


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p112_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P112 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p112")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P112",
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
