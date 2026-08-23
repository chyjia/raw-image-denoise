"""P75: spatially varying noise map → local bilat strength (nonlocal/RPG-VST map).

Baseline: P71 DES ≈ 0.9439
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
from .p8_fusion import deploy_p68_noise


@dataclass
class Recipe:
    name: str
    family: str
    tile: int = 64
    noise_lo: float = 0.003
    noise_hi: float = 0.015
    bilat_lo: float = 0.70
    bilat_hi: float = 0.98
    u_lo: float = 0.10
    u_hi: float = 0.22
    low_fps: float = 1.5
    blur_sigma: float = 8.0


def local_noise_map(img: np.ndarray, tile: int) -> np.ndarray:
    g = cv2.GaussianBlur(img, (0, 0), 1.2)
    hp = (img - g).astype(np.float32)
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    t = max(8, int(tile))
    for y in range(0, h, t):
        for x in range(0, w, t):
            patch = hp[y : y + t, x : x + t]
            out[y : y + t, x : x + t] = float(np.std(patch))
    return out


def deploy_local(sota, edge, fps, r: Recipe) -> np.ndarray:
    if r.family == "baseline":
        return deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    nmap = local_noise_map(guide, r.tile)
    nmap = cv2.GaussianBlur(nmap, (0, 0), r.blur_sigma)
    t = (nmap - r.noise_lo) / max(r.noise_hi - r.noise_lo, 1e-6)
    t = np.clip(t, 0.0, 1.0)
    bilat_mean = float(np.mean(r.bilat_lo + t * (r.bilat_hi - r.bilat_lo)))
    u_mean = float(np.mean(r.u_lo + t * (r.u_hi - r.u_lo)))
    # use mean gate (full per-pixel bilat would need custom filter; mean is cheap proxy)
    # also try residual: stronger bilat in noisier regions via weighted mix of two bilats
    if float(fps) <= r.low_fps:
        base = sota.copy()
    else:
        base, _ = blend_sota_edgekd(sota, edge, guide_dn=guide, temperature=8.0, harden=16.0)
    weak = flat_bilateral_boost(base, guide=base, flat_strength=r.bilat_lo, harden=40.0)
    strong = flat_bilateral_boost(base, guide=base, flat_strength=r.bilat_hi, harden=40.0)
    out = (weak * (1.0 - t) + strong * t).astype(np.float32)
    if u_mean > 0:
        out = edge_unsharp(out, amount=u_mean, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p71", family="baseline")]
    for tile in [32, 48, 64, 96, 128]:
        for lo, hi in [(0.003, 0.015), (0.002, 0.012), (0.004, 0.02)]:
            for sig in [4.0, 8.0, 12.0]:
                for blo, bhi in [(0.70, 0.98), (0.65, 1.0), (0.75, 0.95)]:
                    recipes.append(
                        Recipe(
                            name=f"loc_t{tile}_n{lo:g}_{hi:g}_s{sig:g}",
                            family="local",
                            tile=tile,
                            noise_lo=lo,
                            noise_hi=hi,
                            blur_sigma=sig,
                            bilat_lo=blo,
                            bilat_hi=bhi,
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
        if len(out) >= 100:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    return deploy_local(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p75_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P75 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p75")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P75",
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
