"""P97: spatial MAD noise map → weak/strong bilat+unsharp mix (YOND SNR-map lite).

Baseline: P74 DES ≈ 0.9442 — unlike P75 (std + mean-u), uses MAD tiles + spatial U mix.
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
    blur_sigma: float = 8.0
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    bilat_lo: float = 0.72
    bilat_hi: float = 1.0
    u_lo: float = 0.1
    u_hi: float = 0.22
    low_fps: float = 1.5


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


def deploy_mad_map(sota, edge, fps, r: Recipe) -> np.ndarray:
    if r.family == "baseline":
        return deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    nmap = local_mad_map(guide, r.tile)
    nmap = cv2.GaussianBlur(nmap, (0, 0), r.blur_sigma)
    t = (nmap - r.noise_lo) / max(r.noise_hi - r.noise_lo, 1e-6)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    if float(fps) <= r.low_fps:
        base = sota.copy()
    else:
        base, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
    weak = flat_bilateral_boost(base, guide=base, flat_strength=r.bilat_lo, harden=40.0)
    strong = flat_bilateral_boost(base, guide=base, flat_strength=r.bilat_hi, harden=40.0)
    out = weak * (1.0 - t) + strong * t
    u_weak = edge_unsharp(out, amount=r.u_lo, sigma=1.4, harden=16.0)
    u_strong = edge_unsharp(out, amount=r.u_hi, sigma=1.4, harden=16.0)
    out = u_weak * (1.0 - t) + u_strong * t
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for tile in [32, 48, 64, 96]:
        for sig in [4.0, 8.0, 12.0]:
            for lo, hi in [(0.002, 0.012), (0.003, 0.015)]:
                for blo, bhi in [(0.72, 1.0), (0.65, 1.0), (0.75, 0.98)]:
                    for ulo, uhi in [(0.1, 0.22), (0.08, 0.24)]:
                        recipes.append(
                            Recipe(
                                name=f"madm_t{tile}_s{sig:g}_n{lo:g}_{hi:g}_b{blo:g}",
                                family="mad_map",
                                tile=tile,
                                blur_sigma=sig,
                                noise_lo=lo,
                                noise_hi=hi,
                                bilat_lo=blo,
                                bilat_hi=bhi,
                                u_lo=ulo,
                                u_hi=uhi,
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
    return deploy_mad_map(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p97_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P97 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p97")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P97",
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
