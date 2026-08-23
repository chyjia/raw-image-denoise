"""P74: robust percentile flat-noise proxy (RPG-VST Student-t lite).

Baseline: P71 DES ≈ 0.9439
Uses MAD/percentile of flat HP instead of std for bilat/unsharp gate.
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
    mode: str = "std"  # std | mad | p75 | p90
    noise_lo: float = 0.003
    noise_hi: float = 0.015
    bilat_lo: float = 0.70
    bilat_hi: float = 0.98
    u_lo: float = 0.10
    u_hi: float = 0.22
    low_fps: float = 1.5
    flat_pct: float = 40.0


def robust_noise_proxy(img: np.ndarray, mode: str, flat_pct: float) -> float:
    g = cv2.GaussianBlur(img, (0, 0), 1.2)
    hp = np.abs(img - g)
    sob = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        g, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    flat = sob < float(np.percentile(sob, flat_pct))
    vals = hp[flat] if np.any(flat) else hp.ravel()
    if mode == "mad":
        med = float(np.median(vals))
        return float(np.median(np.abs(vals - med)) * 1.4826)
    if mode == "p75":
        return float(np.percentile(vals, 75.0))
    if mode == "p90":
        return float(np.percentile(vals, 90.0))
    return float(np.std(vals))


def deploy_robust(sota, edge, fps, r: Recipe) -> np.ndarray:
    if r.family == "baseline":
        return deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = robust_noise_proxy(guide, r.mode, r.flat_pct)
    t = (n - r.noise_lo) / max(r.noise_hi - r.noise_lo, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    bilat = r.bilat_lo + t * (r.bilat_hi - r.bilat_lo)
    u = r.u_lo + t * (r.u_hi - r.u_lo)
    if float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(sota, edge, guide_dn=guide, temperature=8.0, harden=16.0)
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p71", family="baseline")]
    for mode in ["mad", "p75", "p90", "std"]:
        for lo, hi in [(0.002, 0.012), (0.003, 0.015), (0.004, 0.018), (0.003, 0.02)]:
            for fp in [30.0, 40.0, 50.0]:
                for blo, bhi in [(0.70, 0.98), (0.72, 1.0), (0.68, 0.96)]:
                    recipes.append(
                        Recipe(
                            name=f"{mode}_{lo:g}_{hi:g}_f{fp:g}_b{blo:g}",
                            family="robust",
                            mode=mode,
                            noise_lo=lo,
                            noise_hi=hi,
                            flat_pct=fp,
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
    return deploy_robust(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p74_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P74 hook {best['name']} — sync deploy_p68_noise if mode=std", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p74")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P74",
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
