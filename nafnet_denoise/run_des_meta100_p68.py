"""P68: flat-variance noise proxy gates bilat strength (RPG-VST / noise-map lite).

Baseline: P64 DES ≈ 0.9382
Lit: RPG-VST reliability signal; AdaptiveISP module strength from scene stats.
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
from .p8_fusion import deploy_p64_exposure


@dataclass
class Recipe:
    name: str
    family: str
    noise_lo: float = 0.008
    noise_hi: float = 0.025
    bilat_lo: float = 0.75
    bilat_hi: float = 0.98
    u_lo: float = 0.12
    u_hi: float = 0.24
    low_fps: float = 1.67
    mix_p64: float = 0.0  # blend toward fixed P64


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


def flat_noise_proxy(img: np.ndarray) -> float:
    g = cv2.GaussianBlur(img, (0, 0), 1.2)
    hp = img - g
    sob = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        g, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    flat = sob < float(np.percentile(sob, 40.0))
    if not np.any(flat):
        return float(np.std(hp))
    return float(np.std(hp[flat]))


def deploy_noise(sota, edge, fps, exposure_ms, r: Recipe) -> np.ndarray:
    if r.family == "baseline":
        return deploy_p64_exposure(sota, edge, fps, exposure_ms)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide)
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
    if r.mix_p64 > 0:
        base = deploy_p64_exposure(sota, edge, fps, exposure_ms)
        out = ((1.0 - r.mix_p64) * out + r.mix_p64 * base).astype(np.float32)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p64", family="baseline")]
    for lo, hi in [(0.005, 0.02), (0.008, 0.025), (0.01, 0.03), (0.006, 0.022)]:
        for blo, bhi in [(0.75, 0.98), (0.80, 0.95), (0.70, 1.0), (0.85, 0.97)]:
            for ulo, uhi in [(0.12, 0.24), (0.10, 0.22), (0.14, 0.22), (0.12, 0.20)]:
                for mix in [0.0, 0.3, 0.5]:
                    recipes.append(
                        Recipe(
                            name=f"n{lo:g}_{hi:g}_b{blo:g}_{bhi:g}_m{mix:g}",
                            family="noise",
                            noise_lo=lo,
                            noise_hi=hi,
                            bilat_lo=blo,
                            bilat_hi=bhi,
                            u_lo=ulo,
                            u_hi=uhi,
                            mix_p64=mix,
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
    return deploy_noise(sota, edge, fps, exposure_of(cache_dir, stem), recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p68_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P68 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p68")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P68",
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
