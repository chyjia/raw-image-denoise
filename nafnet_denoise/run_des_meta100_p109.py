"""P109: GALOSH-lite WHT 8x8 soft-shrink on flats after P103 gated Anscombe.

Baseline: P103 DES ≈ 0.9506
Lit: GALOSH local WHT shrinkage (training-free RAW denoise).
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
from .p8_fusion import deploy_p103


def _fwht8(block: np.ndarray) -> np.ndarray:
    """In-place-ish 2D Walsh–Hadamard on 8x8 via separable ± butterflies."""
    x = block.astype(np.float32).copy()
    for _ in range(2):
        for i in range(8):
            row = x[i]
            # 8-point fast WHT
            a0, a1, a2, a3 = row[0] + row[1], row[2] + row[3], row[4] + row[5], row[6] + row[7]
            b0, b1, b2, b3 = row[0] - row[1], row[2] - row[3], row[4] - row[5], row[6] - row[7]
            c0, c1 = a0 + a1, a2 + a3
            c2, c3 = a0 - a1, a2 - a3
            d0, d1 = b0 + b1, b2 + b3
            d2, d3 = b0 - b1, b2 - b3
            row[:] = [
                c0 + c1,
                c0 - c1,
                c2 + c3,
                c2 - c3,
                d0 + d1,
                d0 - d1,
                d2 + d3,
                d2 - d3,
            ]
        x = x.T
    return x


def wht_flat_shrink(
    img: np.ndarray,
    *,
    strength: float,
    k_mad: float,
    flat_pct: float = 50.0,
    harden: float = 40.0,
    stride: int = 4,
) -> np.ndarray:
    x = np.ascontiguousarray(img, dtype=np.float32)
    if strength <= 0:
        return x
    h, w = x.shape
    acc = np.zeros_like(x)
    wgt = np.zeros_like(x)
    _, flat = _edge_flat_maps(x, temperature=8.0, harden=harden)
    if flat_pct > 0:
        sob = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            x, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = ((flat > 0.5) & (sob < np.percentile(sob, flat_pct))).astype(np.float32)
    # MAD from flat HP
    g = cv2.GaussianBlur(x, (0, 0), 1.2)
    hp = np.abs(x - g)
    vals = hp[flat > 0.5] if np.any(flat > 0.5) else hp.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)) * 1.4826)
    thr = float(k_mad) * max(mad, 1e-6)
    st = max(1, int(stride))
    for y in range(0, h - 7, st):
        for xx in range(0, w - 7, st):
            blk = x[y : y + 8, xx : xx + 8]
            c = _fwht8(blk)
            # soft-threshold AC; keep DC
            dc = c[0, 0]
            ac = np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)
            ac[0, 0] = dc
            rec = _fwht8(ac) / 64.0
            acc[y : y + 8, xx : xx + 8] += rec
            wgt[y : y + 8, xx : xx + 8] += 1.0
    rec_full = acc / np.maximum(wgt, 1e-6)
    s = float(np.clip(strength, 0.0, 1.0))
    return (x * (1.0 - s * flat) + rec_full * (s * flat)).astype(np.float32)


@dataclass
class Recipe:
    name: str
    family: str
    strength: float = 0.0
    k_mad: float = 1.0
    flat_pct: float = 50.0
    stride: int = 4


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p103", family="baseline")]
    for s in [0.1, 0.15, 0.2, 0.25, 0.35, 0.5]:
        for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for fp in [40.0, 50.0, 60.0]:
                for st in [4, 8]:
                    recipes.append(
                        Recipe(
                            name=f"wht_s{s:g}_k{k:g}_f{fp:g}_st{st}",
                            family="wht",
                            strength=s,
                            k_mad=k,
                            flat_pct=fp,
                            stride=st,
                        )
                    )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        out.append(r)
        if len(out) >= 80:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    base = deploy_p103(sota, edge, fps)
    if recipe.family == "baseline" or recipe.strength <= 0:
        return base
    return wht_flat_shrink(
        base,
        strength=recipe.strength,
        k_mad=recipe.k_mad,
        flat_pct=recipe.flat_pct,
        stride=recipe.stride,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p109_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P109 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p109")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P109",
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
