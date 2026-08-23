"""P39: wavelet (Haar pyramid) LF/HF fuse of SOTA↔edge arms.

Baseline: P31 DES ≈ 0.9292
Lit: WaveUIR / DualEx HF — LF from flat SOTA, HF from edge.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str
    levels: int = 2
    edge_hf: float = 0.85
    edge_lf: float = 0.0


def haar_split(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-level approx via Gaussian pyramid as wavelet-lite."""
    low = cv2.pyrDown(img)
    low_up = cv2.pyrUp(low, dstsize=(img.shape[1], img.shape[0]))
    if low_up.shape != img.shape:
        low_up = cv2.resize(low_up, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    high = img - low_up
    return low_up.astype(np.float32), high.astype(np.float32)


def wavelet_fuse(sota, edge, levels: int, edge_hf: float, edge_lf: float) -> np.ndarray:
    s = np.ascontiguousarray(sota, dtype=np.float32)
    e = np.ascontiguousarray(edge, dtype=np.float32)
    s_highs, e_highs = [], []
    for _ in range(max(1, levels)):
        s_low, s_hi = haar_split(s)
        e_low, e_hi = haar_split(e)
        s_highs.append(s_hi)
        e_highs.append(e_hi)
        s, e = s_low, e_low
    ehf = float(np.clip(edge_hf, 0.0, 1.0))
    elf = float(np.clip(edge_lf, 0.0, 1.0))
    out = (1.0 - elf) * s + elf * e
    for sh, eh in zip(reversed(s_highs), reversed(e_highs)):
        out = out + (1.0 - ehf) * sh + ehf * eh
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p31", family="baseline")]
    for lv in [1, 2, 3]:
        for ehf in [0.6, 0.75, 0.85, 0.95, 1.0]:
            for elf in [0.0, 0.1, 0.2]:
                recipes.append(
                    Recipe(
                        name=f"wav_l{lv}_hf{ehf:g}_lf{elf:g}",
                        family="wav",
                        levels=lv,
                        edge_hf=ehf,
                        edge_lf=elf,
                    )
                )
    while len(recipes) < 80:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_wav_{i}",
                family="wav",
                levels=1 + (i % 3),
                edge_hf=0.5 + (i % 6) * 0.1,
                edge_lf=(i % 4) * 0.05,
            )
        )
    seen = set()
    out = []
    for r in recipes:
        n = r.name if r.name not in seen else f"{r.name}_x{len(out)}"
        seen.add(n)
        out.append(Recipe(name=n, family=r.family, levels=r.levels, edge_hf=r.edge_hf, edge_lf=r.edge_lf))
    return out[:80]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    fused = wavelet_fuse(sota, edge, recipe.levels, recipe.edge_hf, recipe.edge_lf)
    return deploy_p7_base(fused, fused, fps)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p39_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"P39 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout"))
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p39"))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P39",
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
