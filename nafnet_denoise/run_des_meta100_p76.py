"""P76: noise-gate + edge-arm HF residual pull (DualEx / detail recovery lite).

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
from .p8_fusion import deploy_p68_noise


@dataclass
class Recipe:
    name: str
    family: str
    res_amt: float = 0.0
    res_sigma: float = 1.4
    edge_only: bool = True
    harden: float = 16.0


def edge_map(img: np.ndarray, harden: float) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = float(np.percentile(mag, 70.0)) + 1e-6
    return np.clip((mag / thr) ** (harden / 16.0), 0.0, 1.0).astype(np.float32)


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    base = deploy_p68_noise(sota, edge, fps)
    if recipe.family == "baseline" or recipe.res_amt <= 0:
        return base
    # HF of edge arm relative to base
    e_low = cv2.GaussianBlur(edge, (0, 0), recipe.res_sigma)
    b_low = cv2.GaussianBlur(base, (0, 0), recipe.res_sigma)
    hf = (edge - e_low) - (base - b_low)
    if recipe.edge_only:
        em = edge_map(base, recipe.harden)
        hf = hf * em
    out = (base + float(recipe.res_amt) * hf).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p71", family="baseline")]
    for amt in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
        for sig in [0.8, 1.2, 1.4, 1.8, 2.2]:
            for eo in [True, False]:
                for h in [8.0, 16.0, 24.0]:
                    recipes.append(
                        Recipe(
                            name=f"res_a{amt:g}_s{sig:g}_e{int(eo)}_h{h:g}",
                            family="res",
                            res_amt=amt,
                            res_sigma=sig,
                            edge_only=eo,
                            harden=h,
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


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p76_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P76 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p76")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P76",
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
