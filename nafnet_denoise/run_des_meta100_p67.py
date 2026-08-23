"""P67: fps×exposure 2D schedule (P61 bands × P64 exposure).

Baseline: P64 DES ≈ 0.9382
Lit: AdaptiveISP — jointly select pipeline params from scene cues (fps + exposure).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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
    low_fps: float = 1.67
    mid_fps: float = 10.0
    t_dark: float = 3.0
    t_bright: float = 15.0
    # dark/mid/bright × (optional mid-fps vs high-fps bilat scale)
    bilat_d: float = 0.95
    bilat_m: float = 0.88
    bilat_b: float = 0.80
    u_d: float = 0.22
    u_m: float = 0.15
    u_b: float = 0.14
    high_bilat_scale: float = 1.0  # scale bilat when fps > mid_fps
    high_u_scale: float = 1.0


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


def deploy_2d(sota, edge, fps, exposure_ms, r: Recipe) -> np.ndarray:
    if r.family == "baseline":
        return deploy_p64_exposure(sota, edge, fps, exposure_ms)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    exp = float(exposure_ms)
    if exp <= r.t_dark:
        bilat, u = r.bilat_d, r.u_d
    elif exp >= r.t_bright:
        bilat, u = r.bilat_b, r.u_b
    else:
        bilat, u = r.bilat_m, r.u_m
    if float(fps) > r.mid_fps:
        bilat *= r.high_bilat_scale
        u *= r.high_u_scale
    if float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(sota, edge, guide_dn=guide, temperature=8.0, harden=16.0)
        if bilat > 0:
            out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p64", family="baseline")]
    for low in [1.5, 1.67, 2.0]:
        for mid in [5.0, 8.0, 10.0, 12.0, 15.0]:
            for hs in [0.0, 0.5, 0.85, 1.0]:
                for us in [0.7, 0.85, 1.0, 1.1]:
                    recipes.append(
                        Recipe(
                            name=f"2d_{low:g}_{mid:g}_hb{hs:g}_hu{us:g}",
                            family="2d",
                            low_fps=low,
                            mid_fps=mid,
                            high_bilat_scale=hs,
                            high_u_scale=us,
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
    return deploy_2d(sota, edge, fps, exposure_of(cache_dir, stem), recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p67_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P67 hook {best['name']} — manual deploy sync if needed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p67")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P67",
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
