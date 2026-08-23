"""P96: YOND-lite exposure×MAD joint schedule (SNR × exposure cues).

Baseline: P74 DES ≈ 0.9442
Scale noise-gated bilat/unsharp by exposure band multipliers.
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
from .p8_fusion import deploy_p68_noise, flat_noise_proxy


@dataclass
class Recipe:
    name: str
    family: str
    t_dark: float = 3.0
    t_bright: float = 15.0
    dark_b: float = 1.05
    bright_b: float = 0.92
    dark_u: float = 1.08
    bright_u: float = 0.9
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    bilat_lo: float = 0.72
    bilat_hi: float = 1.0
    u_lo: float = 0.1
    u_hi: float = 0.22
    low_fps: float = 1.5
    flat_pct: float = 30.0


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


def deploy_joint(sota, edge, fps, exposure_ms, r: Recipe) -> np.ndarray:
    if r.family == "baseline":
        return deploy_p68_noise(sota, edge, fps)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide, mode="mad", flat_pct=r.flat_pct)
    t = (n - r.noise_lo) / max(r.noise_hi - r.noise_lo, 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    bilat = r.bilat_lo + t * (r.bilat_hi - r.bilat_lo)
    u = r.u_lo + t * (r.u_hi - r.u_lo)
    exp = float(exposure_ms)
    if exp <= r.t_dark:
        bilat *= r.dark_b
        u *= r.dark_u
    elif exp >= r.t_bright:
        bilat *= r.bright_b
        u *= r.bright_u
    bilat = float(np.clip(bilat, 0.4, 1.0))
    u = float(np.clip(u, 0.05, 0.35))
    if float(fps) <= r.low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
        )
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)
    return out.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for td, tb in [(3.0, 15.0), (5.0, 20.0), (2.0, 10.0)]:
        for db, bb in [(1.05, 0.92), (1.1, 0.88), (1.0, 0.95), (1.08, 0.9)]:
            for du, bu in [(1.08, 0.9), (1.12, 0.85), (1.0, 0.95), (1.15, 0.88)]:
                for lo, hi in [(0.002, 0.012), (0.003, 0.015)]:
                    recipes.append(
                        Recipe(
                            name=f"xn_td{td:g}_tb{tb:g}_db{db:g}_bb{bb:g}_du{du:g}",
                            family="exp_mad",
                            t_dark=td,
                            t_bright=tb,
                            dark_b=db,
                            bright_b=bb,
                            dark_u=du,
                            bright_u=bu,
                            noise_lo=lo,
                            noise_hi=hi,
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
    return deploy_joint(sota, edge, fps, exposure_of(cache_dir, stem), recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p96_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P96 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p96")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P96",
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
