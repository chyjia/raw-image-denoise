"""P81: fine MAD threshold / bilat grid around P74 winner.

Baseline: P74 DES ≈ 0.9442
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise


@dataclass
class Recipe:
    name: str
    family: str
    noise_lo: float = 0.002
    noise_hi: float = 0.012
    bilat_lo: float = 0.72
    bilat_hi: float = 1.0
    u_lo: float = 0.10
    u_hi: float = 0.22
    flat_pct: float = 30.0
    low_fps: float = 1.5


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for lo in [0.0015, 0.002, 0.0025, 0.003]:
        for hi in [0.010, 0.011, 0.012, 0.013, 0.014]:
            if hi <= lo:
                continue
            for blo in [0.68, 0.70, 0.72, 0.74, 0.76]:
                for bhi in [0.96, 0.98, 1.0]:
                    for fp in [25.0, 30.0, 35.0]:
                        for ulo, uhi in [(0.10, 0.22), (0.09, 0.21), (0.11, 0.23)]:
                            recipes.append(
                                Recipe(
                                    name=f"mad_{lo:g}_{hi:g}_b{blo:g}_{bhi:g}_f{fp:g}",
                                    family="mad",
                                    noise_lo=lo,
                                    noise_hi=hi,
                                    bilat_lo=blo,
                                    bilat_hi=bhi,
                                    flat_pct=fp,
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
    return deploy_p68_noise(
        sota,
        edge,
        fps,
        noise_lo=recipe.noise_lo,
        noise_hi=recipe.noise_hi,
        bilat_lo=recipe.bilat_lo,
        bilat_hi=recipe.bilat_hi,
        u_lo=recipe.u_lo,
        u_hi=recipe.u_hi,
        low_fps=recipe.low_fps,
        noise_mode="mad",
        flat_pct=recipe.flat_pct,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p81_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    r = best.get("recipe") or {}
    dep = Path("nafnet_denoise/infer_blend_deploy.py")
    text = dep.read_text(encoding="utf-8")
    mapping = [
        ("noise_lo", r"noise_lo: float = [0-9.]+", "noise_lo"),
        ("noise_hi", r"noise_hi: float = [0-9.]+", "noise_hi"),
        ("bilat_lo", r"noise_bilat_lo: float = [0-9.]+", "noise_bilat_lo"),
        ("bilat_hi", r"noise_bilat_hi: float = [0-9.]+", "noise_bilat_hi"),
        ("u_lo", r"noise_u_lo: float = [0-9.]+", "noise_u_lo"),
        ("u_hi", r"noise_u_hi: float = [0-9.]+", "noise_u_hi"),
        ("flat_pct", r"noise_flat_pct: float = [0-9.]+", "noise_flat_pct"),
        ("low_fps", r"low_fps: float = [0-9.]+", "low_fps"),
    ]
    for key, pat, default in mapping:
        if key in r:
            text = re.sub(pat, f"{default}: float = {float(r[key])}", text, count=1)
    dep.write_text(text, encoding="utf-8")
    print(f"Baked P81 into deploy {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p81")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P81",
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
