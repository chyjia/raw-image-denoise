"""P71: refine P68 noise-gate thresholds (AdaptiveISP param search on RPG-VST-lite).

Baseline: P68 DES ≈ 0.9438
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
    noise_lo: float = 0.005
    noise_hi: float = 0.02
    bilat_lo: float = 0.75
    bilat_hi: float = 0.98
    u_lo: float = 0.10
    u_hi: float = 0.22
    low_fps: float = 1.67


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p68", family="baseline")]
    for lo in [0.003, 0.004, 0.005, 0.006, 0.008]:
        for hi in [0.015, 0.018, 0.02, 0.022, 0.025, 0.03]:
            if hi <= lo:
                continue
            for blo, bhi in [
                (0.70, 0.98),
                (0.75, 0.98),
                (0.75, 1.0),
                (0.78, 0.96),
                (0.72, 0.95),
                (0.80, 0.98),
            ]:
                for ulo, uhi in [
                    (0.08, 0.20),
                    (0.10, 0.22),
                    (0.10, 0.24),
                    (0.12, 0.22),
                    (0.09, 0.21),
                ]:
                    for lf in [1.5, 1.67, 2.0]:
                        recipes.append(
                            Recipe(
                                name=f"n{lo:g}_{hi:g}_b{blo:g}_{bhi:g}_u{ulo:g}_f{lf:g}",
                                family="noise",
                                noise_lo=lo,
                                noise_hi=hi,
                                bilat_lo=blo,
                                bilat_hi=bhi,
                                u_lo=ulo,
                                u_hi=uhi,
                                low_fps=lf,
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
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p71_hook.json").write_text(
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
        ("low_fps", r"low_fps: float = [0-9.]+", "low_fps"),
    ]
    for key, pat, default in mapping:
        if key in r:
            text = re.sub(pat, f"{default}: float = {float(r[key])}", text, count=1)
    dep.write_text(text, encoding="utf-8")
    print(f"Baked P71 into deploy {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p71")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P71",
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
