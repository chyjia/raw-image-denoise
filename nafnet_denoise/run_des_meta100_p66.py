"""P66: refine P64 exposure thresholds / bilat-unsharp (AdaptiveISP-style param search).

Baseline: P64 DES ≈ 0.9382
Lit: AdaptiveISP / MAS-ISP — scene-adaptive ISP hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p64_exposure


@dataclass
class Recipe:
    name: str
    family: str
    t_dark: float = 3.0
    t_bright: float = 15.0
    bilat_d: float = 0.95
    bilat_m: float = 0.88
    bilat_b: float = 0.80
    u_d: float = 0.22
    u_m: float = 0.15
    u_b: float = 0.14
    low_fps: float = 1.67


_EXP: dict[str, float] = {}


def exposure_of(cache_dir: Path, stem: str) -> float:
    if stem not in _EXP:
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _EXP[stem] = float(meta.get("exposure_ms", 10.0))
    return _EXP[stem]


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p64", family="baseline")]
    for td in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for tb in [12.0, 14.0, 15.0, 16.0, 18.0, 20.0]:
            if tb <= td:
                continue
            for bd in [0.93, 0.95, 0.97, 0.98]:
                for bm in [0.86, 0.88, 0.90, 0.92]:
                    for bb in [0.75, 0.80, 0.85]:
                        for ud, um, ub in [
                            (0.22, 0.15, 0.14),
                            (0.24, 0.16, 0.14),
                            (0.20, 0.14, 0.12),
                            (0.22, 0.16, 0.15),
                            (0.25, 0.15, 0.12),
                        ]:
                            recipes.append(
                                Recipe(
                                    name=f"e{td:g}_{tb:g}_b{bd:g}_{bm:g}_{bb:g}_u{ud:g}",
                                    family="exp",
                                    t_dark=td,
                                    t_bright=tb,
                                    bilat_d=bd,
                                    bilat_m=bm,
                                    bilat_b=bb,
                                    u_d=ud,
                                    u_m=um,
                                    u_b=ub,
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
    return deploy_p64_exposure(
        sota,
        edge,
        fps,
        exposure_of(cache_dir, stem),
        t_dark=recipe.t_dark,
        t_bright=recipe.t_bright,
        bilat_d=recipe.bilat_d,
        bilat_m=recipe.bilat_m,
        bilat_b=recipe.bilat_b,
        u_d=recipe.u_d,
        u_m=recipe.u_m,
        u_b=recipe.u_b,
        low_fps=recipe.low_fps,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p66_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    r = best.get("recipe") or {}
    dep = Path("nafnet_denoise/infer_blend_deploy.py")
    text = dep.read_text(encoding="utf-8")
    for key, pat, default in [
        ("t_dark", r"exp_t_dark: float = [0-9.]+", "exp_t_dark"),
        ("t_bright", r"exp_t_bright: float = [0-9.]+", "exp_t_bright"),
        ("bilat_d", r"exp_bilat_d: float = [0-9.]+", "exp_bilat_d"),
        ("bilat_m", r"exp_bilat_m: float = [0-9.]+", "exp_bilat_m"),
        ("bilat_b", r"exp_bilat_b: float = [0-9.]+", "exp_bilat_b"),
        ("u_d", r"exp_u_d: float = [0-9.]+", "exp_u_d"),
        ("u_m", r"exp_u_m: float = [0-9.]+", "exp_u_m"),
        ("u_b", r"exp_u_b: float = [0-9.]+", "exp_u_b"),
        ("low_fps", r"low_fps: float = [0-9.]+", "low_fps"),
    ]:
        if key in r:
            import re

            text = re.sub(pat, f"{default}: float = {float(r[key])}", text, count=1)
    dep.write_text(text, encoding="utf-8")
    print(f"Baked P66 into deploy {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p66")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P66",
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
