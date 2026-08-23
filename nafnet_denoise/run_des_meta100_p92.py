"""P92: UMGF structure transfer post on P74 MAD deploy (TIP'21).

Baseline: P74 DES ≈ 0.9442
Lit: Unsharp-Mask Guided Filtering — edge-arm HF transfer after noise-gated deploy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise, umgf_fuse, gh_gif


@dataclass
class Recipe:
    name: str
    family: str
    mode: str = "umgf"  # umgf | ghgif
    amount: float = 0.0
    sigma: float = 1.4
    harden: float = 16.0
    mix: float = 1.0  # blend with base


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for mode in ["umgf", "ghgif"]:
        for amt in [0.08, 0.12, 0.18, 0.25, 0.35, 0.5]:
            for sig in [0.8, 1.2, 1.6, 2.0, 2.5]:
                for mix in [0.5, 0.75, 1.0]:
                    for h in [8.0, 16.0, 24.0]:
                        recipes.append(
                            Recipe(
                                name=f"{mode}_a{amt:g}_s{sig:g}_m{mix:g}_h{h:g}",
                                family=mode,
                                mode=mode,
                                amount=amt,
                                sigma=sig,
                                harden=h,
                                mix=mix,
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
    base = deploy_p68_noise(sota, edge, fps)
    if recipe.family == "baseline" or recipe.amount <= 0:
        return base
    if recipe.mode == "ghgif":
        fused = gh_gif(
            base, edge, amount=recipe.amount, sigma=recipe.sigma, harden=recipe.harden
        )
    else:
        fused = umgf_fuse(
            base, edge, amount=recipe.amount, sigma=recipe.sigma, harden=recipe.harden
        )
    m = float(recipe.mix)
    return ((1.0 - m) * base + m * fused).astype("float32")


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p92_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P92 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p92")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P92",
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
