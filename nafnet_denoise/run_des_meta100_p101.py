"""P101: Anscombe fine-grid around P98 bake (s=0.2,k=0.5,sig=2,f50).

Baseline: P98 DES ≈ 0.9472
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p98


@dataclass
class Recipe:
    name: str
    family: str
    strength: float = 0.2
    k_mad: float = 0.5
    sigma: float = 2.0
    flat_pct: float = 50.0
    harden: float = 40.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p98", family="baseline")]
    # prioritize neighborhood of bake ans_s0.2_k0.5_sig2_f50 first
    for s in [0.2, 0.22, 0.18, 0.25, 0.15, 0.28, 0.12, 0.32]:
        for k in [0.5, 0.4, 0.6, 0.3, 0.7, 0.85]:
            for sig in [2.0, 2.2, 1.8, 2.5, 1.6, 3.0]:
                for fp in [50.0, 55.0, 45.0, 60.0, 40.0]:
                    recipes.append(
                        Recipe(
                            name=f"ansf_s{s:g}_k{k:g}_sig{sig:g}_f{fp:g}",
                            family="ans_fine",
                            strength=s,
                            k_mad=k,
                            sigma=sig,
                            flat_pct=fp,
                        )
                    )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            continue
        seen.add(name)
        out.append(r)
        if len(out) >= 100:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    return deploy_p98(
        sota,
        edge,
        fps,
        ans_strength=recipe.strength,
        ans_k_mad=recipe.k_mad,
        ans_sigma=recipe.sigma,
        ans_flat_pct=recipe.flat_pct,
        ans_harden=recipe.harden,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p101_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    # sync infer defaults via hook note; deploy_p98 kwargs updated in bake if needed
    print(f"Wrote P101 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p101")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P101",
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
