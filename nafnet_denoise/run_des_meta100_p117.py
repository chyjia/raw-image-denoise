"""P117: Haar DWT cycle-spin shrink on flats after P103 (Parhi LSP'23 / EURASIP'25).

Baseline: P103 DES ≈ 0.9506
P109 8x8 WHT failed; Haar + few shifts is a different dictionary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p103_cne, haar_cycle_spin_shrink


@dataclass
class Recipe:
    name: str
    family: str
    strength: float = 0.0
    k_mad: float = 1.0
    max_shift: int = 1
    flat_pct: float = 50.0


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p103", family="baseline")]
    for s in [0.08, 0.12, 0.18, 0.25, 0.35]:
        for k in [0.5, 0.75, 1.0, 1.25, 1.5]:
            for ms in [0, 1, 2]:
                for fp in [40.0, 50.0, 60.0]:
                    recipes.append(
                        Recipe(
                            name=f"haar_s{s:g}_k{k:g}_m{ms}_f{fp:g}",
                            family="haar",
                            strength=s,
                            k_mad=k,
                            max_shift=ms,
                            flat_pct=fp,
                        )
                    )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        out.append(r)
        if len(out) >= 100:
            break
    return out


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    base = deploy_p103_cne(sota, edge, fps)
    if recipe.family == "baseline" or recipe.strength <= 0:
        return base
    return haar_cycle_spin_shrink(
        base,
        strength=recipe.strength,
        k_mad=recipe.k_mad,
        max_shift=recipe.max_shift,
        flat_pct=recipe.flat_pct,
    )


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p117_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P113 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p117")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P117",
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
