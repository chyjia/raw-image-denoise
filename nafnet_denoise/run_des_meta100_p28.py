"""P28: DEU-style multi-geom self-ensemble of cached TTA views (mean/median).

Baseline: P22 DES ≈ 0.9290
Lit: DEU / DualEx self-ensemble — aggregate multiple geometric denoisings.
Uses existing ``*_sota_g_*`` / ``*_edge_g_*`` caches (zero re-forward).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from .des_cycle_common import (
    BASELINE,
    P19_TAG,
    available_geom_tags,
    ensemble_arms,
    load_arms,
    run_meta_loop,
    stamp_deploy_des,
)
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | ens
    tags: tuple[str, ...] = (P19_TAG,)
    mode: str = "mean"  # mean | median


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p22", family="baseline", tags=(P19_TAG,), mode="mean")]
    # Prefer known good tags that exist
    preferred = [
        P19_TAG,
        "id_r90_r180_r270",
        "id_lr_ud_udlr",
        "id_r90_r180_lr_ud",
        "id_r90_r270_lr_ud_udlr",
        "id_lr_ud_udlr_r90_lr_r270_lr",
        "id_lr_r90_r270_r90_lr_r270_lr",
        "id_ud_r90_r270_r90_lr_r270_lr",
        "id_r90_r180_r270_lr_ud",
        "id_lr_ud_udlr_r90_r180_r270_r90_lr",
        "id_r90_r270_r180_r90_lr",
        "id_lr_ud_udlr_r90_r180_r270_r90_lr_r270_lr_r90_ud",
    ]
    # pairs / triples / more
    combos = [
        (P19_TAG, "id_r90_r180_r270"),
        (P19_TAG, "id_lr_ud_udlr"),
        (P19_TAG, "id_r90_r180_lr_ud"),
        (P19_TAG, "id_r90_r270_lr_ud_udlr"),
        (P19_TAG, "id_lr_ud_udlr", "id_r90_r180_r270"),
        (P19_TAG, "id_lr_r90_r270_r90_lr_r270_lr"),
        (P19_TAG, "id_ud_r90_r270_r90_lr_r270_lr"),
        (
            P19_TAG,
            "id_lr_ud_udlr",
            "id_r90_r180_r270",
            "id_r90_r180_lr_ud",
        ),
        (
            P19_TAG,
            "id_lr_ud_udlr_r90_lr_r270_lr",
            "id_r90_r270_lr_ud_udlr",
        ),
        tuple(preferred[:5]),
        tuple(preferred[:7]),
        tuple(preferred[:9]),
    ]
    for tags in combos:
        for mode in ("mean", "median"):
            short = "+".join(t[:12] for t in tags[:3])
            recipes.append(
                Recipe(
                    name=f"ens_{mode}_{len(tags)}_{short}",
                    family="ens",
                    tags=tags,
                    mode=mode,
                )
            )
    # pad
    while len(recipes) < 100:
        i = len(recipes)
        tags = tuple(preferred[: 2 + (i % 6)])
        recipes.append(
            Recipe(
                name=f"pad_{i}_{'med' if i % 2 else 'mean'}",
                family="ens",
                tags=tags,
                mode="median" if i % 2 else "mean",
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(name=name, family=r.family, tags=r.tags, mode=r.mode)
        seen.add(name)
        out.append(r)
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    if recipe.family == "baseline":
        sota, edge = load_arms(cache_dir, stem, P19_TAG)
    else:
        sota, edge = ensemble_arms(cache_dir, stem, recipe.tags, recipe.mode)
    return deploy_p7_base(sota, edge, fps)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p28_hook.json").write_text(
        __import__("json").dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p28/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n"
        f"Wire multi-geom tags into deploy if baking permanently.\n"
        f"recipe={best.get('recipe')}\n",
        encoding="utf-8",
    )
    print("Wrote P28 hook (multi-geom; deploy stamp updated)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p28")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()

    # filter recipes to tags that exist
    have = set(available_geom_tags(args.cache_dir))
    have.add(P19_TAG)
    recipes = []
    for r in build_recipes():
        if r.family == "baseline" or all(t in have for t in r.tags):
            recipes.append(r)
        else:
            # drop missing tags
            tags = tuple(t for t in r.tags if t in have)
            if len(tags) >= 2:
                recipes.append(
                    Recipe(name=r.name + "_filt", family="ens", tags=tags, mode=r.mode)
                )
    recipes = recipes[:100]
    print(f"P28 recipes={len(recipes)} geom_tags={len(have)}", flush=True)

    run_meta_loop(
        cycle="P28",
        recipes=recipes,
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
