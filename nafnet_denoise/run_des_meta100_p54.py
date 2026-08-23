"""P54: SS-TTA / DualEx consistency — pull flats toward FE when |deploy−FE| large.



Baseline: P40 DES ≈ 0.9294

Lit: SS-TTA synth consistency; DualEx disagreement gate.

"""



from __future__ import annotations



import argparse

import json

from dataclasses import dataclass

from pathlib import Path



import numpy as np



from .des_cycle_common import BASELINE, load_arms, run_meta_loop, stamp_deploy_des

from .multiband_fuse import _edge_flat_maps

from .p8_fusion import deploy_p7_base





@dataclass

class Recipe:

    name: str

    family: str  # baseline | consist

    amount: float = 0.35

    thr: float = 0.02

    harden: float = 40.0

    only_low_mid: bool = False





def load_fe(cache_dir: Path, stem: str) -> np.ndarray | None:

    for name in (f"{stem}_fe.npy", f"{stem}_temporal.npy", f"{stem}_wiener.npy"):

        p = cache_dir / name

        if p.exists():

            return np.load(p).astype(np.float32)

    return None





def consist_pull(

    deploy: np.ndarray,

    fe: np.ndarray,

    amount: float,

    thr: float,

    harden: float,

) -> np.ndarray:

    img = deploy.astype(np.float32)

    fe = fe.astype(np.float32)

    if fe.shape != img.shape:

        return img

    _, flat = _edge_flat_maps(img, temperature=8.0, harden=harden)

    err = np.abs(img - fe)

    gate = np.clip((err - thr) / max(thr, 1e-4), 0.0, 1.0) * flat

    a = float(np.clip(amount, 0.0, 1.0))

    return (img * (1.0 - a * gate) + fe * (a * gate)).astype(np.float32)





def build_recipes() -> list[Recipe]:

    recipes = [Recipe(name="p40", family="baseline")]

    for amt in [0.15, 0.25, 0.35, 0.5, 0.65]:

        for thr in [0.008, 0.015, 0.025, 0.04, 0.06]:

            recipes.append(

                Recipe(

                    name=f"cons_a{amt:g}_t{thr:g}",

                    family="consist",

                    amount=amt,

                    thr=thr,

                )

            )

    while len(recipes) < 100:

        i = len(recipes)

        recipes.append(

            Recipe(

                name=f"pad54_c_{i}",

                family="consist",

                amount=0.1 + (i % 7) * 0.08,

                thr=0.01 + (i % 5) * 0.01,

            )

        )

    seen: set[str] = set()

    out: list[Recipe] = []

    for r in recipes:

        name = r.name if r.name not in seen else f"{r.name}_x{len(out)}"

        seen.add(name)

        out.append(

            Recipe(

                name=name,

                family=r.family,

                amount=r.amount,

                thr=r.thr,

                harden=r.harden,

                only_low_mid=r.only_low_mid,

            )

        )

    return out[:100]





def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):

    sota, edge = load_arms(cache_dir, stem)

    base = deploy_p7_base(sota, edge, fps)

    if recipe.family == "baseline":

        return base

    if recipe.only_low_mid and float(fps) > 5.0:

        return base

    fe = load_fe(cache_dir, stem)

    if fe is None:

        return base

    return consist_pull(base, fe, recipe.amount, recipe.thr, recipe.harden)





def bake_fn(best: dict) -> None:

    Path("nafnet_denoise/deploy_p54_hook.json").write_text(

        json.dumps(best, indent=2), encoding="utf-8"

    )

    stamp_deploy_des(float(best["mean_des"]))

    Path("nafnet_denoise/compare_meta100_p54").mkdir(parents=True, exist_ok=True)

    Path("nafnet_denoise/compare_meta100_p54/BAKE_NOTE.txt").write_text(

        f"Winner {best['name']} DES={best['mean_des']:.4f}\nFE consistency pull.\n",

        encoding="utf-8",

    )

    print(f"Wrote P54 hook {best['name']}", flush=True)





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))

    parser.add_argument(

        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")

    )

    parser.add_argument(

        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p54")

    )

    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--min-eval", type=int, default=16)

    parser.add_argument("--baseline", type=float, default=BASELINE)

    args = parser.parse_args()

    run_meta_loop(

        cycle="P54",

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


