"""P55: Pixel2Pixel/TTAD pixel-bank flat median (fast intensity-bin proxy).



Baseline: P40 DES ≈ 0.9294

Lit: non-local similar-pixel bank → aggregate in flats.

Fast proxy: per intensity bin, replace flat pixels with bin median (global) /

local median mix — O(N) vs patch-search.

"""



from __future__ import annotations



import argparse

import json

from dataclasses import dataclass

from pathlib import Path



import cv2

import numpy as np



from .des_cycle_common import BASELINE, load_arms, run_meta_loop, stamp_deploy_des

from .multiband_fuse import _edge_flat_maps

from .p8_fusion import deploy_p7_base





@dataclass

class Recipe:

    name: str

    family: str  # baseline | bank

    bins: int = 32

    local_k: int = 5  # odd medianBlur

    amount: float = 0.55

    local_mix: float = 0.5  # mix bin-median vs local-median

    harden: float = 40.0

    only_low_mid: bool = True





def bank_flat_boost(

    image: np.ndarray,

    bins: int,

    local_k: int,

    amount: float,

    local_mix: float,

    harden: float,

) -> np.ndarray:

    img = np.ascontiguousarray(image, dtype=np.float32)

    lo, hi = float(img.min()), float(img.max())

    span = max(hi - lo, 1e-3)

    u8 = np.clip((img - lo) / span * 255.0, 0, 255).astype(np.uint8)

    k = int(local_k) if int(local_k) % 2 == 1 else int(local_k) + 1

    local = cv2.medianBlur(u8, k).astype(np.float32)

    f = u8.astype(np.float32)

    nb = max(4, int(bins))

    bin_id = np.clip((f / 255.0 * nb).astype(np.int32), 0, nb - 1)

    bin_med = np.empty(nb, dtype=np.float32)

    for b in range(nb):

        m = bin_id == b

        bin_med[b] = float(np.median(f[m])) if np.any(m) else float(b) * 255.0 / nb

    global_med = bin_med[bin_id]

    lm = float(np.clip(local_mix, 0.0, 1.0))

    den = (1.0 - lm) * global_med + lm * local

    den_f = den / 255.0 * span + lo

    _, flat = _edge_flat_maps(img, temperature=8.0, harden=harden)

    a = float(np.clip(amount, 0.0, 1.0))

    return (img * (1.0 - a * flat) + den_f * (a * flat)).astype(np.float32)





def build_recipes() -> list[Recipe]:

    recipes = [Recipe(name="p40", family="baseline")]

    for bins in [16, 24, 32, 48, 64]:

        for amt in [0.3, 0.45, 0.6, 0.75]:

            for lk in [3, 5, 7]:

                for lm in [0.25, 0.5, 0.75]:

                    recipes.append(

                        Recipe(

                            name=f"bank_b{bins}_a{amt:g}_k{lk}_m{lm:g}",

                            family="bank",

                            bins=bins,

                            amount=amt,

                            local_k=lk,

                            local_mix=lm,

                        )

                    )

    while len(recipes) < 100:

        i = len(recipes)

        recipes.append(

            Recipe(

                name=f"pad55_bank_{i}",

                family="bank",

                bins=16 + (i % 6) * 8,

                amount=0.25 + (i % 5) * 0.1,

                local_k=3 + 2 * (i % 3),

                local_mix=0.2 + (i % 4) * 0.2,

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

                bins=r.bins,

                local_k=r.local_k,

                amount=r.amount,

                local_mix=r.local_mix,

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

    return bank_flat_boost(

        base,

        recipe.bins,

        recipe.local_k,

        recipe.amount,

        recipe.local_mix,

        recipe.harden,

    )





def bake_fn(best: dict) -> None:

    Path("nafnet_denoise/deploy_p55_hook.json").write_text(

        json.dumps(best, indent=2), encoding="utf-8"

    )

    stamp_deploy_des(float(best["mean_des"]))

    note = Path("nafnet_denoise/compare_meta100_p55/BAKE_NOTE.txt")

    note.parent.mkdir(parents=True, exist_ok=True)

    note.write_text(

        f"Winner {best['name']} DES={best['mean_des']:.4f}\npixel-bank flat median.\n",

        encoding="utf-8",

    )

    print(f"Wrote P55 hook {best['name']}", flush=True)





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))

    parser.add_argument(

        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")

    )

    parser.add_argument(

        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p55")

    )

    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--min-eval", type=int, default=16)

    parser.add_argument("--baseline", type=float, default=BASELINE)

    args = parser.parse_args()

    run_meta_loop(

        cycle="P55",

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


