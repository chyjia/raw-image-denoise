"""P46: TTAD/Pixel2Pixel-inspired flat NL-means boost on P40 deploy.



Baseline: P40 DES ≈ 0.9294

Lit: ICCV'25 TTAD / Pixel2Pixel — non-local self-similarity pixel bank.

Zero-train proxy: edge-masked fastNlMeans on flats after P40 schedule.

Speed: half-res NLMeans + upsample (full-res is too slow for meta loop).

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

    family: str  # baseline | nl

    h: float = 3.0

    template: int = 5

    search: int = 11

    amount: float = 0.5

    harden: float = 40.0

    scale: float = 0.5

    only_low_mid: bool = True





def nl_flat_boost(

    image: np.ndarray,

    h: float,

    template: int,

    search: int,

    amount: float,

    harden: float,

    scale: float = 0.5,

) -> np.ndarray:

    img = np.ascontiguousarray(image, dtype=np.float32)

    lo, hi = float(img.min()), float(img.max())

    span = max(hi - lo, 1e-3)

    u8 = np.clip((img - lo) / span * 255.0, 0, 255).astype(np.uint8)

    sc = float(np.clip(scale, 0.25, 1.0))

    if sc < 0.999:

        small = cv2.resize(u8, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

        den_s = cv2.fastNlMeansDenoising(

            small,

            None,

            h=float(h),

            templateWindowSize=int(template),

            searchWindowSize=int(search),

        )

        den = cv2.resize(den_s, (u8.shape[1], u8.shape[0]), interpolation=cv2.INTER_LINEAR)

    else:

        den = cv2.fastNlMeansDenoising(

            u8,

            None,

            h=float(h),

            templateWindowSize=int(template),

            searchWindowSize=int(search),

        )

    den_f = den.astype(np.float32) / 255.0 * span + lo

    _, flat = _edge_flat_maps(img, temperature=8.0, harden=harden)

    a = float(np.clip(amount, 0.0, 1.0))

    return (img * (1.0 - a * flat) + den_f * (a * flat)).astype(np.float32)





def build_recipes() -> list[Recipe]:

    # Compact grid — NLMeans still costly even at half-res.

    recipes = [Recipe(name="p40", family="baseline")]

    for h in [2.0, 3.0, 4.0, 6.0]:

        for amt in [0.3, 0.5, 0.7]:

            for tw, sw in [(5, 11), (7, 15)]:

                recipes.append(

                    Recipe(

                        name=f"nl_h{h:g}_a{amt:g}_t{tw}_s{sw}",

                        family="nl",

                        h=h,

                        amount=amt,

                        template=tw,

                        search=sw,

                        scale=0.5,

                    )

                )

    recipes.append(

        Recipe(name="nl_full_h3_a0.4", family="nl", h=3.0, amount=0.4, template=5, search=11, scale=1.0)

    )

    while len(recipes) < 100:

        i = len(recipes)

        recipes.append(

            Recipe(

                name=f"pad_nl_{i}",

                family="nl",

                h=2.0 + (i % 5),

                amount=0.25 + (i % 4) * 0.15,

                template=5,

                search=11 + 2 * (i % 2),

                scale=0.5,

            )

        )

    seen: set[str] = set()

    out: list[Recipe] = []

    for r in recipes:

        name = r.name

        if name in seen:

            name = f"{name}_x{len(out)}"

            r = Recipe(

                name=name,

                family=r.family,

                h=r.h,

                template=r.template,

                search=r.search,

                amount=r.amount,

                harden=r.harden,

                scale=r.scale,

                only_low_mid=r.only_low_mid,

            )

        seen.add(name)

        out.append(r)

    return out[:100]





def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):

    sota, edge = load_arms(cache_dir, stem)

    base = deploy_p7_base(sota, edge, fps)

    if recipe.family == "baseline":

        return base

    if recipe.only_low_mid and float(fps) > 5.0:

        return base

    return nl_flat_boost(

        base,

        recipe.h,

        recipe.template,

        recipe.search,

        recipe.amount,

        recipe.harden,

        scale=recipe.scale,

    )





def bake_fn(best: dict) -> None:

    Path("nafnet_denoise/deploy_p46_hook.json").write_text(

        json.dumps(best, indent=2), encoding="utf-8"

    )

    stamp_deploy_des(float(best["mean_des"]))

    note = Path("nafnet_denoise/compare_meta100_p46/BAKE_NOTE.txt")

    note.parent.mkdir(parents=True, exist_ok=True)

    note.write_text(

        f"Winner {best['name']} DES={best['mean_des']:.4f}\nNL flat boost — wire into deploy if kept.\n",

        encoding="utf-8",

    )

    print(f"Wrote P46 hook {best['name']}", flush=True)





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))

    parser.add_argument(

        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")

    )

    parser.add_argument(

        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p46")

    )

    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--min-eval", type=int, default=16)

    parser.add_argument("--baseline", type=float, default=BASELINE)

    args = parser.parse_args()

    run_meta_loop(

        cycle="P46",

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


