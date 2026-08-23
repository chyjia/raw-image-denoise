"""P52: intensity-bin EnsIR-lite fuse of SOTA↔edge arms then P40 schedule.



Baseline: P40 DES ≈ 0.9294

Lit: NeurIPS'24 EnsIR — range-wise ensemble weights (LUT) without GT:

heuristic: prefer quieter arm (lower local highpass energy) per intensity bin.

"""



from __future__ import annotations



import argparse

import json

from dataclasses import dataclass

from pathlib import Path



import cv2

import numpy as np



from .des_cycle_common import BASELINE, load_arms, run_meta_loop, stamp_deploy_des

from .infer_ensemble import blend_sota_edgekd

from .multiband_fuse import flat_bilateral_boost

from .p7_fusion import edge_unsharp

from .p8_fusion import deploy_p7_base





@dataclass

class Recipe:

    name: str

    family: str  # baseline | ens

    bins: int = 8

    quiet_bias: float = 0.65  # weight toward quieter arm

    low_fps: float = 1.67

    mid_fps: float = 5.0

    u_low: float = 0.20

    u_mid: float = 0.16

    u_high: float = 0.12

    low_bilat: float = 0.9

    mid_bilat: float = 0.85





def local_hp_energy(x: np.ndarray, sigma: float = 1.2) -> np.ndarray:

    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma)

    return cv2.GaussianBlur(np.abs(x - blur), (0, 0), sigmaX=2.0)





def ensir_lite_fuse(sota: np.ndarray, edge: np.ndarray, bins: int, quiet_bias: float) -> np.ndarray:

    s = sota.astype(np.float32)

    e = edge.astype(np.float32)

    guide = 0.5 * (s + e)

    lo, hi = float(guide.min()), float(guide.max())

    span = max(hi - lo, 1e-6)

    bin_id = np.clip(((guide - lo) / span * bins).astype(np.int32), 0, bins - 1)

    es = local_hp_energy(s)

    ee = local_hp_energy(e)

    out = s.copy()

    qb = float(np.clip(quiet_bias, 0.0, 1.0))

    for b in range(bins):

        m = bin_id == b

        if not np.any(m):

            continue

        # mean energy in bin → arm weight

        ms = float(es[m].mean())

        me = float(ee[m].mean())

        # quieter gets qb, louder gets 1-qb

        if ms <= me:

            ws, we = qb, 1.0 - qb

        else:

            ws, we = 1.0 - qb, qb

        out[m] = ws * s[m] + we * e[m]

    return out.astype(np.float32)





def deploy_ens(sota, edge, fps, r: Recipe) -> np.ndarray:

    fused = ensir_lite_fuse(sota, edge, r.bins, r.quiet_bias)

    # still apply fps schedule post filters like P40, but start from fused

    guide = fused

    if float(fps) <= r.low_fps:

        out = fused.copy()

        out = flat_bilateral_boost(out, guide=out, flat_strength=r.low_bilat, harden=40.0)

        u = r.u_low

    elif float(fps) <= r.mid_fps:

        # re-blend fused with edge for mid structure

        out, _ = blend_sota_edgekd(

            fused, edge, guide_dn=guide, temperature=8.0, harden=16.0

        )

        out = flat_bilateral_boost(out, guide=out, flat_strength=r.mid_bilat, harden=40.0)

        u = r.u_mid

    else:

        out, _ = blend_sota_edgekd(

            fused, edge, guide_dn=guide, temperature=8.0, harden=16.0

        )

        u = r.u_high

    if u > 0:

        out = edge_unsharp(out, amount=u, sigma=1.4, harden=16.0)

    return out.astype(np.float32)





def build_recipes() -> list[Recipe]:

    recipes = [Recipe(name="p40", family="baseline")]

    for bins in [4, 6, 8, 12, 16]:

        for qb in [0.55, 0.65, 0.75, 0.85, 0.9]:

            recipes.append(

                Recipe(

                    name=f"ens_b{bins}_q{qb:g}",

                    family="ens",

                    bins=bins,

                    quiet_bias=qb,

                )

            )

    while len(recipes) < 100:

        i = len(recipes)

        recipes.append(

            Recipe(

                name=f"pad_ens_{i}",

                family="ens",

                bins=4 + (i % 6) * 2,

                quiet_bias=0.5 + (i % 5) * 0.08,

                u_low=0.18 + (i % 4) * 0.01,

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

                quiet_bias=r.quiet_bias,

                low_fps=r.low_fps,

                mid_fps=r.mid_fps,

                u_low=r.u_low,

                u_mid=r.u_mid,

                u_high=r.u_high,

                low_bilat=r.low_bilat,

                mid_bilat=r.mid_bilat,

            )

        )

    return out[:100]





def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):

    sota, edge = load_arms(cache_dir, stem)

    if recipe.family == "baseline":

        return deploy_p7_base(sota, edge, fps)

    return deploy_ens(sota, edge, fps, recipe)





def bake_fn(best: dict) -> None:

    Path("nafnet_denoise/deploy_p52_hook.json").write_text(

        json.dumps(best, indent=2), encoding="utf-8"

    )

    stamp_deploy_des(float(best["mean_des"]))

    Path("nafnet_denoise/compare_meta100_p52").mkdir(parents=True, exist_ok=True)

    Path("nafnet_denoise/compare_meta100_p52/BAKE_NOTE.txt").write_text(

        f"Winner {best['name']} DES={best['mean_des']:.4f}\nEnsIR-lite quiet-bin fuse.\n",

        encoding="utf-8",

    )

    print(f"Wrote P52 hook {best['name']}", flush=True)





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))

    parser.add_argument(

        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")

    )

    parser.add_argument(

        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p52")

    )

    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--min-eval", type=int, default=16)

    parser.add_argument("--baseline", type=float, default=BASELINE)

    args = parser.parse_args()

    run_meta_loop(

        cycle="P52",

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


