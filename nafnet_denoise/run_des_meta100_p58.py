"""P58: ADFNet-lite spatial↔frequency adaptive gate on P40 arms.

Baseline: P40 DES ≈ 0.9294
Lit: ADFNet (2025) — fuse spatial prediction with frequency-domain mix via
adaptive gate. Here: soft blend (spatial) vs Fourier LF/HF fuse, then gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .infer_ensemble import blend_sota_edgekd
from .p8_fusion import deploy_p7_base


@dataclass
class Recipe:
    name: str
    family: str  # baseline | adf
    keep_lf: float = 0.15
    edge_hf: float = 0.85
    gate_k: float = 8.0
    gate_bias: float = 0.3  # prefer spatial when soft-edge high
    spatial_weight: float = 0.5  # mix after gate mean


def fourier_fuse(sota, edge, keep_lf: float, edge_hf: float) -> np.ndarray:
    h, w = sota.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(np.sqrt(cy**2 + cx**2)) + 1e-6
    lf = (r <= float(keep_lf) * r_max).astype(np.float32)
    hf = 1.0 - lf
    Fs = np.fft.fftshift(np.fft.fft2(sota.astype(np.float32)))
    Fe = np.fft.fftshift(np.fft.fft2(edge.astype(np.float32)))
    ehf = float(np.clip(edge_hf, 0.0, 1.0))
    F = Fs * lf + ((1.0 - ehf) * Fs + ehf * Fe) * hf
    out = np.fft.ifft2(np.fft.ifftshift(F)).real
    return out.astype(np.float32)


def adf_fuse(sota, edge, r: Recipe) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    spatial, emap = blend_sota_edgekd(
        sota, edge, guide_dn=guide, temperature=8.0, harden=16.0
    )
    freq = fourier_fuse(sota, edge, r.keep_lf, r.edge_hf)
    # gate: high edge map → spatial; flats → more freq (or vice versa via bias)
    g = 1.0 / (1.0 + np.exp(-float(r.gate_k) * (emap - float(r.gate_bias))))
    # g~1 on edges → spatial
    fused = g * spatial + (1.0 - g) * freq
    sw = float(np.clip(r.spatial_weight, 0.0, 1.0))
    return ((1.0 - sw) * fused + sw * spatial).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p40", family="baseline")]
    for lf in [0.08, 0.12, 0.15, 0.2, 0.25]:
        for ehf in [0.7, 0.85, 1.0]:
            for bias in [0.2, 0.35, 0.5]:
                for sw in [0.3, 0.5, 0.7]:
                    recipes.append(
                        Recipe(
                            name=f"adf_lf{lf:g}_ehf{ehf:g}_b{bias:g}_sw{sw:g}",
                            family="adf",
                            keep_lf=lf,
                            edge_hf=ehf,
                            gate_bias=bias,
                            spatial_weight=sw,
                        )
                    )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_adf_{i}",
                family="adf",
                keep_lf=0.1 + (i % 5) * 0.03,
                edge_hf=0.65 + (i % 4) * 0.1,
                gate_bias=0.15 + (i % 5) * 0.08,
                spatial_weight=0.2 + (i % 6) * 0.1,
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        n = r.name if r.name not in seen else f"{r.name}_x{len(out)}"
        seen.add(n)
        out.append(Recipe(**{**r.__dict__, "name": n}))
    return out[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    # Replace dual arms with ADF fused image, then P40 post schedule
    fused = adf_fuse(sota, edge, recipe)
    return deploy_p7_base(fused, fused, fps)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p58_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    note = Path("nafnet_denoise/compare_meta100_p58/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\n{best.get('recipe')}\n",
        encoding="utf-8",
    )
    print(f"Wrote P58 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p58")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P58",
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
