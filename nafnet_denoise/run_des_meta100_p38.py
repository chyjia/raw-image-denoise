"""P38: leave-one-clip-out EnsIR intensity LUT for SOTA↔edge weights.

Baseline: P31 DES ≈ 0.9292
Lit: EnsIR (NeurIPS'24) — range-wise ensemble weights via LUT.
Fit on other holdout clips' temporal refs (LOO) to avoid self-label gaming.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p7_base
from .validate import aligned_temporal_trimmed_mean, central_roi
from .common import memmap_frames, parse_geometry


@dataclass
class Recipe:
    name: str
    family: str
    bins: int = 16
    # fallback parametric if LOO fails
    edge_lo: float = 0.2
    edge_hi: float = 0.9


def fit_lut(
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    bins: int,
) -> np.ndarray:
    """Fit per-bin edge weight minimizing | (1-w)*sota + w*edge - ref |."""
    sums = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.float64)
    for sota, edge, ref in pairs:
        guide = 0.5 * sota + 0.5 * edge
        gmin, gmax = float(guide.min()), float(guide.max())
        norm = (guide - gmin) / (gmax - gmin + 1e-6)
        idx = np.clip((norm * bins).astype(np.int32), 0, bins - 1)
        # closed-form: for each pixel, optimal w = proj of (ref-sota) onto (edge-sota)
        den = edge - sota
        num = ref - sota
        # avoid div0
        w_pix = np.where(np.abs(den) > 1e-6, num / den, 0.5)
        w_pix = np.clip(w_pix, 0.0, 1.0)
        for b in range(bins):
            m = idx == b
            if np.any(m):
                sums[b] += float(w_pix[m].mean())
                counts[b] += 1.0
    lut = np.full(bins, 0.5, dtype=np.float32)
    ok = counts > 0
    lut[ok] = (sums[ok] / counts[ok]).astype(np.float32)
    # smooth
    if bins >= 3:
        lut = np.convolve(lut, np.ones(3) / 3.0, mode="same").astype(np.float32)
    return lut


def apply_lut(sota, edge, lut: np.ndarray) -> np.ndarray:
    bins = len(lut)
    guide = 0.5 * sota + 0.5 * edge
    gmin, gmax = float(guide.min()), float(guide.max())
    norm = (guide - gmin) / (gmax - gmin + 1e-6)
    idx = np.clip((norm * bins).astype(np.int32), 0, bins - 1)
    w = lut[idx]
    return ((1.0 - w) * sota + w * edge).astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p31", family="baseline")]
    for bins in [8, 12, 16, 24, 32]:
        recipes.append(Recipe(name=f"loo_b{bins}", family="loo", bins=bins))
    for bins, lo, hi in [
        (16, 0.1, 0.9),
        (16, 0.2, 0.8),
        (24, 0.15, 0.85),
        (12, 0.0, 1.0),
    ]:
        recipes.append(
            Recipe(name=f"lin_b{bins}_{lo:g}_{hi:g}", family="linear", bins=bins, edge_lo=lo, edge_hi=hi)
        )
    while len(recipes) < 40:
        i = len(recipes)
        recipes.append(Recipe(name=f"pad_loo_{i}", family="loo", bins=8 + (i % 5) * 4))
    return recipes[:40]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout"))
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p38"))
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-eval", type=int, default=12)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]

    # Preload arms + temporal refs for LOO
    arms = {}
    refs = {}
    for path, meta in zip(files, metas):
        stem = path.stem
        arms[stem] = load_arms(args.cache_dir, stem)
        # temporal from cache if present else compute
        tp = args.cache_dir / f"{stem}_temporal.npy"
        if tp.exists():
            refs[stem] = np.load(tp)
        else:
            width, height, _fps = parse_geometry(path.name)
            frames = memmap_frames(path, width, height)
            target = frames.shape[0] // 2
            ys, xs = central_roi(height, width)
            refs[stem] = aligned_temporal_trimmed_mean(
                frames, target, ys, xs, frames.shape[0], min(16, frames.shape[0]),
                trim_fraction=0.10, exclude_frame_index=target,
            )
            np.save(tp, refs[stem])

    stems = [p.stem for p in files]

    def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
        sota, edge = arms[stem]
        if recipe.family == "baseline":
            return deploy_p7_base(sota, edge, fps)
        if recipe.family == "linear":
            lut = np.linspace(recipe.edge_lo, recipe.edge_hi, recipe.bins, dtype=np.float32)
            fused = apply_lut(sota, edge, lut)
            return deploy_p7_base(fused, fused, fps)
        # LOO fit
        pairs = []
        for other in stems:
            if other == stem:
                continue
            so, eo = arms[other]
            pairs.append((so, eo, refs[other]))
        lut = fit_lut(pairs, recipe.bins)
        fused = apply_lut(sota, edge, lut)
        return deploy_p7_base(fused, fused, fps)

    run_meta_loop(
        cycle="P38",
        recipes=build_recipes(),
        apply_fn=apply_recipe,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        baseline=float(args.baseline),
        patience=int(args.patience),
        min_eval=int(args.min_eval),
        bake_fn=lambda best: (
            Path("nafnet_denoise/deploy_p38_hook.json").write_text(
                json.dumps(best, indent=2), encoding="utf-8"
            ),
            stamp_deploy_des(float(best["mean_des"])),
            print(f"P38 hook {best['name']}", flush=True),
        ),
    )


if __name__ == "__main__":
    main()
