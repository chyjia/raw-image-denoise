"""P87: per-image self-supervised ISP param TTT (AdaptiveISP online lite).

Baseline: P74 DES ≈ 0.9442
For each clip, search a small bilat/unsharp offset that minimizes flat HP energy
while preserving edge energy (no GT). Then score with DES.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .des_cycle_common import BASELINE, P19_TAG, load_arms, run_meta_loop, stamp_deploy_des
from .p8_fusion import deploy_p68_noise, flat_noise_proxy


@dataclass
class Recipe:
    name: str
    family: str
    # search ranges for per-image offsets
    d_bilat: float = 0.05
    d_u: float = 0.03
    n_trials: int = 9
    edge_w: float = 0.5
    seed: int = 0


def _energy(img: np.ndarray) -> tuple[float, float]:
    g = cv2.GaussianBlur(img, (0, 0), 1.2)
    hp = img - g
    sob = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        g, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    flat = sob < float(np.percentile(sob, 40.0))
    edge = sob > float(np.percentile(sob, 70.0))
    flat_e = float(np.mean(hp[flat] ** 2)) if np.any(flat) else float(np.mean(hp**2))
    edge_e = float(np.mean(hp[edge] ** 2)) if np.any(edge) else flat_e
    return flat_e, edge_e


def ttt_isp(sota, edge, fps, r: Recipe) -> np.ndarray:
    base = deploy_p68_noise(sota, edge, fps)
    if r.family == "baseline" or r.n_trials <= 1:
        return base
    rng = np.random.default_rng(r.seed + hash(str(fps)) % 10007)
    best = base
    best_score = None
    n0 = flat_noise_proxy(0.5 * sota + 0.5 * edge)
    for i in range(int(r.n_trials)):
        db = float(rng.uniform(-r.d_bilat, r.d_bilat))
        du = float(rng.uniform(-r.d_u, r.d_u))
        # shift gate by remapping lo/hi slightly
        cand = deploy_p68_noise(
            sota,
            edge,
            fps,
            bilat_lo=float(np.clip(0.72 + db, 0.5, 0.95)),
            bilat_hi=float(np.clip(1.0 + db, 0.8, 1.0)),
            u_lo=float(np.clip(0.10 + du, 0.05, 0.2)),
            u_hi=float(np.clip(0.22 + du, 0.12, 0.3)),
        )
        flat_e, edge_e = _energy(cand)
        # lower flat energy good; keep edge energy
        score = flat_e - float(r.edge_w) * edge_e
        if best_score is None or score < best_score:
            best_score = score
            best = cand
    _ = n0  # reserved for future noise-conditioned priors
    return best.astype(np.float32)


def build_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p74", family="baseline")]
    for db in [0.03, 0.05, 0.08, 0.1]:
        for du in [0.02, 0.03, 0.05]:
            for nt in [5, 9, 15]:
                for ew in [0.3, 0.5, 0.8, 1.0]:
                    recipes.append(
                        Recipe(
                            name=f"ttt_b{db:g}_u{du:g}_n{nt}_e{ew:g}",
                            family="ttt",
                            d_bilat=db,
                            d_u=du,
                            n_trials=nt,
                            edge_w=ew,
                        )
                    )
    return recipes[:100]


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float):
    sota, edge = load_arms(cache_dir, stem, P19_TAG)
    return ttt_isp(sota, edge, fps, recipe)


def bake_fn(best: dict) -> None:
    Path("nafnet_denoise/deploy_p87_hook.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    stamp_deploy_des(float(best["mean_des"]))
    print(f"Wrote P87 hook {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p87")
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=BASELINE)
    args = parser.parse_args()
    run_meta_loop(
        cycle="P87",
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
