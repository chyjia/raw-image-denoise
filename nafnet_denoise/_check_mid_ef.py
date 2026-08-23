import json
from pathlib import Path

import numpy as np

from nafnet_denoise.run_des_loop100 import Recipe, apply_recipe, score_image

cache = Path("nafnet_denoise/cache_dual_holdout")
metas = [
    json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(cache.glob("*.json"))
]
base = Recipe(name="p5", mid_fps=0.0, mid_bilat=0.0, low_bilat=0.65)
win = Recipe(
    name="win",
    mid_fps=5.0,
    mid_bilat=0.7,
    low_bilat=0.75,
    bilat_harden=40.0,
)
g3 = Recipe(
    name="g3",
    mid_fps=4.0,
    mid_bilat=0.85,
    low_bilat=0.85,
    bilat_harden=44.0,
)
for meta in metas:
    sota = np.load(cache / f"{meta['stem']}_sota.npy")
    edge = np.load(cache / f"{meta['stem']}_edge.npy")
    if meta["fps"] > 5.5 and "74824541" not in meta["file"]:
        continue
    print(f"\n{meta['stem'][-40:]} fps={meta['fps']}")
    for name, rec in [("p5", base), ("g2", win), ("g3", g3)]:
        out = apply_recipe(rec, sota, edge, meta["fps"])
        sc = score_image(meta, cache, name, out)
        print(
            f"  {name}: DES={sc['des']:.4f} ng={sc['noise_gain']:.4f} "
            f"ef={sc['edge_fidelity']:.4f}"
        )
