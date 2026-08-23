import csv
import json
from pathlib import Path

from nafnet_denoise.run_des_loop100 import Recipe, apply_recipe, score_image

cache = Path("nafnet_denoise/cache_dual_holdout")
metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.json"))]
base = Recipe(name="p5")
win = Recipe(
    name="win",
    mid_fps=5.0,
    mid_bilat=0.7,
    low_bilat=0.75,
    bilat_harden=40.0,
)
for meta in metas:
    sota = np.load(cache / f"{meta['stem']}_sota.npy")
    edge = np.load(cache / f"{meta['stem']}_edge.npy")
    for name, rec in [("p5", base), ("win", win)]:
        out = apply_recipe(rec, sota, edge, meta["fps"])
        sc = score_image(meta, cache, name, out)
        if meta["fps"] <= 5.5 or "74824541" in meta["file"]:
            print(
                f"{meta['stem'][-30:]} fps={meta['fps']} {name}: "
                f"DES={sc['des']:.4f} ng={sc['noise_gain']:.4f} ef={sc['edge_fidelity']:.4f}"
            )
