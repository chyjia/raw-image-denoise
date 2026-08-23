"""Generation-3: refine around g2_f5_m0.7_l0.75 until DES saturates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .run_des_loop100 import Recipe, apply_recipe, edge_safe, score_image


def build_gen3() -> list[Recipe]:
    recipes: list[Recipe] = []
    recipes.append(
        Recipe(
            name="g3_base",
            mid_fps=5.0,
            mid_bilat=0.7,
            low_bilat=0.75,
            bilat_harden=40.0,
        )
    )
    for mid_fps in (4.0, 5.0, 5.5, 6.0):
        for mid_s in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            for low_s in (0.7, 0.75, 0.8, 0.85):
                for h in (36.0, 40.0, 44.0):
                    recipes.append(
                        Recipe(
                            name=f"g3_f{mid_fps:g}_m{mid_s:g}_l{low_s:g}_h{h:g}",
                            mid_fps=mid_fps,
                            mid_bilat=mid_s,
                            low_bilat=low_s,
                            bilat_harden=h,
                        )
                    )
    seen: set[str] = set()
    uniq: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        uniq.append(r)
    return uniq[:100]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("nafnet_denoise/cache_dual_holdout"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_loop100_gen3"),
    )
    args = parser.parse_args()
    metas = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(args.cache_dir.glob("*.json"))
    ]
    packs = [
        {
            "meta": m,
            "sota": np.load(args.cache_dir / f"{m['stem']}_sota.npy"),
            "edge": np.load(args.cache_dir / f"{m['stem']}_edge.npy"),
        }
        for m in metas
    ]
    recipes = build_gen3()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_safe = None
    F10, F05, F1 = "74824541", "53940814", "53741354"
    for idx, recipe in enumerate(recipes):
        des_list = []
        f10 = f05 = f1 = None
        for pack in packs:
            meta = pack["meta"]
            out = apply_recipe(recipe, pack["sota"], pack["edge"], meta["fps"])
            sc = score_image(meta, args.cache_dir, recipe.name, out)
            des_list.append(sc["des"])
            if F10 in meta["file"]:
                f10 = sc
            if F05 in meta["file"]:
                f05 = sc
            if F1 in meta["file"]:
                f1 = sc
        mean_des = float(sum(des_list) / len(des_list))
        f10_ef = float(f10["edge_fidelity"])
        f05_ef = float(f05["edge_fidelity"])
        f05_ng = float(f05["noise_gain"])
        f1_ef = float(f1["edge_fidelity"])
        ok = edge_safe(f10_ef, f05_ef, f1_ef)
        rec = {
            "iter": idx,
            "name": recipe.name,
            "mean_des": mean_des,
            "f10_ef": f10_ef,
            "f05_ef": f05_ef,
            "f05_ng": f05_ng,
            "f1_ef": f1_ef,
            "edge_safe": ok,
            "recipe": asdict(recipe),
        }
        history.append(rec)
        if ok and (best_safe is None or mean_des > best_safe["mean_des"]):
            best_safe = rec
            print(
                f"[{idx:03d}/100] NEW SAFE BEST {recipe.name} DES={mean_des:.4f} "
                f"f05_ef={f05_ef:.4f} f1_ef={f1_ef:.4f}",
                flush=True,
            )
        else:
            print(
                f"[{idx:03d}/100] {recipe.name}: DES={mean_des:.4f} "
                f"({'ok' if ok else 'unsafe'})",
                flush=True,
            )
    with (args.output_dir / "metrics_by_recipe.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iter",
                "name",
                "mean_des",
                "f10_ef",
                "f05_ef",
                "f05_ng",
                "f1_ef",
                "edge_safe",
            ],
        )
        writer.writeheader()
        for rec in history:
            writer.writerow({k: rec[k] for k in writer.fieldnames})
    (args.output_dir / "summary.json").write_text(
        json.dumps({"best_safe": best_safe, "n": len(history)}, indent=2),
        encoding="utf-8",
    )
    print(
        f"BEST_SAFE {best_safe['name']} mean={best_safe['mean_des']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
