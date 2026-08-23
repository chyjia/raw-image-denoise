"""Generation-2: 100 recipes refined around mid-fps bilateral winner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .run_des_loop100 import (
    Recipe,
    apply_recipe,
    edge_safe,
    score_image,
)


def build_gen2_recipes() -> list[Recipe]:
    """Local search around mid_f5_ms0.5_ls0.65 + a few new axes."""
    recipes: list[Recipe] = []
    # baseline winner
    recipes.append(
        Recipe(
            name="g2_base_mid5_m0.5_l0.65",
            mid_fps=5.0,
            mid_bilat=0.5,
            low_bilat=0.65,
            bilat_harden=40.0,
        )
    )
    # mid strength x mid_fps grid
    for mid_fps in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        for mid_s in (0.35, 0.45, 0.5, 0.55, 0.6, 0.7):
            for low_s in (0.55, 0.65, 0.75):
                recipes.append(
                    Recipe(
                        name=f"g2_f{mid_fps:g}_m{mid_s:g}_l{low_s:g}",
                        mid_fps=mid_fps,
                        mid_bilat=mid_s,
                        low_bilat=low_s,
                        bilat_harden=40.0,
                    )
                )
    # harden variants on winner neighborhood
    for h in (24.0, 32.0, 40.0, 48.0, 56.0):
        for mid_s in (0.5, 0.55, 0.6):
            recipes.append(
                Recipe(
                    name=f"g2_h{h:g}_m{mid_s:g}",
                    mid_fps=5.0,
                    mid_bilat=mid_s,
                    low_bilat=0.65,
                    bilat_harden=h,
                )
            )
    # winner + mild detail transfer
    for amt in (0.25, 0.4, 0.55, 0.7):
        for mid_s in (0.45, 0.5, 0.55):
            recipes.append(
                Recipe(
                    name=f"g2_dt{amt:g}_m{mid_s:g}",
                    mid_fps=5.0,
                    mid_bilat=mid_s,
                    low_bilat=0.65,
                    dt_amount=amt,
                    dt_sigma=1.2,
                    dt_harden=16.0,
                    bilat_harden=40.0,
                )
            )
    # pad/truncate to 100
    seen: set[str] = set()
    uniq: list[Recipe] = []
    for r in recipes:
        if r.name in seen:
            continue
        seen.add(r.name)
        uniq.append(r)
    while len(uniq) < 100:
        i = len(uniq)
        uniq.append(
            Recipe(
                name=f"g2_pad_{i}",
                mid_fps=5.0 + (i % 5),
                mid_bilat=0.4 + 0.02 * (i % 10),
                low_bilat=0.65,
                bilat_harden=40.0,
            )
        )
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
        default=Path("nafnet_denoise/compare_loop100_gen2"),
    )
    args = parser.parse_args()

    metas = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(args.cache_dir.glob("*.json"))
    ]
    if len(metas) < 11:
        raise SystemExit(f"Expected cached metas under {args.cache_dir}")
    packs = []
    for meta in metas:
        stem = meta["stem"]
        packs.append(
            {
                "meta": meta,
                "sota": np.load(args.cache_dir / f"{stem}_sota.npy"),
                "edge": np.load(args.cache_dir / f"{stem}_edge.npy"),
            }
        )
    recipes = build_gen2_recipes()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_safe = None
    for idx, recipe in enumerate(recipes):
        des_list = []
        f10 = f05 = f1 = None
        F10, F05, F1 = "74824541", "53940814", "53741354"
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
                f"[{idx:03d}/100] NEW SAFE BEST {recipe.name} DES={mean_des:.4f}",
                flush=True,
            )
        else:
            print(
                f"[{idx:03d}/100] {recipe.name}: DES={mean_des:.4f} "
                f"({'ok' if ok else 'unsafe'})",
                flush=True,
            )
        if (idx + 1) % 20 == 0:
            (args.output_dir / "history.json").write_text(
                json.dumps({"history": history, "best_safe": best_safe}, indent=2),
                encoding="utf-8",
            )

    csv_path = args.output_dir / "metrics_by_recipe.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
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
    summary = {"best_safe": best_safe, "n": len(history)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"BEST_SAFE {best_safe['name']} mean={best_safe['mean_des']:.4f}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
