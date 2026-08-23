"""Meta DES loop: lit-family catalogs × up to 100 generations.

Each generation:
  1) Build 100 recipes from literature families (rotate + refine winners)
  2) Score on cached dual holdout (edge-safe)
  3) Keep global best; bake note if improved
  4) Early-stop after ``patience`` gens with ΔDES < 1e-4 (default)

Does not re-push bilateral strength (known DES gaming).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .run_des_loop100 import F1, F05, F10, cache_forwards, edge_safe, score_image
from .run_des_loop100_p7 import Recipe, apply_recipe, build_100_recipes


LIT_FAMILIES = [
    "residual",  # DualExNet |Δ| gate
    "variance",  # SNR-aware
    "freq",  # HF/LF split
    "morph",  # morph soft gate
    "gen2_post",  # guided/median/gauss/unsharp/tta
]


def refine_around(best: Recipe | None, gen: int, n: int = 100) -> list[Recipe]:
    """Generation catalog: seed P7, then local refine + rotate families."""
    if gen == 0:
        return build_100_recipes()[:n]

    recipes: list[Recipe] = []
    # always include current global best + gen2
    recipes.append(Recipe(name="gen2_deploy", family="gen2"))
    if best is not None:
        recipes.append(
            Recipe(**{**asdict(best), "name": f"seed_{best.name}"[:60]})
        )

    fam = LIT_FAMILIES[gen % len(LIT_FAMILIES)]
    # Local refine around best if same family, else explore fam
    base = best if (best and best.family == fam) else Recipe(name="tmp", family=fam)

    if fam == "residual":
        for i in range(40):
            recipes.append(
                Recipe(
                    name=f"g{gen}_res_{i}",
                    family="residual",
                    gate_temp=max(4.0, float(base.gate_temp) + (i % 9 - 4) * 1.0),
                    gate_harden=max(4.0, float(base.gate_harden) + (i % 5 - 2) * 4.0),
                    gate_mix=float(np.clip(base.gate_mix + (i % 7 - 3) * 0.05, 0.3, 1.0)),
                    gate_pct=float(np.clip(base.gate_pct + (i % 5 - 2) * 2.0, 70.0, 98.0)),
                    mid_bilat=float(np.clip(0.55 + (i % 5) * 0.05, 0.4, 0.75)),
                    apply_gen2_bilat=True,
                )
            )
    elif fam == "variance":
        for i in range(40):
            recipes.append(
                Recipe(
                    name=f"g{gen}_var_{i}",
                    family="variance",
                    var_win=[5, 7, 9, 11][i % 4],
                    gate_temp=8.0 + (i % 6),
                    edge_bias=0.25 + (i % 8) * 0.08,
                    gate_pct=75.0 + (i % 5) * 4.0,
                    apply_gen2_bilat=True,
                )
            )
    elif fam == "freq":
        for i in range(40):
            recipes.append(
                Recipe(
                    name=f"g{gen}_freq_{i}",
                    family="freq",
                    freq_sigma=0.8 + (i % 10) * 0.25,
                    high_edge=0.55 + (i % 9) * 0.05,
                    low_edge=(i % 4) * 0.05,
                    mid_bilat=0.55 + (i % 5) * 0.04,
                    apply_gen2_bilat=True,
                )
            )
    elif fam == "morph":
        for i in range(30):
            recipes.append(
                Recipe(
                    name=f"g{gen}_morph_{i}",
                    family="morph",
                    dilate=i % 4,
                    erode=(i // 4) % 3,
                    harden=12.0 + (i % 5) * 4.0,
                    apply_gen2_bilat=True,
                )
            )
    else:  # gen2_post
        posts = [
            ("guided", 0.25 + (i % 8) * 0.05, {"radius": 3 + (i % 4), "eps": 1e-3})
            for i in range(15)
        ] + [
            ("unsharp", 0.15 + (i % 8) * 0.05, {"sigma": 0.8 + (i % 5) * 0.2})
            for i in range(15)
        ] + [
            ("median", 0.25 + (i % 6) * 0.05, {"ksize": 3 if i % 2 == 0 else 5})
            for i in range(10)
        ]
        for i, (post, strength, kw) in enumerate(posts):
            recipes.append(
                Recipe(
                    name=f"g{gen}_{post}_{i}",
                    family="gen2_post",
                    post=post,
                    post_strength=strength,
                    post_kw=kw,
                )
            )

    # Cross hybrids fill to n
    i = 0
    while len(recipes) < n:
        recipes.append(
            Recipe(
                name=f"g{gen}_hy_{i}",
                family="residual",
                gate_temp=10.0 + (i % 8),
                gate_mix=0.5 + (i % 10) * 0.05,
                post="unsharp" if i % 3 == 0 else "none",
                post_strength=0.2 if i % 3 == 0 else 0.0,
                post_kw={"sigma": 1.0} if i % 3 == 0 else {},
                apply_gen2_bilat=True,
            )
        )
        i += 1

    # unique names
    seen: set[str] = set()
    uniq: list[Recipe] = []
    for r in recipes[:n]:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(uniq)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        uniq.append(r)
    return uniq[:n]


def run_generation(
    packs: list[dict],
    cache_dir: Path,
    recipes: list[Recipe],
    out_dir: Path,
    gen: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict | None = None
    best_any: dict | None = None
    for idx, recipe in enumerate(recipes):
        des_list: list[float] = []
        f10 = f05 = f1 = None
        for pack in packs:
            meta = pack["meta"]
            out = apply_recipe(recipe, pack["sota"], pack["edge"], meta["fps"])
            sc = score_image(meta, cache_dir, recipe.name, out)
            des_list.append(sc["des"])
            if F10 in meta["file"]:
                f10 = sc
            if F05 in meta["file"]:
                f05 = sc
            if F1 in meta["file"]:
                f1 = sc
        mean_des = float(sum(des_list) / len(des_list))
        f10_ef = float(f10["edge_fidelity"]) if f10 else float("nan")
        f05_ef = float(f05["edge_fidelity"]) if f05 else float("nan")
        f05_ng = float(f05["noise_gain"]) if f05 else float("nan")
        f1_ef = float(f1["edge_fidelity"]) if f1 else float("nan")
        ok = edge_safe(f10_ef, f05_ef, f1_ef)
        rec = {
            "gen": gen,
            "iter": idx,
            "name": recipe.name,
            "family": recipe.family,
            "mean_des": mean_des,
            "f10_ef": f10_ef,
            "f05_ef": f05_ef,
            "f05_ng": f05_ng,
            "f1_ef": f1_ef,
            "edge_safe": ok,
            "recipe": asdict(recipe),
        }
        history.append(rec)
        if best_any is None or mean_des > best_any["mean_des"]:
            best_any = rec
        if ok and (best_safe is None or mean_des > best_safe["mean_des"]):
            best_safe = rec
            print(
                f"[g{gen:02d} {idx:03d}/{len(recipes)}] NEW SAFE {recipe.name} "
                f"DES={mean_des:.4f}",
                flush=True,
            )
        elif (idx + 1) % 20 == 0 or idx == 0:
            print(
                f"[g{gen:02d} {idx:03d}/{len(recipes)}] {recipe.name}: "
                f"DES={mean_des:.4f} ({'ok' if ok else 'unsafe'})",
                flush=True,
            )

    csv_path = out_dir / f"gen{gen:02d}_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gen",
                "iter",
                "name",
                "family",
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

    summary = {
        "gen": gen,
        "best_safe": best_safe,
        "best_any": best_any,
        "n": len(history),
    }
    (out_dir / f"gen{gen:02d}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100")
    )
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edge",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt"),
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--max-gens", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start-gen", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    metas = cache_forwards(
        files,
        args.cache_dir,
        args.checkpoint_sota,
        args.checkpoint_edge,
        args.tile_size,
    )
    packs = [
        {
            "meta": m,
            "sota": np.load(args.cache_dir / f"{m['stem']}_sota.npy"),
            "edge": np.load(args.cache_dir / f"{m['stem']}_edge.npy"),
        }
        for m in metas
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_best: dict | None = None
    best_recipe: Recipe | None = None
    stagnant = 0
    ledger: list[dict] = []

    for gen in range(int(args.start_gen), int(args.max_gens)):
        print(f"\n===== META GEN {gen} family~{LIT_FAMILIES[gen % len(LIT_FAMILIES)]} =====", flush=True)
        recipes = refine_around(best_recipe, gen, n=int(args.iters))
        summary = run_generation(packs, args.cache_dir, recipes, args.output_dir, gen)
        bs = summary.get("best_safe")
        improved = False
        if bs is not None:
            if global_best is None or bs["mean_des"] > global_best["mean_des"] + 1e-4:
                global_best = bs
                best_recipe = Recipe(**bs["recipe"])
                improved = True
                stagnant = 0
                (args.output_dir / "global_best.json").write_text(
                    json.dumps(global_best, indent=2), encoding="utf-8"
                )
                print(
                    f"GLOBAL BEST ← {bs['name']} DES={bs['mean_des']:.4f}",
                    flush=True,
                )
            else:
                stagnant += 1
        else:
            stagnant += 1
        ledger.append(
            {
                "gen": gen,
                "best_safe_des": None if bs is None else bs["mean_des"],
                "best_safe_name": None if bs is None else bs["name"],
                "improved": improved,
                "stagnant": stagnant,
            }
        )
        (args.output_dir / "ledger.json").write_text(
            json.dumps({"ledger": ledger, "global_best": global_best}, indent=2),
            encoding="utf-8",
        )
        if stagnant >= int(args.patience):
            print(
                f"Early stop: {stagnant} gens without +1e-4 DES "
                f"(patience={args.patience}). global_best="
                f"{None if global_best is None else global_best['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- meta done ---", flush=True)
    if global_best:
        print(
            f"FINAL {global_best['name']} mean={global_best['mean_des']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
