"""P11 meta: finish untested P10 levers + lit extensions.

Baseline: P10 deploy (TTA id+lr) DES≈0.9232
Priority (P10 early-stopped before these):
  1) multi-tile average (NTIRE / Hann-style multi-size)
  2) ×4 flip TTA (id+lr+ud+udlr)
  3) tile×TTA combos
  4) AIM mask FT (short)

Uses existing cache_dual_holdout TTA/tile npy from P10.
min_eval before early-stop so tile family is actually tried.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .p8_fusion import deploy_p7_base
from .p10_tta import AUGS
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image
from .run_des_meta100_p10 import (
    apply_recipe as p10_apply,
    cache_fe_and_tta,
    ensure_input_mask_flag,
    run_mask_ft,
)
from .run_des_meta100_p10 import Recipe as P10Recipe
from .compare_sota_edgekd_blend import _spatial_on_merged
from .infer import load_model
import torch

BASELINE = 0.9232


@dataclass
class Recipe:
    name: str
    family: str
    augs: tuple[str, ...] = ("id", "lr")
    weights: tuple[float, ...] | None = None
    tta_sota: bool = True
    tta_edge: bool = True
    tiles: tuple[int, ...] = (256,)
    u_amt: float = 0.15
    u_sig: float = 1.4
    mask_ratio: float = 0.0
    epochs: int = 0
    target: str = "edge"


def build_100_recipes() -> list[Recipe]:
    recipes: list[Recipe] = [
        Recipe(name="p10_deploy", family="baseline", augs=("id", "lr"), tiles=(256,))
    ]

    # --- multi-tile first (untested) ---
    for tiles in [
        (256, 320),
        (256, 384),
        (192, 256),
        (256, 320, 384),
        (224, 256, 320),
        (192, 256, 320),
        (256, 288),
        (320, 384),
        (192, 320),
        (256, 192, 384),
        (192, 256, 384),
        (256, 384, 320),
        (192, 256, 384),
        (256, 320, 192),
        (320, 384, 192),
    ]:
        tiles = tuple(dict.fromkeys(t for t in tiles if t in (192, 256, 320, 384)))
        if len(tiles) < 2:
            continue
        recipes.append(
            Recipe(
                name=f"tile_{'_'.join(str(t) for t in tiles)}",
                family="tile",
                augs=("id", "lr"),  # keep P10 TTA
                tiles=tiles,
            )
        )

    # tile without TTA (ablation)
    for tiles in [(256, 320), (256, 384), (192, 256, 320), (256, 320, 384)]:
        recipes.append(
            Recipe(
                name=f"tile_notta_{'_'.join(str(t) for t in tiles)}",
                family="tile",
                augs=("id",),
                tiles=tiles,
            )
        )

    # ×4 TTA
    recipes.append(
        Recipe(
            name="tta4",
            family="tta",
            augs=("id", "lr", "ud", "udlr"),
            tiles=(256,),
        )
    )
    recipes.append(
        Recipe(
            name="tta4_sota",
            family="tta",
            augs=("id", "lr", "ud", "udlr"),
            tta_edge=False,
            tiles=(256,),
        )
    )
    recipes.append(
        Recipe(
            name="tta3_id_lr_ud",
            family="tta",
            augs=("id", "lr", "ud"),
            tiles=(256,),
        )
    )
    recipes.append(
        Recipe(
            name="tta3_id_lr_udlr",
            family="tta",
            augs=("id", "lr", "udlr"),
            tiles=(256,),
        )
    )

    # tile × TTA4
    for tiles in [(256, 320), (256, 384), (192, 256), (256, 320, 384)]:
        recipes.append(
            Recipe(
                name=f"tile_tta4_{'_'.join(str(t) for t in tiles)}",
                family="tile_tta",
                augs=("id", "lr", "ud", "udlr"),
                tiles=tiles,
            )
        )

    # weighted TTA around id+lr
    for w in [
        (0.55, 0.45),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.45, 0.35, 0.2),
        (0.5, 0.3, 0.2),
        (0.4, 0.3, 0.2, 0.1),
        (0.35, 0.35, 0.15, 0.15),
        (0.5, 0.25, 0.25),
    ]:
        augs = ("id", "lr") if len(w) == 2 else (("id", "lr", "ud") if len(w) == 3 else AUGS)
        recipes.append(
            Recipe(
                name=f"w_{'_'.join(f'{x:g}' for x in w)}",
                family="weighted",
                augs=augs[: len(w)],
                weights=w,
                tiles=(256,),
            )
        )

    # unsharp refine on P10 TTA
    for amt, sig in [
        (0.12, 1.6),
        (0.14, 1.5),
        (0.16, 1.3),
        (0.10, 1.8),
        (0.18, 1.2),
        (0.15, 1.6),
        (0.13, 1.4),
        (0.17, 1.4),
    ]:
        recipes.append(
            Recipe(
                name=f"p10_u{amt:g}_s{sig:g}",
                family="unsharp",
                augs=("id", "lr"),
                u_amt=amt,
                u_sig=sig,
            )
        )

    # AIM mask FT
    for i, (ratio, target) in enumerate(
        [
            (0.25, "edge"),
            (0.35, "edge"),
            (0.50, "edge"),
            (0.40, "sota"),
            (0.50, "sota"),
            (0.60, "edge"),
            (0.30, "sota"),
            (0.45, "edge"),
            (0.55, "edge"),
            (0.35, "sota"),
            (0.40, "edge"),
            (0.70, "edge"),
            (0.50, "edge"),
            (0.45, "sota"),
            (0.55, "sota"),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"maskft_{i}_r{ratio:g}_{target}",
                family="mask_ft",
                augs=("id", "lr"),
                mask_ratio=ratio,
                epochs=4,
                target=target,
            )
        )

    # pad to 100
    i = 0
    while len(recipes) < 100:
        recipes.append(
            Recipe(
                name=f"pad_tile_{i}",
                family="tile",
                augs=("id", "lr"),
                tiles=(256, 192 + 32 * (i % 5)) if (192 + 32 * (i % 5)) in (192, 224, 256, 288, 320, 384) else (256, 320),
            )
        )
        i += 1

    # unique
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes[:100]:
        # filter invalid tiles
        tiles = tuple(t for t in r.tiles if t in (192, 256, 288, 320, 384))
        if not tiles:
            tiles = (256,)
        r = Recipe(**{**asdict(r), "tiles": tiles})
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        out.append(r)
    return out[:100]


def to_p10(r: Recipe) -> P10Recipe:
    return P10Recipe(
        name=r.name,
        family=r.family,
        augs=r.augs,
        weights=r.weights,
        tta_sota=r.tta_sota,
        tta_edge=r.tta_edge,
        tiles=r.tiles,
        u_amt=r.u_amt,
        u_sig=r.u_sig,
        mask_ratio=r.mask_ratio,
        epochs=r.epochs,
        target=r.target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p11")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--min-eval",
        type=int,
        default=40,
        help="Do not early-stop before this many recipes (cover tile family).",
    )
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    sota_ckpt = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge_ckpt = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    if not args.skip_cache:
        # ensure tile 288 exists too if needed — only 192/256/320/384 from P10
        cache_fe_and_tta(
            files, args.cache_dir, sota_ckpt, edge_ckpt, tiles=(192, 256, 320, 384)
        )

    recipes = build_100_recipes()[: int(args.max_recipes)]
    # drop recipes needing missing tile 288
    recipes = [
        r
        for r in recipes
        if all(t in (192, 256, 320, 384) for t in r.tiles)
    ]
    while len(recipes) < int(args.max_recipes):
        recipes.append(
            Recipe(
                name=f"fill_tta4_{len(recipes)}",
                family="tta",
                augs=("id", "lr", "ud", "udlr"),
            )
        )
    recipes = recipes[: int(args.max_recipes)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]

    history: list[dict] = []
    best_safe: dict | None = None
    stagnant = 0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        print(f"\n===== P11 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)

        if recipe.family == "mask_ft":
            ensure_input_mask_flag()
            ckpt = run_mask_ft(to_p10(recipe), args.output_dir)
            if ckpt is None:
                stagnant += 1
                history.append(
                    {
                        "iter": idx,
                        "name": recipe.name,
                        "mean_des": 0.0,
                        "edge_safe": False,
                        "failed": True,
                    }
                )
                if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                    break
                continue
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, n_in, _ = load_model(ckpt, None, device)
            des_list = []
            f10 = f05 = f1 = None
            for path, meta in zip(files, metas):
                stem = path.stem
                fe = np.load(args.cache_dir / f"{stem}_fe.npy")
                # TTA id+lr on new arm
                outs = []
                for aug in recipe.augs:
                    from .p10_tta import apply_aug, undo_aug

                    fe_t = apply_aug(fe, aug)
                    dn = _spatial_on_merged(
                        model, fe_t, float(meta["exposure_ms"]), device, n_in, 256
                    )
                    outs.append(undo_aug(dn, aug))
                dn_new = np.mean(np.stack(outs, 0), 0).astype(np.float32)
                if recipe.target == "edge":
                    sota = np.load(args.cache_dir / f"{stem}_sota_t256_lr.npy")
                    # average id+lr for sota from cache
                    sota = 0.5 * (
                        np.load(args.cache_dir / f"{stem}_sota.npy")
                        + np.load(args.cache_dir / f"{stem}_sota_t256_lr.npy")
                    )
                    edge = dn_new
                else:
                    sota = dn_new
                    edge = 0.5 * (
                        np.load(args.cache_dir / f"{stem}_edge.npy")
                        + np.load(args.cache_dir / f"{stem}_edge_t256_lr.npy")
                    )
                out = deploy_p7_base(
                    sota, edge, float(meta["fps"]), recipe.u_amt, recipe.u_sig
                )
                sc = score_image(meta, args.cache_dir, recipe.name, out)
                des_list.append(sc["des"])
                if F10 in path.name:
                    f10 = sc
                if F05 in path.name:
                    f05 = sc
                if F1 in path.name:
                    f1 = sc
            mean_des = float(sum(des_list) / len(des_list))
            f10_ef = float(f10["edge_fidelity"]) if f10 else float("nan")
            f05_ef = float(f05["edge_fidelity"]) if f05 else float("nan")
            f05_ng = float(f05["noise_gain"]) if f05 else float("nan")
            f1_ef = float(f1["edge_fidelity"]) if f1 else float("nan")
            ok = edge_safe(f10_ef, f05_ef, f1_ef)
            rec = {
                "iter": idx,
                "name": recipe.name,
                "family": recipe.family,
                "mean_des": mean_des,
                "f10_ef": f10_ef,
                "f05_ef": f05_ef,
                "f05_ng": f05_ng,
                "f1_ef": f1_ef,
                "edge_safe": ok,
                "ckpt": str(ckpt),
                "recipe": asdict(recipe),
            }
        else:
            des_list = []
            f10 = f05 = f1 = None
            for path, meta in zip(files, metas):
                out = p10_apply(
                    to_p10(recipe), args.cache_dir, path.stem, float(meta["fps"])
                )
                sc = score_image(meta, args.cache_dir, recipe.name, out)
                des_list.append(sc["des"])
                if F10 in path.name:
                    f10 = sc
                if F05 in path.name:
                    f05 = sc
                if F1 in path.name:
                    f1 = sc
            mean_des = float(sum(des_list) / len(des_list))
            f10_ef = float(f10["edge_fidelity"]) if f10 else float("nan")
            f05_ef = float(f05["edge_fidelity"]) if f05 else float("nan")
            f05_ng = float(f05["noise_gain"]) if f05 else float("nan")
            f1_ef = float(f1["edge_fidelity"]) if f1 else float("nan")
            ok = edge_safe(f10_ef, f05_ef, f1_ef)
            rec = {
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
        if ok and (best_safe is None or mean_des > best_safe["mean_des"] + 1e-4):
            best_safe = rec
            stagnant = 0
            (args.output_dir / "global_best.json").write_text(
                json.dumps(best_safe, indent=2), encoding="utf-8"
            )
            print(f"NEW SAFE BEST {recipe.name} DES={mean_des:.4f}", flush=True)
        else:
            if recipe.family != "baseline":
                stagnant += 1
            print(
                f"{recipe.name}: DES={mean_des:.4f} "
                f"({'ok' if ok else 'unsafe'}) stagnant={stagnant}",
                flush=True,
            )

        (args.output_dir / "history.json").write_text(
            json.dumps({"history": history, "best_safe": best_safe}, indent=2),
            encoding="utf-8",
        )
        with (args.output_dir / "metrics_by_recipe.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
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
            for h in history:
                writer.writerow({k: h.get(k) for k in writer.fieldnames})

        if (
            idx + 1 >= int(args.min_eval)
            and stagnant >= int(args.patience)
            and idx > 0
        ):
            print(
                f"Early stop after min_eval: stagnant={stagnant} "
                f"best={None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P11 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        bake = args.output_dir / "bake_recipe.json"
        bake.write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
        Path("nafnet_denoise/deploy_p11_hook.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        _bake_deploy(best_safe)
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


def _bake_deploy(best: dict) -> None:
    """Update infer_blend_deploy TTA augs / note tiles."""
    recipe = best.get("recipe") or {}
    augs = list(recipe.get("augs") or ["id", "lr"])
    tiles = list(recipe.get("tiles") or [256])
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    import re

    text2 = re.sub(
        r"tta_augs: list\[str\] \| None = \[[^\]]*\]",
        f"tta_augs: list[str] | None = {augs!r}",
        text,
        count=1,
    )
    text2 = re.sub(
        r"mean DES ~0\.\d+",
        f"mean DES ~{best['mean_des']:.4f}",
        text2,
        count=1,
    )
    # unsharp
    if "u_amt" in recipe:
        text2 = re.sub(
            r"unsharp_amount: float = [0-9.]+",
            f"unsharp_amount: float = {float(recipe['u_amt'])}",
            text2,
            count=1,
        )
        text2 = re.sub(
            r"unsharp_sigma: float = [0-9.]+",
            f"unsharp_sigma: float = {float(recipe['u_sig'])}",
            text2,
            count=1,
        )
    path.write_text(text2, encoding="utf-8")
    note = Path("nafnet_denoise/compare_meta100_p11/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Baked augs={augs} tiles={tiles} DES={best['mean_des']:.4f}\n"
        f"Multi-tile averaging not wired into deploy path yet if tiles!=[256]; "
        f"TTA augs updated. For multi-tile deploy, keep hook JSON.\n",
        encoding="utf-8",
    )
    # If multi-tile winner, wire simple multi-tile into _spatial_on_merged
    if len(tiles) > 1 and all(isinstance(t, int) for t in tiles):
        if "tile_sizes" not in text2:
            text2 = path.read_text(encoding="utf-8")
            text2 = text2.replace(
                "tta_augs: list[str] | None = ",
                f"tile_sizes: list[int] | None = {tiles!r},\n    tta_augs: list[str] | None = ",
                1,
            )
            # enhance _spatial_on_merged to average tiles
            old = """def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile,
                       tta_augs=None):
    augs = list(tta_augs) if tta_augs else ["id"]
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    outs = []
    try:
        for aug in augs:
            fe = apply_aug(merged, aug)
            frames = [fe.copy() for _ in range(input_frames)]
            dn = denoise_from_dn_frames(model, frames, exposure_ms, device, tile=tile)
            outs.append(undo_aug(dn, aug))
    finally:
        model.wiener_front_end = was
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)"""
            new = """def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile,
                       tta_augs=None, tile_sizes=None):
    augs = list(tta_augs) if tta_augs else ["id"]
    tiles = list(tile_sizes) if tile_sizes else [tile]
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    outs = []
    try:
        for tsz in tiles:
            for aug in augs:
                fe = apply_aug(merged, aug)
                frames = [fe.copy() for _ in range(input_frames)]
                dn = denoise_from_dn_frames(
                    model, frames, exposure_ms, device, tile=int(tsz)
                )
                outs.append(undo_aug(dn, aug))
    finally:
        model.wiener_front_end = was
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)"""
            if old in text2:
                text2 = text2.replace(old, new, 1)
                text2 = text2.replace(
                    "tile, tta_augs=tta_augs\n    )",
                    "tile, tta_augs=tta_augs, tile_sizes=tile_sizes\n    )",
                )
                path.write_text(text2, encoding="utf-8")
                print(f"Baked multi-tile {tiles} into deploy", flush=True)
                return
    print(f"Updated deploy TTA augs={augs}", flush=True)


if __name__ == "__main__":
    main()
