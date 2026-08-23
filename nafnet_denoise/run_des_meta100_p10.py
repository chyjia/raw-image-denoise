"""P10 meta: true geometric TTA + multi-tile + AIM mask FT (postprocess/FT saturated).

Lit (unused before):
  NTIRE/SwinIR ×4–8 geometric self-ensemble on *model input* (not DN-post avg)
  Multi-tile size average (256/320/384)
  AIM25 MAE random input masking during short FT

Baseline deploy P7 ≈ 0.9221. Up to 100 recipes; early-stop patience=5.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .common import memmap_frames, parse_exposure_ms, parse_geometry
from .compare_sota_edgekd_blend import _spatial_on_merged
from .fe_schedule import gated_spatial_from_temporal
from .infer import load_model
from .p8_fusion import deploy_p7_base
from .p10_tta import AUGS, apply_aug, average_augs, undo_aug
from .run_des_loop100 import (
    F1,
    F05,
    F10,
    cache_forwards,
    edge_safe,
    score_image,
)
from .wiener_merge import merge_from_memmap

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 0.9221


@dataclass
class Recipe:
    name: str
    family: str = "tta"
    augs: tuple[str, ...] = ("id",)
    weights: tuple[float, ...] | None = None
    # which arm to TTA
    tta_sota: bool = True
    tta_edge: bool = True
    # tile key: "t256" default cache; optional extras
    tiles: tuple[int, ...] = (256,)
    # optional post override unsharp
    u_amt: float = 0.15
    u_sig: float = 1.4
    # mask FT (family=mask_ft)
    mask_ratio: float = 0.0
    epochs: int = 0
    target: str = "edge"


def build_100_recipes() -> list[Recipe]:
    recipes: list[Recipe] = [Recipe(name="p7_deploy", family="baseline", augs=("id",))]

    # geometric TTA subsets
    for augs in [
        ("id", "lr"),
        ("id", "ud"),
        ("id", "udlr"),
        ("id", "lr", "ud"),
        ("id", "lr", "ud", "udlr"),
        ("lr", "ud"),
        ("id", "lr", "udlr"),
        ("id", "ud", "udlr"),
    ]:
        tag = "_".join(augs)
        recipes.append(Recipe(name=f"tta_{tag}", family="tta", augs=augs))
        recipes.append(
            Recipe(
                name=f"tta_edge_{tag}",
                family="tta",
                augs=augs,
                tta_sota=False,
                tta_edge=True,
            )
        )
        recipes.append(
            Recipe(
                name=f"tta_sota_{tag}",
                family="tta",
                augs=augs,
                tta_sota=True,
                tta_edge=False,
            )
        )

    # weighted TTA
    for w in [
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.4, 0.3, 0.3),
        (0.4, 0.2, 0.2, 0.2),
        (0.55, 0.15, 0.15, 0.15),
    ]:
        augs = AUGS[: len(w)]
        recipes.append(
            Recipe(
                name=f"wttta_{'_'.join(f'{x:g}' for x in w)}",
                family="tta",
                augs=augs,
                weights=w,
            )
        )

    # multi-tile
    for tiles in [
        (256, 320),
        (256, 384),
        (192, 256),
        (256, 320, 384),
        (224, 256, 320),
        (256, 288),
        (192, 256, 320),
        (256, 384, 448),
    ]:
        recipes.append(
            Recipe(
                name=f"tile_{'_'.join(str(t) for t in tiles)}",
                family="tile",
                augs=("id",),
                tiles=tiles,
            )
        )
        recipes.append(
            Recipe(
                name=f"tile_tta_{'_'.join(str(t) for t in tiles)}",
                family="tile_tta",
                augs=("id", "lr", "ud", "udlr"),
                tiles=tiles,
            )
        )

    # TTA + mild unsharp refine
    for amt, sig in [
        (0.12, 1.6),
        (0.15, 1.4),
        (0.10, 1.8),
        (0.18, 1.2),
        (0.14, 1.5),
    ]:
        recipes.append(
            Recipe(
                name=f"tta4_u{amt:g}_s{sig:g}",
                family="tta",
                augs=("id", "lr", "ud", "udlr"),
                u_amt=amt,
                u_sig=sig,
            )
        )

    # mask FT recipes (run after zero-train if used in sequential loop)
    for i, (ratio, target) in enumerate(
        [
            (0.25, "edge"),
            (0.35, "edge"),
            (0.50, "edge"),
            (0.50, "sota"),
            (0.35, "sota"),
            (0.40, "edge"),
            (0.60, "edge"),
            (0.30, "sota"),
            (0.45, "edge"),
            (0.55, "sota"),
            (0.40, "sota"),
            (0.50, "edge"),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"maskft_{i}_r{ratio:g}_{target}",
                family="mask_ft",
                augs=("id",),
                mask_ratio=ratio,
                epochs=4,
                target=target,
            )
        )

    recipes = recipes[:100]
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_tta_{i}",
                family="tta",
                augs=("id", "lr") if i % 2 == 0 else ("id", "ud", "lr"),
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_d{len(out)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        out.append(r)
    return out[:100]


def cache_fe_and_tta(
    files: list[Path],
    cache_dir: Path,
    ckpt_sota: Path,
    ckpt_edge: Path,
    tiles: tuple[int, ...] = (256, 192, 320, 384),
) -> None:
    """Cache FE + spatial TTA / multi-tile DN for both arms."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sota, n_in, _ = load_model(ckpt_sota, None, device)
    model_edge, n_e, _ = load_model(ckpt_edge, None, device)
    assert n_in == n_e

    for path in files:
        stem = path.stem
        fe_path = cache_dir / f"{stem}_fe.npy"
        meta_path = cache_dir / f"{stem}.json"
        if not meta_path.exists():
            raise SystemExit(f"missing dual cache meta {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not fe_path.exists():
            width, height, fps = parse_geometry(path.name)
            frames = memmap_frames(path, width, height)
            target_index = int(meta["target_index"])
            temporal, _ = merge_from_memmap(
                frames,
                target_index,
                input_frames=16,
                tile_size=32,
                overlap=16,
                c_factor=8.0,
                align=True,
                spatial_wiener=False,
            )
            fe_gated, params, fe_sigma = gated_spatial_from_temporal(
                temporal,
                fps=float(fps),
                n_frames_averaged=16,
                tile_size=32,
                overlap=16,
            )
            np.save(fe_path, fe_gated.astype(np.float32))
            meta["fe_schedule"] = params.name
            meta["fe_sigma_dn"] = float(fe_sigma)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"cached FE {stem}", flush=True)
        else:
            fe_gated = np.load(fe_path)
            print(f"FE hit {stem}", flush=True)

        exposure_ms = float(meta["exposure_ms"])
        for arm, model in (("sota", model_sota), ("edge", model_edge)):
            for tile in tiles:
                for aug in AUGS:
                    out_path = cache_dir / f"{stem}_{arm}_t{tile}_{aug}.npy"
                    if out_path.exists():
                        continue
                    # identity tile=256 can copy from dual cache
                    if tile == 256 and aug == "id":
                        src = cache_dir / f"{stem}_{arm}.npy"
                        if src.exists():
                            np.save(out_path, np.load(src))
                            continue
                    fe_t = apply_aug(fe_gated, aug)
                    dn = _spatial_on_merged(
                        model, fe_t, exposure_ms, device, n_in, tile
                    )
                    dn = undo_aug(dn, aug)
                    np.save(out_path, dn.astype(np.float32))
                    print(f"cached {out_path.name}", flush=True)


def load_arm(
    cache_dir: Path,
    stem: str,
    arm: str,
    augs: tuple[str, ...],
    tiles: tuple[int, ...],
    weights: tuple[float, ...] | None,
) -> np.ndarray:
    # average over tiles then augs (or all pairs equally)
    imgs: list[np.ndarray] = []
    for tile in tiles:
        by_aug = {}
        for aug in augs:
            p = cache_dir / f"{stem}_{arm}_t{tile}_{aug}.npy"
            if not p.exists():
                # fallback id 256
                p = cache_dir / f"{stem}_{arm}.npy"
            by_aug[aug] = np.load(p)
        imgs.append(average_augs(by_aug, augs, weights=weights))
    if len(imgs) == 1:
        return imgs[0]
    return np.mean(np.stack(imgs, axis=0), axis=0).astype(np.float32)


def apply_recipe(
    recipe: Recipe,
    cache_dir: Path,
    stem: str,
    fps: float,
) -> np.ndarray:
    sota_id = np.load(cache_dir / f"{stem}_sota.npy")
    edge_id = np.load(cache_dir / f"{stem}_edge.npy")
    if recipe.family == "baseline":
        return deploy_p7_base(sota_id, edge_id, fps)

    sota = (
        load_arm(
            cache_dir, stem, "sota", recipe.augs, recipe.tiles, recipe.weights
        )
        if recipe.tta_sota
        else sota_id
    )
    edge = (
        load_arm(
            cache_dir, stem, "edge", recipe.augs, recipe.tiles, recipe.weights
        )
        if recipe.tta_edge
        else edge_id
    )
    # custom unsharp via deploy_p7_base args
    return deploy_p7_base(
        sota, edge, fps, unsharp_amount=recipe.u_amt, unsharp_sigma=recipe.u_sig
    )


def ensure_input_mask_flag() -> None:
    """Idempotently add --input-mask-ratio support to train_distill if missing."""
    path = Path("nafnet_denoise/train_distill.py")
    text = path.read_text(encoding="utf-8")
    if "--input-mask-ratio" in text:
        return
    # insert argparse
    needle = 'parser.add_argument("--highpass-weight", type=float, default=0.25)'
    if needle not in text:
        print("WARN: could not patch train_distill for mask", flush=True)
        return
    insert = (
        needle
        + "\n    parser.add_argument(\"--input-mask-ratio\", type=float, default=0.0,"
        + "\n                        help=\"AIM-style random 16x16 patch mask ratio on image channels.\")"
        + "\n    parser.add_argument(\"--input-mask-patch\", type=int, default=16)"
    )
    text = text.replace(needle, insert, 1)
    # apply mask after inputs loaded
    apply_needle = "            noisy_center = inputs[:, center : center + 1]"
    apply_code = '''            if float(getattr(args, "input_mask_ratio", 0.0) or 0.0) > 0.0:
                ratio = float(args.input_mask_ratio)
                psz = int(getattr(args, "input_mask_patch", 16) or 16)
                if torch.rand(()) < 0.5:
                    b, c, h, w = inputs.shape
                    gh, gw = max(1, h // psz), max(1, w // psz)
                    nmask = max(1, int(gh * gw * ratio))
                    flat = torch.rand(b, gh * gw, device=inputs.device)
                    # keep lowest scores as masked
                    kth = torch.kthvalue(flat, nmask, dim=1).values.unsqueeze(1)
                    patch_mask = (flat <= kth).view(b, 1, gh, gw).float()
                    patch_mask = torch.nn.functional.interpolate(
                        patch_mask, size=(h, w), mode="nearest"
                    )
                    n_img = int(args.input_frames)
                    inputs = inputs.clone()
                    inputs[:, :n_img] = inputs[:, :n_img] * (1.0 - patch_mask)
            noisy_center = inputs[:, center : center + 1]'''
    if apply_needle in text:
        text = text.replace(apply_needle, apply_code, 1)
    path.write_text(text, encoding="utf-8")
    print("Patched train_distill with --input-mask-ratio", flush=True)


def run_mask_ft(recipe: Recipe, out_root: Path) -> Path | None:
    ensure_input_mask_flag()
    out_dir = out_root / f"ckpts_{recipe.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    resume = edge if recipe.target == "edge" else sota
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    material = Path(r"D:\denoise\val_raw")
    cmd = [
        sys.executable, "-u", "-m", "nafnet_denoise.train_distill",
        "--arch", "dual_head", "--use-nonlocal", "--nonlocal-count", "1",
        "--align-soft-gate", "--wiener-front-end", "--wiener-merge-frames", "16",
        "--wiener-tile", "32", "--wiener-overlap", "16", "--wiener-c-factor", "8.0",
        "--wiener-spatial", "--wiener-spatial-adaptive",
        "--wiener-fe-schedule", "fps_sigma",
        "--teacher-mode", "partitioned",
        "--wiener-bm3d-teacher", "--wiener-bm3d-sigma", "0.35",
        "--wiener-bm3d-prob", "1.0",
        "--distill-edge-harden", "16.0",
        "--loss-domain", "dn",
        "--real-manifest", str(hard_manifest),
        "--out-dir", str(out_dir),
        "--resume", str(resume),
        "--fresh-resume",
        "--use-sigma", "--input-frames", "4", "--width", "32",
        "--patch-size", "256", "--micro-batch", "2", "--accum-steps", "4",
        "--epochs", str(max(3, recipe.epochs)),
        "--patches-per-epoch", "128",
        "--real-fraction", "1.0", "--lr", "1.5e-6",
        "--highpass-weight", "0.55",
        "--des-edge-fid-weight", "1.5",
        "--edge-weight", "0.40",
        "--grad-weight", "0.40",
        "--input-mask-ratio", str(recipe.mask_ratio),
        "--input-mask-patch", "16",
        "--validation-every", "99",
        "--validation-dir", str(material),
        "--num-workers", "0",
        "--blend-kd-sota", str(sota),
        "--blend-kd-edgekd", str(edge),
        "--blend-kd-mix", "0.85",
        "--blend-kd-temperature", "8.0",
        "--blend-kd-harden", "16.0",
    ]
    if recipe.target == "edge":
        cmd.extend(["--freeze-flat-head", "--use-lap-edge", "--use-lap-edge-ms"])
    else:
        cmd.append("--freeze-edge-head")
    log = out_dir / "train.log"
    print(f"MASK-FT {recipe.name}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    ckpt = out_dir / "best.pt"
    if ckpt.exists():
        return ckpt
    last = out_dir / "last.pt"
    return last if last.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p10")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip TTA/tile cache rebuild",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    sota_ckpt = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge_ckpt = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    cache_forwards(files, args.cache_dir, sota_ckpt, edge_ckpt, 256)
    if not args.skip_cache:
        cache_fe_and_tta(
            files,
            args.cache_dir,
            sota_ckpt,
            edge_ckpt,
            tiles=(256, 192, 320, 384),
        )

    recipes = build_100_recipes()[: int(args.max_recipes)]
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
        print(f"\n===== P10 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)

        # mask FT path: train then need live score — use trained edge/sota with id only
        if recipe.family == "mask_ft":
            ckpt = run_mask_ft(recipe, args.output_dir)
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
                if stagnant >= int(args.patience):
                    break
                continue
            # Score via reusing dual cache but replace one arm with new forwards
            # For speed: only recompute id@256 for trained arm
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, n_in, _ = load_model(ckpt, None, device)
            des_list = []
            f10 = f05 = f1 = None
            for path, meta in zip(files, metas):
                stem = path.stem
                fe = np.load(args.cache_dir / f"{stem}_fe.npy")
                dn_new = _spatial_on_merged(
                    model, fe, float(meta["exposure_ms"]), device, n_in, 256
                )
                if recipe.target == "edge":
                    sota = np.load(args.cache_dir / f"{stem}_sota.npy")
                    edge = dn_new
                else:
                    sota = dn_new
                    edge = np.load(args.cache_dir / f"{stem}_edge.npy")
                out = deploy_p7_base(sota, edge, float(meta["fps"]))
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
                out = apply_recipe(recipe, args.cache_dir, path.stem, float(meta["fps"]))
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
        if ok and (
            best_safe is None or mean_des > best_safe["mean_des"] + 1e-4
        ):
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

        if stagnant >= int(args.patience) and idx > 0:
            print(
                f"Early stop: {stagnant} without +1e-4. "
                f"best={None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P10 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        bake = args.output_dir / "bake_recipe.json"
        bake.write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
        hook = Path("nafnet_denoise/deploy_p10_hook.json")
        hook.write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        # Patch deploy to use TTA if family is tta
        if best_safe.get("family") in ("tta", "tile", "tile_tta"):
            dep = Path("nafnet_denoise/infer_blend_deploy.py")
            text = dep.read_text(encoding="utf-8")
            if "tta_augs" not in text:
                # add optional TTA after spatial — write note for manual; also
                # append small helper call in denoise_blend_deploy via hook file only
                note = args.output_dir / "BAKE_NOTE.txt"
                note.write_text(
                    f"Winner needs TTA augs={best_safe['recipe'].get('augs')}\n"
                    f"tiles={best_safe['recipe'].get('tiles')}\n"
                    f"See deploy_p10_hook.json — apply in infer_blend_deploy.\n",
                    encoding="utf-8",
                )
            # Always try to bake TTA into deploy
            _bake_tta_into_deploy(best_safe)
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


def _bake_tta_into_deploy(best: dict) -> None:
    """Inject light TTA (flip augs) into infer_blend_deploy spatial stage."""
    recipe = best.get("recipe") or {}
    augs = recipe.get("augs") or ["id"]
    if augs == ["id"] or augs == ("id",):
        return
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    if "from .p10_tta import" in text:
        # update defaults in docstring only
        pass
    else:
        text = text.replace(
            "from .p7_fusion import edge_unsharp\n",
            "from .p7_fusion import edge_unsharp\n"
            "from .p10_tta import apply_aug, undo_aug\n",
            1,
        )
        # wrap _spatial_on_merged usage — replace function with TTA version
        old = '''def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile):
    frames = [merged.copy() for _ in range(input_frames)]
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    try:
        return denoise_from_dn_frames(model, frames, exposure_ms, device, tile=tile)
    finally:
        model.wiener_front_end = was'''
        new = f'''def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile,
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
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)'''
        if old in text:
            text = text.replace(old, new, 1)
        # add param to denoise_blend_deploy
        if "tta_augs: list[str] | None = None," not in text:
            text = text.replace(
                "unsharp_harden: float = 16.0,\n) -> tuple[np.ndarray, dict]:",
                "unsharp_harden: float = 16.0,\n"
                f"    tta_augs: list[str] | None = {list(augs)!r},\n"
                ") -> tuple[np.ndarray, dict]:",
                1,
            )
            text = text.replace(
                "model_sota, fe_gated, exposure_ms, device, n_in, tile\n    )",
                "model_sota, fe_gated, exposure_ms, device, n_in, tile, tta_augs=tta_augs\n    )",
                1,
            )
            text = text.replace(
                "model_edgekd, fe_gated, exposure_ms, device, n_in, tile\n    )",
                "model_edgekd, fe_gated, exposure_ms, device, n_in, tile, tta_augs=tta_augs\n    )",
                1,
            )
        # docstring
        text = text.replace(
            f"mean DES ~0.9221",
            f"mean DES ~{best['mean_des']:.4f}",
            1,
        )
        path.write_text(text, encoding="utf-8")
        print(f"Baked TTA augs={augs} into infer_blend_deploy.py", flush=True)


if __name__ == "__main__":
    main()
