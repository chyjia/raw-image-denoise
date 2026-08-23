"""P12 meta: AIM mask FT first (never reached in P11) + tile-shift ensemble.

Baseline: P11 deploy TTA4 DES≈0.9238
Lit:
  AIM25 MR-CAS random patch mask FT
  Tile-shift / offset ensemble (industrial X-ray multi-offset inference)
  S2R few-shot real FT with stronger hard-scene focus

min_eval=20 so mask family runs before early-stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .compare_sota_edgekd_blend import _spatial_on_merged
from .infer import load_model
from .p8_fusion import deploy_p7_base
from .p10_tta import apply_aug, undo_aug
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 0.9238
TTA4 = ("id", "lr", "ud", "udlr")


@dataclass
class Recipe:
    name: str
    family: str
    # mask FT
    mask_ratio: float = 0.0
    mask_patch: int = 16
    target: str = "edge"  # edge | sota
    epochs: int = 4
    patches: int = 128
    lr: float = 1.5e-6
    highpass_weight: float = 0.55
    des_edge_fid: float = 1.5
    # shift ensemble (zero-train)
    shifts: tuple[tuple[int, int], ...] = ()
    # score TTA
    augs: tuple[str, ...] = TTA4


def build_100_recipes() -> list[Recipe]:
    recipes: list[Recipe] = [
        Recipe(name="p11_tta4", family="baseline", epochs=0),
    ]

    # Mask FT first (AIM) — ~50 recipes
    configs = []
    for ratio in [0.25, 0.35, 0.40, 0.50, 0.55, 0.60, 0.70, 0.75]:
        for psz in [8, 16, 24]:
            for target in ["edge", "sota"]:
                configs.append((ratio, psz, target))
    for i, (ratio, psz, target) in enumerate(configs[:48]):
        recipes.append(
            Recipe(
                name=f"mask_{i}_r{ratio:g}_p{psz}_{target}",
                family="mask_ft",
                mask_ratio=ratio,
                mask_patch=psz,
                target=target,
                epochs=4,
                des_edge_fid=1.4 + 0.1 * (i % 5),
                highpass_weight=0.45 + 0.05 * (i % 4),
            )
        )

    # Tile-shift ensembles (zero-train)
    for shifts in [
        ((0, 0), (8, 0)),
        ((0, 0), (0, 8)),
        ((0, 0), (8, 8)),
        ((0, 0), (-8, 0), (8, 0)),
        ((0, 0), (0, -8), (0, 8)),
        ((0, 0), (16, 0), (-16, 0)),
        ((0, 0), (0, 16), (0, -16)),
        ((0, 0), (8, 0), (0, 8), (8, 8)),
        ((0, 0), (16, 16), (-16, -16)),
        ((0, 0), (12, 0), (0, 12)),
        ((0, 0), (8, -8), (-8, 8)),
        ((0, 0), (4, 4), (-4, -4), (4, -4), (-4, 4)),
    ]:
        tag = "_".join(f"{dy}x{dx}" for dy, dx in shifts)
        recipes.append(
            Recipe(
                name=f"shift_{tag}",
                family="shift",
                epochs=0,
                shifts=shifts,
            )
        )

    # Extra mask with longer/milder
    for i, (ratio, ep, lr) in enumerate(
        [
            (0.50, 6, 1e-6),
            (0.35, 6, 1e-6),
            (0.60, 5, 1.2e-6),
            (0.45, 6, 2e-6),
            (0.50, 4, 2.5e-6),
            (0.40, 5, 1e-6),
            (0.55, 6, 1.5e-6),
            (0.30, 4, 2e-6),
            (0.65, 4, 1e-6),
            (0.50, 8, 8e-7),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"masklong_{i}_r{ratio:g}_e{ep}",
                family="mask_ft",
                mask_ratio=ratio,
                mask_patch=16,
                target="edge" if i % 2 == 0 else "sota",
                epochs=ep,
                lr=lr,
            )
        )

    recipes = recipes[:100]
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_mask_{i}",
                family="mask_ft",
                mask_ratio=0.3 + 0.02 * (i % 20),
                mask_patch=16,
                target="edge" if i % 2 == 0 else "sota",
            )
        )
    seen: set[str] = set()
    out: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(out)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        out.append(r)
    return out[:100]


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def arm_tta_from_cache(
    cache_dir: Path, stem: str, arm: str, augs: tuple[str, ...]
) -> np.ndarray:
    imgs = []
    for aug in augs:
        if aug == "id":
            p = cache_dir / f"{stem}_{arm}.npy"
        else:
            p = cache_dir / f"{stem}_{arm}_t256_{aug}.npy"
        if not p.exists():
            p = cache_dir / f"{stem}_{arm}.npy"
        imgs.append(np.load(p))
    return np.mean(np.stack(imgs, 0), 0).astype(np.float32)


def score_shift_ensemble(
    files: list[Path],
    metas: list[dict],
    cache_dir: Path,
    recipe: Recipe,
) -> dict:
    """Zero-train: average deploy outputs over FE spatial shifts (id TTA arms from cache + live shift)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_ckpt = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge_ckpt = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")
    if not edge_ckpt.exists():
        edge_ckpt = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    sota_m, n_in, _ = load_model(sota_ckpt, None, device)
    edge_m, _, _ = load_model(edge_ckpt, None, device)
    des_list = []
    f10 = f05 = f1 = None
    for path, meta in zip(files, metas):
        stem = path.stem
        fe = np.load(cache_dir / f"{stem}_fe.npy")
        outs = []
        for dy, dx in recipe.shifts:
            fe_s = shift_image(fe, dy, dx)
            # TTA4 on shifted FE
            sota_outs, edge_outs = [], []
            for aug in recipe.augs:
                fe_t = apply_aug(fe_s, aug)
                sota_outs.append(
                    undo_aug(
                        _spatial_on_merged(
                            sota_m,
                            fe_t,
                            float(meta["exposure_ms"]),
                            device,
                            n_in,
                            256,
                        ),
                        aug,
                    )
                )
                edge_outs.append(
                    undo_aug(
                        _spatial_on_merged(
                            edge_m,
                            fe_t,
                            float(meta["exposure_ms"]),
                            device,
                            n_in,
                            256,
                        ),
                        aug,
                    )
                )
            sota = np.mean(np.stack(sota_outs, 0), 0)
            edge = np.mean(np.stack(edge_outs, 0), 0)
            out = deploy_p7_base(sota, edge, float(meta["fps"]))
            # unshift output
            if dy or dx:
                out = shift_image(out, -dy, -dx)
            outs.append(out)
        final = np.mean(np.stack(outs, 0), 0).astype(np.float32)
        sc = score_image(meta, cache_dir, recipe.name, final)
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
    return {
        "mean_des": mean_des,
        "f10_ef": f10_ef,
        "f05_ef": f05_ef,
        "f05_ng": f05_ng,
        "f1_ef": f1_ef,
        "edge_safe": edge_safe(f10_ef, f05_ef, f1_ef),
    }


def train_mask(recipe: Recipe, out_root: Path) -> Path | None:
    out_dir = out_root / f"ckpts_{recipe.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / "best.pt"
    if existing.exists():
        print(f"Reuse existing {existing}", flush=True)
        return existing
    sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    resume = edge if recipe.target == "edge" else sota
    hard = Path("nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json")
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
        "--wiener-bm3d-prob", "1.0", "--distill-edge-harden", "16.0",
        "--loss-domain", "dn",
        "--real-manifest", str(hard),
        "--out-dir", str(out_dir),
        "--resume", str(resume), "--fresh-resume",
        "--use-sigma", "--input-frames", "4", "--width", "32",
        "--patch-size", "256", "--micro-batch", "2", "--accum-steps", "4",
        "--epochs", str(recipe.epochs),
        "--patches-per-epoch", str(recipe.patches),
        "--real-fraction", "1.0", "--lr", str(recipe.lr),
        "--highpass-weight", str(recipe.highpass_weight),
        "--des-edge-fid-weight", str(recipe.des_edge_fid),
        "--edge-weight", "0.40", "--grad-weight", "0.40",
        "--input-mask-ratio", str(recipe.mask_ratio),
        "--input-mask-patch", str(recipe.mask_patch),
        "--blend-kd-sota", str(sota),
        "--blend-kd-edgekd", str(edge),
        "--blend-kd-mix", "0.85",
        "--blend-kd-temperature", "8.0",
        "--blend-kd-harden", "16.0",
        "--validation-every", "99",
        "--validation-dir", str(material),
        "--num-workers", "0",
    ]
    if recipe.target == "edge":
        cmd.extend(["--freeze-flat-head", "--use-lap-edge", "--use-lap-edge-ms"])
    else:
        cmd.append("--freeze-edge-head")
    log = out_dir / "train.log"
    print(f"MASK-FT {recipe.name} r={recipe.mask_ratio} p={recipe.mask_patch}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    ckpt = out_dir / "best.pt"
    if ckpt.exists():
        return ckpt
    last = out_dir / "last.pt"
    if last.exists():
        return last
    print(f"TRAIN FAIL {recipe.name} exit={code}", flush=True)
    return None


def score_with_ckpt(
    files, metas, cache_dir, recipe: Recipe, ckpt: Path
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_in, _ = load_model(ckpt, None, device)
    des_list = []
    f10 = f05 = f1 = None
    for path, meta in zip(files, metas):
        stem = path.stem
        fe = np.load(cache_dir / f"{stem}_fe.npy")
        outs = []
        for aug in recipe.augs:
            fe_t = apply_aug(fe, aug)
            outs.append(
                undo_aug(
                    _spatial_on_merged(
                        model, fe_t, float(meta["exposure_ms"]), device, n_in, 256
                    ),
                    aug,
                )
            )
        dn_new = np.mean(np.stack(outs, 0), 0).astype(np.float32)
        if recipe.target == "edge":
            sota = arm_tta_from_cache(cache_dir, stem, "sota", recipe.augs)
            edge = dn_new
        else:
            sota = dn_new
            edge = arm_tta_from_cache(cache_dir, stem, "edge", recipe.augs)
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, recipe.name, out)
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
    return {
        "mean_des": mean_des,
        "f10_ef": f10_ef,
        "f05_ef": f05_ef,
        "f05_ng": f05_ng,
        "f1_ef": f1_ef,
        "edge_safe": edge_safe(f10_ef, f05_ef, f1_ef),
    }


def score_baseline(files, metas, cache_dir) -> dict:
    des_list = []
    f10 = f05 = f1 = None
    for path, meta in zip(files, metas):
        stem = path.stem
        sota = arm_tta_from_cache(cache_dir, stem, "sota", TTA4)
        edge = arm_tta_from_cache(cache_dir, stem, "edge", TTA4)
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, "p11_tta4", out)
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
    return {
        "mean_des": mean_des,
        "f10_ef": f10_ef,
        "f05_ef": f05_ef,
        "f05_ng": f05_ng,
        "f1_ef": f1_ef,
        "edge_safe": edge_safe(f10_ef, f05_ef, f1_ef),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p12")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-eval", type=int, default=25)
    parser.add_argument(
        "--max-shift",
        type=int,
        default=4,
        help="Max shift-ensemble recipes (expensive); rest skipped if stagnant.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    # need FE
    for p in files:
        if not (args.cache_dir / f"{p.stem}_fe.npy").exists():
            raise SystemExit(f"Missing FE cache for {p.stem}; run P10 cache first")

    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict | None = None
    stagnant = 0
    shift_count = 0
    hist_path = args.output_dir / "history.json"
    if int(args.start) > 0 and hist_path.exists():
        prev = json.loads(hist_path.read_text(encoding="utf-8"))
        history = list(prev.get("history") or [])
        best_safe = prev.get("best_safe")
        # Fresh patience budget for remaining catalog (avoid early-stop on resume).
        stagnant = 0
        print(
            f"Resuming: loaded {len(history)} history rows, "
            f"best={None if best_safe is None else best_safe.get('mean_des')}, "
            f"stagnant reset to 0",
            flush=True,
        )

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        # limit expensive shift recipes
        if recipe.family == "shift":
            if shift_count >= int(args.max_shift):
                continue
            shift_count += 1

        print(f"\n===== P12 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)

        if recipe.family == "baseline":
            sc = score_baseline(files, metas, args.cache_dir)
            rec = {"iter": idx, "name": recipe.name, "family": recipe.family, "recipe": asdict(recipe), **sc}
        elif recipe.family == "shift":
            sc = score_shift_ensemble(files, metas, args.cache_dir, recipe)
            rec = {"iter": idx, "name": recipe.name, "family": recipe.family, "recipe": asdict(recipe), **sc}
        else:
            ckpt = train_mask(recipe, args.output_dir)
            if ckpt is None:
                stagnant += 1
                history.append(
                    {
                        "iter": idx,
                        "name": recipe.name,
                        "failed": True,
                        "mean_des": 0.0,
                        "edge_safe": False,
                    }
                )
                if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                    break
                continue
            sc = score_with_ckpt(files, metas, args.cache_dir, recipe, ckpt)
            rec = {
                "iter": idx,
                "name": recipe.name,
                "family": recipe.family,
                "ckpt": str(ckpt),
                "recipe": asdict(recipe),
                **sc,
            }

        history.append(rec)
        mean_des = sc["mean_des"]
        ok = sc["edge_safe"]
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

        if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience) and idx > 0:
            # If still in mask_ft, jump to shift family once instead of full stop.
            remaining_shift = [
                (j, r)
                for j, r in enumerate(recipes)
                if j > idx and r.family == "shift"
            ]
            if recipe.family == "mask_ft" and remaining_shift and shift_count < int(args.max_shift):
                print(
                    f"Mask family saturated (stagnant={stagnant}); "
                    f"jumping to shift ensembles",
                    flush=True,
                )
                stagnant = 0
                for j, rshift in remaining_shift:
                    if shift_count >= int(args.max_shift):
                        break
                    shift_count += 1
                    print(
                        f"\n===== P12 [{j:03d}/{len(recipes)}] {rshift.name} =====",
                        flush=True,
                    )
                    sc = score_shift_ensemble(files, metas, args.cache_dir, rshift)
                    rec = {
                        "iter": j,
                        "name": rshift.name,
                        "family": rshift.family,
                        "recipe": asdict(rshift),
                        **sc,
                    }
                    history.append(rec)
                    mean_des = sc["mean_des"]
                    ok = sc["edge_safe"]
                    if ok and (
                        best_safe is None or mean_des > best_safe["mean_des"] + 1e-4
                    ):
                        best_safe = rec
                        stagnant = 0
                        (args.output_dir / "global_best.json").write_text(
                            json.dumps(best_safe, indent=2), encoding="utf-8"
                        )
                        print(
                            f"NEW SAFE BEST {rshift.name} DES={mean_des:.4f}",
                            flush=True,
                        )
                    else:
                        stagnant += 1
                        print(
                            f"{rshift.name}: DES={mean_des:.4f} "
                            f"({'ok' if ok else 'unsafe'}) stagnant={stagnant}",
                            flush=True,
                        )
                    (args.output_dir / "history.json").write_text(
                        json.dumps(
                            {"history": history, "best_safe": best_safe}, indent=2
                        ),
                        encoding="utf-8",
                    )
                    if stagnant >= int(args.patience):
                        print("Shift family stagnant; stopping", flush=True)
                        break
                break
            print(
                f"Early stop: stagnant={stagnant} best="
                f"{None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P12 done ---", flush=True)
    (args.output_dir / "DONE").write_text(
        json.dumps(
            {
                "best_safe": best_safe,
                "n_history": len(history),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        bake = args.output_dir / "bake_recipe.json"
        bake.write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
        Path("nafnet_denoise/deploy_p12_hook.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        if best_safe.get("family") == "mask_ft" and best_safe.get("ckpt"):
            target = best_safe["recipe"]["target"]
            dest = Path(
                "nafnet_denoise/checkpoints_4f_p12_deploy_edge"
                if target == "edge"
                else "nafnet_denoise/checkpoints_4f_p12_deploy_sota"
            )
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_safe["ckpt"], dest / "best.pt")
            dep = Path("nafnet_denoise/infer_blend_deploy.py")
            text = dep.read_text(encoding="utf-8")
            if target == "edge":
                text = text.replace(
                    "nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt",
                    "nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt",
                )
            else:
                text = text.replace(
                    "nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt",
                    "nafnet_denoise/checkpoints_4f_p12_deploy_sota/best.pt",
                )
            import re

            text = re.sub(
                r"mean DES ~0\.\d+",
                f"mean DES ~{best_safe['mean_des']:.4f}",
                text,
                count=1,
            )
            dep.write_text(text, encoding="utf-8")
            print(f"Patched deploy checkpoint → {dest}", flush=True)
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
