"""P16 meta: FE-shift ensemble + wavelet/freq fuse on P13 D4 stack.

Baseline: P13 d4_r90_flip DES ≈ 0.9252
Lit leftovers (P13/14/15 saturated on soup/feat):
  PatchFusion / tile-shift — FE roll-shift before D4 spatial, average
  Wavelet / Gaussian freq-split fuse of D4 SOTA↔edge (WIFE / DualEx HF)
  Mild unsharp / detail-transfer on D4 deploy

min_eval ensures shift family is tried before early-stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .compare_sota_edgekd_blend import _spatial_on_merged
from .infer import load_model
from .multiband_fuse import multiband_fuse
from .p6_fusion import detail_transfer
from .p7_fusion import freq_split_fuse
from .p8_fusion import deploy_p7_base, umgf_fuse
from .p13_tta import geom_forward, geom_inverse
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9252
D4 = ("id", "r90", "r270", "r90_lr", "r270_lr")
SOTA_CKPT = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
EDGE_CKPT = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")
EDGE_FALLBACK = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")


@dataclass
class Recipe:
    name: str
    family: str  # baseline | shift | freq | umgf | unsharp | dt | mb
    augs: tuple[str, ...] = D4
    shifts: tuple[tuple[int, int], ...] = ((0, 0),)
    amount: float = 0.5
    sigma: float = 1.4
    high_edge: float = 0.85
    low_edge: float = 0.0
    u_amt: float = 0.15
    u_sig: float = 1.4
    dt_amount: float = 0.0
    mb_weights: tuple[float, ...] | None = None


def edge_path() -> Path:
    return EDGE_CKPT if EDGE_CKPT.exists() else EDGE_FALLBACK


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    if dy == 0 and dx == 0:
        return np.ascontiguousarray(img, dtype=np.float32)
    return np.roll(np.roll(img, int(dy), axis=0), int(dx), axis=1).astype(np.float32)


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4", family="baseline", shifts=((0, 0),))]

    shift_sets = [
        ((0, 0), (0, 8)),
        ((0, 0), (0, -8)),
        ((0, 0), (8, 0)),
        ((0, 0), (-8, 0)),
        ((0, 0), (0, 8), (0, -8)),
        ((0, 0), (8, 0), (-8, 0)),
        ((0, 0), (0, 8), (8, 0)),
        ((0, 0), (0, 8), (0, -8), (8, 0), (-8, 0)),
        ((0, 0), (8, 8), (-8, -8)),
        ((0, 0), (8, -8), (-8, 8)),
        ((0, 0), (0, 16)),
        ((0, 0), (16, 0)),
        ((0, 0), (0, 16), (16, 0), (0, -16), (-16, 0)),
        ((0, 0), (4, 0), (0, 4), (-4, 0), (0, -4)),
        ((0, 0), (12, 0), (0, 12)),
        ((0, 0), (8, 8), (8, -8), (-8, 8), (-8, -8)),
        ((0, 0), (0, 8), (8, 8), (8, 0)),
        ((0, 0), (6, 6), (-6, -6)),
        ((0, 0), (0, 10), (10, 0), (0, -10), (-10, 0), (10, 10)),
        ((0, 0), (0, 8), (0, 16), (8, 0), (16, 0)),
    ]
    for shifts in shift_sets:
        tag = "_".join(f"{dy}x{dx}" for dy, dx in shifts if (dy, dx) != (0, 0)) or "id"
        recipes.append(Recipe(name=f"shift_{tag}", family="shift", shifts=shifts))

    for sig, he, le in [
        (1.2, 0.85, 0.0),
        (1.5, 0.9, 0.0),
        (1.5, 0.75, 0.0),
        (2.0, 0.85, 0.0),
        (1.0, 1.0, 0.0),
        (1.5, 0.85, 0.1),
        (1.8, 0.7, 0.0),
        (1.2, 0.95, 0.05),
        (2.5, 0.8, 0.0),
        (1.5, 0.6, 0.0),
        (1.4, 0.9, 0.0),
        (1.6, 0.85, 0.05),
        (1.0, 0.8, 0.0),
        (2.0, 0.95, 0.0),
        (1.3, 0.75, 0.1),
    ]:
        recipes.append(
            Recipe(
                name=f"freq_s{sig:g}_he{he:g}_le{le:g}",
                family="freq",
                sigma=sig,
                high_edge=he,
                low_edge=le,
            )
        )

    for amt, sig in [
        (0.2, 1.4),
        (0.3, 1.4),
        (0.4, 1.2),
        (0.25, 1.6),
        (0.35, 1.5),
        (0.15, 1.4),
        (0.45, 1.3),
        (0.3, 1.8),
    ]:
        recipes.append(
            Recipe(name=f"umgf_a{amt:g}_s{sig:g}", family="umgf", amount=amt, sigma=sig)
        )

    for amt, sig in [
        (0.12, 1.6),
        (0.15, 1.4),
        (0.18, 1.2),
        (0.10, 1.8),
        (0.16, 1.5),
        (0.14, 1.3),
        (0.20, 1.4),
        (0.08, 1.6),
    ]:
        recipes.append(
            Recipe(name=f"u_a{amt:g}_s{sig:g}", family="unsharp", u_amt=amt, u_sig=sig)
        )
    for amt in [0.25, 0.35, 0.45, 0.55, 0.4, 0.3]:
        recipes.append(
            Recipe(name=f"dt_{amt:g}", family="dt", dt_amount=amt, sigma=1.2)
        )

    for wts in [
        (1.0, 0.7, 0.0),
        (1.0, 0.85, 0.1, 0.0),
        (0.95, 0.6, 0.0),
        (1.0, 0.5, 0.2, 0.0),
        (0.9, 0.75, 0.15, 0.0),
        (1.0, 0.8, 0.0),
    ]:
        recipes.append(
            Recipe(
                name=f"mb_{'_'.join(f'{w:g}' for w in wts)}",
                family="mb",
                mb_weights=wts,
            )
        )

    for shifts, ua in [
        (((0, 0), (0, 8), (0, -8)), 0.12),
        (((0, 0), (8, 0), (-8, 0)), 0.12),
        (((0, 0), (0, 8), (8, 0), (0, -8), (-8, 0)), 0.10),
        (((0, 0), (8, 8), (-8, -8)), 0.14),
    ]:
        tag = "_".join(f"{dy}x{dx}" for dy, dx in shifts if (dy, dx) != (0, 0))
        recipes.append(
            Recipe(
                name=f"shiftu_{tag}_u{ua:g}",
                family="shift",
                shifts=shifts,
                u_amt=ua,
                u_sig=1.4,
            )
        )

    recipes = recipes[:100]
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_shift_{i}",
                family="shift",
                shifts=((0, 0), (0, 4 + (i % 4) * 4), (4 + (i % 4) * 4, 0)),
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


def spatial_geom(model, fe, exposure_ms, device, n_in, tile, augs) -> np.ndarray:
    outs = []
    for aug in augs:
        fe_t, h, w = geom_forward(fe, aug)
        dn = _spatial_on_merged(model, fe_t, exposure_ms, device, n_in, tile)
        outs.append(geom_inverse(dn, aug, h, w))
    return np.mean(np.stack(outs, 0), 0).astype(np.float32)


def cache_d4_arms(files, metas, cache_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(edge_path(), None, device)
    for path, meta in zip(files, metas):
        stem = path.stem
        sp = cache_dir / f"{stem}_sota_d4.npy"
        ep = cache_dir / f"{stem}_edge_d4.npy"
        if sp.exists() and ep.exists():
            print(f"d4 cache hit {stem}", flush=True)
            continue
        fe = np.load(cache_dir / f"{stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4
        )
        edge = spatial_geom(
            edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4
        )
        np.save(sp, sota)
        np.save(ep, edge)
        print(f"cached d4 {stem}", flush=True)


class ShiftCache:
    """Lazy D4+FE-shift cache; loads models once."""

    def __init__(self, files, metas, cache_dir: Path):
        self.files = files
        self.metas = metas
        self.cache_dir = cache_dir
        self._models = None

    def _ensure_models(self):
        if self._models is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
            edge_m, _, _ = load_model(edge_path(), None, device)
            self._models = (device, sota_m, edge_m, n_in)
        return self._models

    def ensure(self, shifts: tuple[tuple[int, int], ...]) -> None:
        device, sota_m, edge_m, n_in = self._ensure_models()
        for dy, dx in shifts:
            if (dy, dx) == (0, 0):
                continue
            tag = f"{dy}x{dx}"
            for path, meta in zip(self.files, self.metas):
                stem = path.stem
                sp = self.cache_dir / f"{stem}_sota_d4_s{tag}.npy"
                ep = self.cache_dir / f"{stem}_edge_d4_s{tag}.npy"
                if sp.exists() and ep.exists():
                    continue
                fe = shift_image(np.load(self.cache_dir / f"{stem}_fe.npy"), dy, dx)
                sota = spatial_geom(
                    sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4
                )
                edge = spatial_geom(
                    edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4
                )
                np.save(sp, shift_image(sota, -dy, -dx))
                np.save(ep, shift_image(edge, -dy, -dx))
                print(f"cached shift {tag} {stem}", flush=True)


def load_d4_avg(
    cache_dir: Path, stem: str, arm: str, shifts: tuple[tuple[int, int], ...]
) -> np.ndarray:
    imgs = []
    for dy, dx in shifts:
        if (dy, dx) == (0, 0):
            p = cache_dir / f"{stem}_{arm}_d4.npy"
        else:
            p = cache_dir / f"{stem}_{arm}_d4_s{dy}x{dx}.npy"
        imgs.append(np.load(p))
    return np.mean(np.stack(imgs, 0), 0).astype(np.float32)


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float) -> np.ndarray:
    if recipe.family == "baseline":
        sota = np.load(cache_dir / f"{stem}_sota_d4.npy")
        edge = np.load(cache_dir / f"{stem}_edge_d4.npy")
        return deploy_p7_base(sota, edge, fps)

    if recipe.family == "shift":
        sota = load_d4_avg(cache_dir, stem, "sota", recipe.shifts)
        edge = load_d4_avg(cache_dir, stem, "edge", recipe.shifts)
        return deploy_p7_base(
            sota, edge, fps, unsharp_amount=recipe.u_amt, unsharp_sigma=recipe.u_sig
        )

    sota = np.load(cache_dir / f"{stem}_sota_d4.npy")
    edge = np.load(cache_dir / f"{stem}_edge_d4.npy")
    if recipe.family == "freq":
        fused = freq_split_fuse(
            sota,
            edge,
            sigma=recipe.sigma,
            high_edge=recipe.high_edge,
            low_edge=recipe.low_edge,
        )
        return deploy_p7_base(fused, fused, fps)

    if recipe.family == "umgf":
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0)
        return umgf_fuse(base, edge, amount=recipe.amount, sigma=recipe.sigma)

    if recipe.family == "unsharp":
        return deploy_p7_base(
            sota, edge, fps, unsharp_amount=recipe.u_amt, unsharp_sigma=recipe.u_sig
        )

    if recipe.family == "dt":
        base = deploy_p7_base(sota, edge, fps)
        return detail_transfer(
            base, edge, amount=recipe.dt_amount, sigma=recipe.sigma, edge_only=True
        )

    if recipe.family == "mb" and recipe.mb_weights:
        fused = multiband_fuse(sota, edge, list(recipe.mb_weights))
        return deploy_p7_base(fused, fused, fps)

    return deploy_p7_base(sota, edge, fps)


def summarize(des_list, f10, f05, f1) -> dict:
    return {
        "mean_des": float(sum(des_list) / max(len(des_list), 1)),
        "f10_ef": float(f10["edge_fidelity"]) if f10 else float("nan"),
        "f05_ef": float(f05["edge_fidelity"]) if f05 else float("nan"),
        "f05_ng": float(f05["noise_gain"]) if f05 else float("nan"),
        "f1_ef": float(f1["edge_fidelity"]) if f1 else float("nan"),
        "edge_safe": edge_safe(
            float(f10["edge_fidelity"]) if f10 else 0.0,
            float(f05["edge_fidelity"]) if f05 else 0.0,
            float(f1["edge_fidelity"]) if f1 else 0.0,
        ),
    }


def bake_deploy(best: dict) -> None:
    recipe = best.get("recipe") or {}
    fam = recipe.get("family", "")
    hook = Path("nafnet_denoise/deploy_p16_hook.json")
    hook.write_text(json.dumps(best, indent=2), encoding="utf-8")
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"mean DES ~0\.\d+",
        f"mean DES ~{best['mean_des']:.4f}",
        text,
        count=1,
    )
    if fam == "unsharp":
        text = re.sub(
            r"unsharp_amount: float = [0-9.]+",
            f"unsharp_amount: float = {float(recipe['u_amt'])}",
            text,
            count=1,
        )
        text = re.sub(
            r"unsharp_sigma: float = [0-9.]+",
            f"unsharp_sigma: float = {float(recipe['u_sig'])}",
            text,
            count=1,
        )
        path.write_text(text, encoding="utf-8")
        print("Baked unsharp into deploy", flush=True)
        return

    if fam == "shift":
        shifts = [list(s) for s in recipe.get("shifts", [[0, 0]])]
        # Prefer documenting in hook; patch FE shifts if helper signature matches.
        if "fe_shifts" not in text:
            helper = """def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile,
                       tta_augs=None):
    augs = list(tta_augs) if tta_augs else ["id"]
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    outs = []
    try:
        for aug in augs:
            fe, h, w = geom_forward(merged, aug)
            frames = [fe.copy() for _ in range(input_frames)]
            dn = denoise_from_dn_frames(model, frames, exposure_ms, device, tile=tile)
            outs.append(geom_inverse(dn, aug, h, w))
    finally:
        model.wiener_front_end = was
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)"""
            new_helper = f"""def _spatial_on_merged(model, merged, exposure_ms, device, input_frames, tile,
                       tta_augs=None, fe_shifts=None):
    augs = list(tta_augs) if tta_augs else ["id"]
    shifts = list(fe_shifts) if fe_shifts is not None else {shifts!r}
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    outs = []
    try:
        for dy, dx in shifts:
            base = merged
            if dy or dx:
                base = np.roll(np.roll(merged, int(dy), axis=0), int(dx), axis=1)
            for aug in augs:
                fe, h, w = geom_forward(base, aug)
                frames = [fe.copy() for _ in range(input_frames)]
                dn = denoise_from_dn_frames(
                    model, frames, exposure_ms, device, tile=tile
                )
                dn = geom_inverse(dn, aug, h, w)
                if dy or dx:
                    dn = np.roll(np.roll(dn, -int(dy), axis=0), -int(dx), axis=1)
                outs.append(dn)
    finally:
        model.wiener_front_end = was
    return np.mean(np.stack(outs, axis=0), axis=0).astype(np.float32)"""
            if helper in text:
                text = text.replace(helper, new_helper, 1)
                text = text.replace(
                    "tta_augs: list[str] | None = ['id', 'r90', 'r270', 'r90_lr', 'r270_lr'],",
                    "tta_augs: list[str] | None = ['id', 'r90', 'r270', 'r90_lr', 'r270_lr'],\n"
                    f"    fe_shifts: list[list[int]] | None = {shifts!r},",
                    1,
                )
                text = text.replace(
                    "tile, tta_augs=tta_augs\n    )",
                    "tile, tta_augs=tta_augs, fe_shifts=fe_shifts\n    )",
                )
        path.write_text(text, encoding="utf-8")
        print(f"Baked FE shifts {shifts} into deploy", flush=True)
        return

    path.write_text(text, encoding="utf-8")
    note = Path("nafnet_denoise/compare_meta100_p16/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} family={fam} DES={best['mean_des']:.4f}\n"
        f"See deploy_p16_hook.json\n",
        encoding="utf-8",
    )
    print(f"Bake note for family={fam}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p16")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=35)
    parser.add_argument(
        "--skip-shift",
        action="store_true",
        help="Skip FE-shift recipes (use after shift family saturates).",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    for path, _meta in zip(files, metas):
        if not (args.cache_dir / f"{path.stem}_fe.npy").exists():
            raise SystemExit(f"Missing FE cache for {path.stem}; run P10/P13 cache first")

    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Caching D4 arms...", flush=True)
    cache_d4_arms(files, metas, args.cache_dir)
    shift_cache = ShiftCache(files, metas, args.cache_dir)

    history: list[dict] = []
    best_safe: dict | None = {
        "iter": -1,
        "name": "baseline_seed",
        "family": "baseline",
        "mean_des": BASELINE,
        "edge_safe": True,
    }
    stagnant = 0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        if args.skip_shift and recipe.family == "shift":
            print(f"skip shift {recipe.name}", flush=True)
            continue
        print(f"\n===== P16 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        if recipe.family == "shift":
            shift_cache.ensure(recipe.shifts)
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
        sc = summarize(des_list, f10, f05, f1)
        rec = {
            "iter": idx,
            "name": recipe.name,
            "family": recipe.family,
            "recipe": asdict(recipe),
            **sc,
        }
        history.append(rec)
        if sc["edge_safe"] and sc["mean_des"] > best_safe["mean_des"] + 1e-4:
            best_safe = rec
            stagnant = 0
            (args.output_dir / "global_best.json").write_text(
                json.dumps(best_safe, indent=2), encoding="utf-8"
            )
            print(
                f"NEW SAFE BEST {recipe.name} DES={sc['mean_des']:.4f}",
                flush=True,
            )
        else:
            if recipe.family != "baseline":
                stagnant += 1
            print(
                f"{recipe.name}: DES={sc['mean_des']:.4f} "
                f"({'ok' if sc['edge_safe'] else 'unsafe'}) stagnant={stagnant}",
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
                f"Early stop: stagnant={stagnant} "
                f"best={None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P16 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        (args.output_dir / "bake_recipe.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(
            f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}",
            flush=True,
        )
        bake_deploy(best_safe)
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
