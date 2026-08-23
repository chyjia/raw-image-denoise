"""P19 meta: expand geometric TTA beyond P13 D4 (D8 / extra flips).

Baseline: P13 d4_r90 DES ≈ 0.9252
Lit: NTIRE self-ensemble; PatchFusion-style multi-view average.
Tries larger Dihedral sets on FE→spatial; uses lazy cache like P16.
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
from .p8_fusion import deploy_p7_base
from .p13_tta import geom_forward, geom_inverse
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9252
D4 = ("id", "r90", "r270", "r90_lr", "r270_lr")
D4B = ("id", "r90", "r180", "r270", "lr", "ud")
D8 = ("id", "lr", "ud", "udlr", "r90", "r180", "r270", "r90_lr")
D8B = ("id", "lr", "ud", "udlr", "r90", "r180", "r270", "r90_lr", "r270_lr", "r90_ud")
SOTA_CKPT = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
EDGE_CKPT = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")
EDGE_FALLBACK = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")


@dataclass
class Recipe:
    name: str
    family: str  # baseline | geom
    augs: tuple[str, ...] = D4
    # optional: different augs per arm
    edge_augs: tuple[str, ...] | None = None


def edge_path() -> Path:
    return EDGE_CKPT if EDGE_CKPT.exists() else EDGE_FALLBACK


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4", family="baseline", augs=D4)]
    sets = [
        ("d4b", D4B),
        ("d8", D8),
        ("d8b", D8B),
        ("id_lr_ud_udlr", ("id", "lr", "ud", "udlr")),
        ("id_r90_r180_r270", ("id", "r90", "r180", "r270")),
        ("d4_plus_r180", ("id", "r90", "r180", "r270", "r90_lr", "r270_lr")),
        ("d4_plus_lr", ("id", "lr", "r90", "r270", "r90_lr", "r270_lr")),
        ("d4_plus_ud", ("id", "ud", "r90", "r270", "r90_lr", "r270_lr")),
        ("compact5b", ("id", "r90", "r180", "lr", "ud")),
        ("compact6", ("id", "r90", "r270", "lr", "ud", "udlr")),
        ("r90_only_extra", ("id", "r90", "r270", "r180", "r90_lr")),
        ("flip_heavy", ("id", "lr", "ud", "udlr", "r90_lr", "r270_lr")),
    ]
    for name, augs in sets:
        recipes.append(Recipe(name=f"geom_{name}", family="geom", augs=augs))
    # asymmetric: stronger TTA on edge only
    for name, augs in [("d8", D8), ("d4b", D4B), ("d8b", D8B)]:
        recipes.append(
            Recipe(
                name=f"edge_{name}_sota_d4",
                family="geom",
                augs=D4,
                edge_augs=augs,
            )
        )
        recipes.append(
            Recipe(
                name=f"sota_{name}_edge_d4",
                family="geom",
                augs=augs,
                edge_augs=D4,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        # pad with subsets of D8
        subset = list(D8)[: 2 + (i % 7)]
        recipes.append(
            Recipe(name=f"pad_geom_{i}", family="geom", augs=tuple(subset))
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


def aug_tag(augs: tuple[str, ...]) -> str:
    return "_".join(augs)


def cache_key(stem: str, arm: str, augs: tuple[str, ...]) -> str:
    return f"{stem}_{arm}_g_{aug_tag(augs)}.npy"


class GeomCache:
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

    def get(self, stem: str, arm: str, augs: tuple[str, ...], path, meta) -> np.ndarray:
        # reuse P16 D4 cache
        if augs == D4:
            p = self.cache_dir / f"{stem}_{arm}_d4.npy"
            if p.exists():
                return np.load(p)
        p = self.cache_dir / cache_key(stem, arm, augs)
        if p.exists():
            return np.load(p)
        device, sota_m, edge_m, n_in = self._ensure_models()
        model = sota_m if arm == "sota" else edge_m
        fe = np.load(self.cache_dir / f"{stem}_fe.npy")
        out = spatial_geom(
            model, fe, float(meta["exposure_ms"]), device, n_in, 256, augs
        )
        np.save(p, out)
        print(f"cached {arm} {aug_tag(augs)[:40]} {stem}", flush=True)
        return out


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
    augs = list(recipe.get("augs") or D4)
    edge_augs = recipe.get("edge_augs")
    hook = Path("nafnet_denoise/deploy_p19_hook.json")
    hook.write_text(json.dumps(best, indent=2), encoding="utf-8")
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"mean DES ~0\.\d+",
        f"mean DES ~{best['mean_des']:.4f}",
        text,
        count=1,
    )
    # update default tta_augs if symmetric
    if edge_augs is None:
        new_list = "[" + ", ".join(repr(a) for a in augs) + "]"
        text = re.sub(
            r"tta_augs: list\[str\] \| None = \[[^\]]+\]",
            f"tta_augs: list[str] | None = {new_list}",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"Baked TTA augs {augs} into deploy", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p19")
    )
    parser.add_argument("--max-recipes", type=int, default=30)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-eval", type=int, default=12)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = GeomCache(files, metas, args.cache_dir)

    history: list[dict] = []
    best_safe: dict = {
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
        print(f"\n===== P19 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        sota_augs = recipe.augs
        edge_augs = recipe.edge_augs if recipe.edge_augs is not None else recipe.augs
        des_list = []
        f10 = f05 = f1 = None
        for path, meta in zip(files, metas):
            sota = cache.get(path.stem, "sota", sota_augs, path, meta)
            edge = cache.get(path.stem, "edge", edge_augs, path, meta)
            out = deploy_p7_base(sota, edge, float(meta["fps"]))
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
            print(f"NEW SAFE BEST {recipe.name} DES={sc['mean_des']:.4f}", flush=True)
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

        if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience) and idx > 0:
            print(
                f"Early stop: stagnant={stagnant} best={best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P19 done ---", flush=True)
    if best_safe["mean_des"] >= BASELINE + 1e-4 and best_safe.get("name") != "baseline_seed":
        (args.output_dir / "bake_recipe.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        bake_deploy(best_safe)
    else:
        print(f"No bake (best={best_safe['mean_des']:.4f})", flush=True)


if __name__ == "__main__":
    main()
