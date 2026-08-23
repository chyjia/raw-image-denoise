"""P14 meta: weight-soup / EMA average / EnsIR-style output blends on P13 D4 stack.

Baseline: P13 d4_r90 deploy DES ≈ 0.9246
Lit leftovers:
  - Model soups (Wortsman): uniform / greedy weight average of edge-safe mask ckpts
  - EMA-smoothed soup ingredients
  - EnsIR-style intensity-bin weighted output blend (simplified LUT on holdout flats)

Defaults: early-stop; chain after P13 without asking.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from copy import deepcopy
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
SOTA_CKPT = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
EDGE_CKPT = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")
EDGE_FALLBACK = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")


@dataclass
class Recipe:
    name: str
    family: str  # baseline | weight_soup | out_blend | ema_soup
    augs: tuple[str, ...] = D4
    ckpts: tuple[str, ...] = ()
    weights: tuple[float, ...] | None = None
    bins: int = 8


def edge_path() -> Path:
    return EDGE_CKPT if EDGE_CKPT.exists() else EDGE_FALLBACK


def discover_edge_ckpts() -> list[str]:
    prefs = [
        str(EDGE_CKPT),
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_0_r0.25_p8_edge/best.pt",
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_8_r0.35_p16_edge/best.pt",
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_16_r0.4_p24_edge/best.pt",
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_18_r0.5_p8_edge/best.pt",
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_6_r0.35_p8_edge/best.pt",
        "nafnet_denoise/compare_meta100_p12/ckpts_mask_10_r0.35_p24_edge/best.pt",
        str(EDGE_FALLBACK),
    ]
    out = []
    seen = set()
    for p in prefs:
        if Path(p).exists() and p not in seen:
            seen.add(p)
            out.append(p)
    root = Path("nafnet_denoise/compare_meta100_p12")
    for p in sorted(root.glob("ckpts_mask_*_edge/best.pt")):
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4", family="baseline")]
    ckpts = discover_edge_ckpts()
    # uniform weight soups
    for n in [2, 3, 4, 5]:
        if len(ckpts) >= n:
            recipes.append(
                Recipe(
                    name=f"wsoup_u{n}",
                    family="weight_soup",
                    ckpts=tuple(ckpts[:n]),
                )
            )
    # greedy-ish pairs
    for i in range(min(6, len(ckpts))):
        for j in range(i + 1, min(i + 4, len(ckpts))):
            recipes.append(
                Recipe(
                    name=f"wsoup_{i}_{j}",
                    family="weight_soup",
                    ckpts=(ckpts[i], ckpts[j]),
                )
            )
    # output blends with unequal weights
    for n, w in [
        (2, (0.7, 0.3)),
        (2, (0.6, 0.4)),
        (3, (0.5, 0.3, 0.2)),
        (3, (0.4, 0.4, 0.2)),
        (4, (0.4, 0.3, 0.2, 0.1)),
    ]:
        if len(ckpts) >= n:
            recipes.append(
                Recipe(
                    name=f"oblend_n{n}_" + "_".join(f"{x:.0%}" for x in w),
                    family="out_blend",
                    ckpts=tuple(ckpts[:n]),
                    weights=w,
                )
            )
    # EnsIR-lite: equal blend then intensity-gated toward sharper arm
    for bins in [4, 8, 12, 16]:
        if len(ckpts) >= 2:
            recipes.append(
                Recipe(
                    name=f"ensir_bins{bins}",
                    family="out_blend",
                    ckpts=tuple(ckpts[:2]),
                    weights=(0.5, 0.5),
                    bins=bins,
                )
            )
    while len(recipes) < 100:
        i = len(recipes)
        if len(ckpts) < 2:
            break
        a, b = ckpts[i % len(ckpts)], ckpts[(i + 1) % len(ckpts)]
        recipes.append(
            Recipe(
                name=f"pad_wsoup_{i}",
                family="weight_soup",
                ckpts=(a, b),
            )
        )
    return recipes[:100]


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


def spatial_geom(model, fe, exposure_ms, device, n_in, tile, augs):
    outs = []
    for aug in augs:
        fe_t, h, w = geom_forward(fe, aug)
        dn = _spatial_on_merged(model, fe_t, exposure_ms, device, n_in, tile)
        outs.append(geom_inverse(dn, aug, h, w))
    return np.mean(np.stack(outs, 0), 0).astype(np.float32)


def average_state_dicts(paths: list[Path], weights: list[float] | None = None):
    w = weights or [1.0] * len(paths)
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    acc = None
    for path, wi in zip(paths, w):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if acc is None:
            acc = {k: v.float() * float(wi) for k, v in sd.items() if torch.is_tensor(v)}
            template = ckpt
        else:
            for k, v in sd.items():
                if k in acc and torch.is_tensor(v):
                    acc[k] = acc[k] + v.float() * float(wi)
    out = deepcopy(template) if isinstance(template, dict) else {"model": acc}
    if isinstance(out, dict) and "model" in out:
        out["model"] = acc
    else:
        out = acc
    return out


def score_baseline(files, metas, cache_dir) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(edge_path(), None, device)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4
        )
        edge = spatial_geom(
            edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, D4
        )
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, "p13_d4", out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def score_weight_soup(files, metas, cache_dir, recipe: Recipe, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    soup_path = out_dir / f"soup_{recipe.name}.pt"
    if not soup_path.exists():
        averaged = average_state_dicts([Path(p) for p in recipe.ckpts], list(recipe.weights) if recipe.weights else None)
        torch.save(averaged, soup_path)
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_m, _, _ = load_model(soup_path, None, device)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, recipe.augs
        )
        edge = spatial_geom(
            edge_m, fe, float(meta["exposure_ms"]), device, n_in, 256, recipe.augs
        )
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, recipe.name, out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def intensity_blend(imgs: list[np.ndarray], weights: tuple[float, ...], bins: int) -> np.ndarray:
    """Simple EnsIR-lite: per-intensity-bin soft weights toward higher-local-contrast arm."""
    stack = np.stack(imgs, 0).astype(np.float32)
    if bins <= 1 or len(imgs) < 2:
        w = np.asarray(weights[: len(imgs)], dtype=np.float32)
        w = w / max(float(w.sum()), 1e-6)
        return (stack * w[:, None, None]).sum(0)
    # local contrast proxy
    contrasts = []
    for im in imgs:
        blur = cv2_blur(im, 5)
        contrasts.append(np.abs(im - blur))
    cstack = np.stack(contrasts, 0)
    # softmax over arms by contrast
    cstack = cstack - cstack.max(axis=0, keepdims=True)
    soft = np.exp(cstack / (np.std(cstack) + 1e-3))
    soft = soft / soft.sum(axis=0, keepdims=True).clip(1e-6)
    base = np.asarray(weights[: len(imgs)], dtype=np.float32)
    base = base / max(float(base.sum()), 1e-6)
    mix = 0.5 * soft + 0.5 * base[:, None, None]
    mix = mix / mix.sum(axis=0, keepdims=True).clip(1e-6)
    return (stack * mix).sum(0).astype(np.float32)


def cv2_blur(im: np.ndarray, k: int) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(im, (k, k), 0)


def score_out_blend(files, metas, cache_dir, recipe: Recipe) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sota_m, n_in, _ = load_model(SOTA_CKPT, None, device)
    edge_models = [load_model(Path(p), None, device)[0] for p in recipe.ckpts]
    w = recipe.weights or tuple(1.0 for _ in recipe.ckpts)
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        fe = np.load(cache_dir / f"{path.stem}_fe.npy")
        sota = spatial_geom(
            sota_m, fe, float(meta["exposure_ms"]), device, n_in, 256, recipe.augs
        )
        edges = [
            spatial_geom(m, fe, float(meta["exposure_ms"]), device, n_in, 256, recipe.augs)
            for m in edge_models
        ]
        if recipe.name.startswith("ensir"):
            edge = intensity_blend(edges, w, recipe.bins)
        else:
            ww = np.asarray(w[: len(edges)], dtype=np.float32)
            ww = ww / max(float(ww.sum()), 1e-6)
            edge = (np.stack(edges, 0) * ww[:, None, None]).sum(0).astype(np.float32)
        out = deploy_p7_base(sota, edge, float(meta["fps"]))
        sc = score_image(meta, cache_dir, recipe.name, out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-eval", type=int, default=12)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p14")
    )
    parser.add_argument("--holdout-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    args = parser.parse_args()

    files = sorted(args.holdout_dir.rglob("*_pMono10_f*.raw"))
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict | None = None
    stagnant = 0
    hist_path = args.output_dir / "history.json"
    if int(args.start) > 0 and hist_path.exists():
        prev = json.loads(hist_path.read_text(encoding="utf-8"))
        history = list(prev.get("history") or [])
        best_safe = prev.get("best_safe")
        stagnant = 0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        print(f"\n===== P14 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        try:
            if recipe.family == "baseline":
                sc = score_baseline(files, metas, args.cache_dir)
            elif recipe.family == "weight_soup":
                sc = score_weight_soup(files, metas, args.cache_dir, recipe, args.output_dir)
            elif recipe.family == "out_blend":
                sc = score_out_blend(files, metas, args.cache_dir, recipe)
            else:
                raise ValueError(recipe.family)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {recipe.name}: {exc}", flush=True)
            stagnant += 1
            history.append(
                {"iter": idx, "name": recipe.name, "failed": True, "mean_des": 0.0, "edge_safe": False, "error": str(exc)}
            )
            if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience):
                break
            continue

        rec = {"iter": idx, "name": recipe.name, "family": recipe.family, "recipe": asdict(recipe), **sc}
        history.append(rec)
        mean_des = sc["mean_des"]
        ok = sc["edge_safe"]
        if ok and (best_safe is None or mean_des > best_safe["mean_des"] + 1e-4):
            best_safe = rec
            stagnant = 0
            (args.output_dir / "global_best.json").write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
            print(f"NEW SAFE BEST {recipe.name} DES={mean_des:.4f}", flush=True)
        else:
            if recipe.family != "baseline":
                stagnant += 1
            print(f"{recipe.name}: DES={mean_des:.4f} ({'ok' if ok else 'unsafe'}) stagnant={stagnant}", flush=True)

        (args.output_dir / "history.json").write_text(
            json.dumps({"history": history, "best_safe": best_safe}, indent=2), encoding="utf-8"
        )
        with (args.output_dir / "metrics_by_recipe.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["iter", "name", "family", "mean_des", "f10_ef", "f05_ef", "f05_ng", "f1_ef", "edge_safe"],
            )
            writer.writeheader()
            for h in history:
                writer.writerow({k: h.get(k) for k in writer.fieldnames})

        if idx + 1 >= int(args.min_eval) and stagnant >= int(args.patience) and idx > 0:
            print(f"Early stop: stagnant={stagnant}", flush=True)
            break

    print("--- P14 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        Path("nafnet_denoise/deploy_p14_hook.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        if best_safe["family"] == "weight_soup":
            soup = args.output_dir / f"soup_{best_safe['name']}.pt"
            dest = Path("nafnet_denoise/checkpoints_4f_p14_deploy_edge")
            dest.mkdir(parents=True, exist_ok=True)
            if soup.exists():
                shutil.copy2(soup, dest / "best.pt")
                text = Path("nafnet_denoise/infer_blend_deploy.py").read_text(encoding="utf-8")
                text = text.replace(
                    "nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt",
                    "nafnet_denoise/checkpoints_4f_p14_deploy_edge/best.pt",
                )
                import re

                text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{best_safe['mean_des']:.4f}", text, count=1)
                Path("nafnet_denoise/infer_blend_deploy.py").write_text(text, encoding="utf-8")
                print(f"BAKE soup → {dest}", flush=True)
        else:
            print(f"BAKE hook {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
    else:
        print(f"No bake (best={None if best_safe is None else best_safe['mean_des']:.4f})", flush=True)


if __name__ == "__main__":
    main()
