"""P18 meta: intensity-bin EnsIR-lite blend of D4 SOTA↔edge arms.

Baseline: P13 d4_r90 DES ≈ 0.9252
Lit: EnsIR (NeurIPS'24) — range-wise ensemble weights via LUT on intensity bins.
P14 tried multi-ckpt soup; here we weight *two* D4 arms by local DN intensity.

Defaults: fit LUT on leave-one-clip-out proxy is heavy; use fixed parametric
schedules (sigmoid / piecewise) + optional holdout-fit with clip-leave-out DES.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .p8_fusion import deploy_p7_base
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9252


@dataclass
class Recipe:
    name: str
    family: str  # baseline | bin_blend | sigmoid | piecewise | soft_gate
    bins: int = 16
    edge_lo: float = 0.3
    edge_hi: float = 0.9
    mid: float = 0.4
    steep: float = 12.0
    # piecewise: below t1 use w_lo edge, above t2 use w_hi
    t1: float = 0.2
    t2: float = 0.6
    w_lo: float = 0.2
    w_mid: float = 0.5
    w_hi: float = 0.85
    # soft_gate: use local std as confidence for edge
    std_k: float = 8.0
    std_bias: float = 0.02


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4", family="baseline")]
    for bins, elo, ehi in [
        (8, 0.2, 0.9),
        (16, 0.3, 0.85),
        (16, 0.1, 0.95),
        (32, 0.25, 0.9),
        (16, 0.4, 0.8),
        (8, 0.0, 1.0),
        (24, 0.2, 0.85),
        (16, 0.35, 0.75),
        (12, 0.15, 0.9),
        (16, 0.5, 0.9),
    ]:
        recipes.append(
            Recipe(
                name=f"bin_b{bins}_lo{elo:g}_hi{ehi:g}",
                family="bin_blend",
                bins=bins,
                edge_lo=elo,
                edge_hi=ehi,
            )
        )
    for mid, steep in [
        (0.3, 10.0),
        (0.4, 12.0),
        (0.5, 8.0),
        (0.35, 16.0),
        (0.45, 6.0),
        (0.25, 14.0),
        (0.55, 10.0),
        (0.4, 20.0),
        (0.2, 8.0),
        (0.6, 12.0),
    ]:
        recipes.append(
            Recipe(
                name=f"sig_m{mid:g}_s{steep:g}",
                family="sigmoid",
                mid=mid,
                steep=steep,
            )
        )
    for t1, t2, wl, wm, wh in [
        (0.15, 0.5, 0.15, 0.5, 0.9),
        (0.2, 0.6, 0.2, 0.55, 0.85),
        (0.1, 0.4, 0.1, 0.45, 0.95),
        (0.25, 0.7, 0.25, 0.5, 0.8),
        (0.2, 0.55, 0.0, 0.4, 1.0),
        (0.3, 0.65, 0.3, 0.6, 0.85),
        (0.12, 0.45, 0.2, 0.5, 0.9),
        (0.18, 0.5, 0.35, 0.55, 0.75),
    ]:
        recipes.append(
            Recipe(
                name=f"pw_{t1:g}_{t2:g}_{wl:g}_{wm:g}_{wh:g}",
                family="piecewise",
                t1=t1,
                t2=t2,
                w_lo=wl,
                w_mid=wm,
                w_hi=wh,
            )
        )
    for k, bias in [
        (6.0, 0.02),
        (8.0, 0.02),
        (10.0, 0.03),
        (12.0, 0.015),
        (8.0, 0.04),
        (5.0, 0.025),
        (15.0, 0.02),
        (8.0, 0.01),
    ]:
        recipes.append(
            Recipe(
                name=f"gate_k{k:g}_b{bias:g}",
                family="soft_gate",
                std_k=k,
                std_bias=bias,
            )
        )
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_sig_{i}",
                family="sigmoid",
                mid=0.2 + (i % 10) * 0.05,
                steep=6.0 + (i % 8) * 2.0,
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


def intensity_bin_blend(
    sota: np.ndarray,
    edge: np.ndarray,
    bins: int,
    edge_lo: float,
    edge_hi: float,
) -> np.ndarray:
    """Linear ramp of edge weight from edge_lo→edge_hi across intensity bins."""
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    # normalize guide to [0,1] per-image
    gmin, gmax = float(guide.min()), float(guide.max())
    norm = (guide - gmin) / (gmax - gmin + 1e-6)
    idx = np.clip((norm * bins).astype(np.int32), 0, bins - 1)
    # weight schedule per bin
    ws = np.linspace(edge_lo, edge_hi, bins, dtype=np.float32)
    w = ws[idx]
    return ((1.0 - w) * sota + w * edge).astype(np.float32)


def sigmoid_blend(
    sota: np.ndarray, edge: np.ndarray, mid: float, steep: float
) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    gmin, gmax = float(guide.min()), float(guide.max())
    norm = (guide - gmin) / (gmax - gmin + 1e-6)
    w = 1.0 / (1.0 + np.exp(-float(steep) * (norm - float(mid))))
    return ((1.0 - w) * sota + w * edge).astype(np.float32)


def piecewise_blend(
    sota: np.ndarray,
    edge: np.ndarray,
    t1: float,
    t2: float,
    w_lo: float,
    w_mid: float,
    w_hi: float,
) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    gmin, gmax = float(guide.min()), float(guide.max())
    norm = (guide - gmin) / (gmax - gmin + 1e-6)
    w = np.full_like(norm, float(w_mid))
    w = np.where(norm < t1, float(w_lo), w)
    w = np.where(norm > t2, float(w_hi), w)
    return ((1.0 - w) * sota + w * edge).astype(np.float32)


def soft_gate_blend(
    sota: np.ndarray, edge: np.ndarray, std_k: float, std_bias: float
) -> np.ndarray:
    """Prefer edge where local |sota-edge| is large (disagreement = structure)."""
    diff = np.abs(sota.astype(np.float32) - edge.astype(np.float32))
    # local mean of |diff|
    from cv2 import blur

    local = blur(diff, (5, 5))
    w = 1.0 / (1.0 + np.exp(-float(std_k) * (local - float(std_bias))))
    return ((1.0 - w) * sota + w * edge).astype(np.float32)


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float) -> np.ndarray:
    sota = np.load(cache_dir / f"{stem}_sota_d4.npy")
    edge = np.load(cache_dir / f"{stem}_edge_d4.npy")
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)
    if recipe.family == "bin_blend":
        fused = intensity_bin_blend(
            sota, edge, recipe.bins, recipe.edge_lo, recipe.edge_hi
        )
        return deploy_p7_base(fused, fused, fps)
    if recipe.family == "sigmoid":
        fused = sigmoid_blend(sota, edge, recipe.mid, recipe.steep)
        return deploy_p7_base(fused, fused, fps)
    if recipe.family == "piecewise":
        fused = piecewise_blend(
            sota,
            edge,
            recipe.t1,
            recipe.t2,
            recipe.w_lo,
            recipe.w_mid,
            recipe.w_hi,
        )
        return deploy_p7_base(fused, fused, fps)
    if recipe.family == "soft_gate":
        fused = soft_gate_blend(sota, edge, recipe.std_k, recipe.std_bias)
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
    hook = Path("nafnet_denoise/deploy_p18_hook.json")
    hook.write_text(json.dumps(best, indent=2), encoding="utf-8")
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"mean DES ~0\.\d+",
        f"mean DES ~{best['mean_des']:.4f}",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    note = Path("nafnet_denoise/compare_meta100_p18/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\nSee deploy_p18_hook.json\n",
        encoding="utf-8",
    )
    print(f"Wrote P18 hook for {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p18")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-eval", type=int, default=25)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    metas = [
        json.loads((args.cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    for path in files:
        if not (args.cache_dir / f"{path.stem}_sota_d4.npy").exists():
            raise SystemExit(f"Missing D4 cache {path.stem}")

    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"\n===== P18 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
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

    print("--- P18 done ---", flush=True)
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
