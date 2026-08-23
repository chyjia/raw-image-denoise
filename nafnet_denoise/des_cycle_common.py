"""Shared DES cycle helpers (P28+). Baseline: P22 mean DES ≈ 0.9290."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .p8_fusion import deploy_p7_base
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9589
P19_TAG = "id_r90_r180_r270_r90_lr_r270_lr"
SOTA_CKPT = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
EDGE_CKPT = Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt")


def load_arms(cache_dir: Path, stem: str, tag: str = P19_TAG) -> tuple[np.ndarray, np.ndarray]:
    sp = cache_dir / f"{stem}_sota_g_{tag}.npy"
    ep = cache_dir / f"{stem}_edge_g_{tag}.npy"
    if sp.exists() and ep.exists():
        return np.load(sp), np.load(ep)
    # fall back to plain D4
    return (
        np.load(cache_dir / f"{stem}_sota_d4.npy"),
        np.load(cache_dir / f"{stem}_edge_d4.npy"),
    )


def available_geom_tags(cache_dir: Path) -> list[str]:
    tags = sorted(
        {
            f.name.split("_sota_g_", 1)[1][:-4]
            for f in cache_dir.glob("*_sota_g_*.npy")
            if "_sota_g_" in f.name
        }
    )
    # keep only tags that have matching edge for all stems later; filter here loosely
    return tags


def load_arm_tag(cache_dir: Path, stem: str, arm: str, tag: str) -> np.ndarray | None:
    p = cache_dir / f"{stem}_{arm}_g_{tag}.npy"
    if p.exists():
        return np.load(p)
    if tag == "d4" or tag == P19_TAG:
        # try d4 shortcut
        q = cache_dir / f"{stem}_{arm}_d4.npy"
        if q.exists() and tag == "d4":
            return np.load(q)
    return None


def ensemble_arms(
    cache_dir: Path,
    stem: str,
    tags: tuple[str, ...],
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    sotas, edges = [], []
    for tag in tags:
        s = load_arm_tag(cache_dir, stem, "sota", tag)
        e = load_arm_tag(cache_dir, stem, "edge", tag)
        if s is None or e is None:
            continue
        sotas.append(s)
        edges.append(e)
    if not sotas:
        return load_arms(cache_dir, stem)
    S = np.stack(sotas, 0)
    E = np.stack(edges, 0)
    if mode == "median":
        return np.median(S, 0).astype(np.float32), np.median(E, 0).astype(np.float32)
    if mode == "maxabs":
        # pick per-pixel arm farther from mean (detail-biased) — skip, use mean
        return np.mean(S, 0).astype(np.float32), np.mean(E, 0).astype(np.float32)
    return np.mean(S, 0).astype(np.float32), np.mean(E, 0).astype(np.float32)


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


def score_holdout(files, metas, cache_dir, name, make_out) -> dict:
    des_list, f10, f05, f1 = [], None, None, None
    for path, meta in zip(files, metas):
        out = make_out(path.stem, float(meta["fps"]))
        sc = score_image(meta, cache_dir, name, out)
        des_list.append(sc["des"])
        if F10 in path.name:
            f10 = sc
        if F05 in path.name:
            f05 = sc
        if F1 in path.name:
            f1 = sc
    return summarize(des_list, f10, f05, f1)


def stamp_deploy_des(mean_des: float) -> None:
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"mean DES ~0\.\d+", f"mean DES ~{mean_des:.4f}", text, count=1)
    path.write_text(text, encoding="utf-8")


def run_meta_loop(
    *,
    cycle: str,
    recipes: list,
    apply_fn,
    output_dir: Path,
    input_dir: Path,
    cache_dir: Path,
    baseline: float,
    patience: int,
    min_eval: int,
    bake_fn,
) -> dict | None:
    files = sorted(input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {input_dir}")
    metas = [
        json.loads((cache_dir / f"{p.stem}.json").read_text(encoding="utf-8"))
        for p in files
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict = {
        "iter": -1,
        "name": "baseline_seed",
        "family": "baseline",
        "mean_des": baseline,
        "edge_safe": True,
    }
    stagnant = 0

    for idx, recipe in enumerate(recipes):
        name = recipe["name"] if isinstance(recipe, dict) else recipe.name
        family = recipe.get("family") if isinstance(recipe, dict) else recipe.family
        print(f"\n===== {cycle} [{idx:03d}/{len(recipes)}] {name} =====", flush=True)

        def make_out(stem, fps, _r=recipe):
            return apply_fn(_r, cache_dir, stem, fps)

        sc = score_holdout(files, metas, cache_dir, name, make_out)
        rec = {
            "iter": idx,
            "name": name,
            "family": family,
            "recipe": recipe if isinstance(recipe, dict) else None,
            **sc,
        }
        if hasattr(recipe, "__dataclass_fields__"):
            from dataclasses import asdict

            rec["recipe"] = asdict(recipe)
        history.append(rec)
        if sc["edge_safe"] and sc["mean_des"] > best_safe["mean_des"] + 1e-4:
            best_safe = rec
            stagnant = 0
            (output_dir / "global_best.json").write_text(
                json.dumps(best_safe, indent=2), encoding="utf-8"
            )
            print(f"NEW SAFE BEST {name} DES={sc['mean_des']:.4f}", flush=True)
        else:
            if family != "baseline":
                stagnant += 1
            print(
                f"{name}: DES={sc['mean_des']:.4f} "
                f"({'ok' if sc['edge_safe'] else 'unsafe'}) stagnant={stagnant}",
                flush=True,
            )
        (output_dir / "history.json").write_text(
            json.dumps({"history": history, "best_safe": best_safe}, indent=2),
            encoding="utf-8",
        )
        with (output_dir / "metrics_by_recipe.csv").open(
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
        if idx + 1 >= min_eval and stagnant >= patience and idx > 0:
            print(
                f"Early stop: stagnant={stagnant} best={best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print(f"--- {cycle} done ---", flush=True)
    if best_safe["mean_des"] >= baseline + 1e-4 and best_safe.get("name") != "baseline_seed":
        (output_dir / "bake_recipe.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        bake_fn(best_safe)
        return best_safe
    print(f"No bake (best={best_safe['mean_des']:.4f})", flush=True)
    return None
