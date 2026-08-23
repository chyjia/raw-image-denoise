"""P7: 100 lit-inspired DES trials (non-bilateral-strength axes).

Paper map (after Gen2 0.9175 bilat saturation):
  DualExNet / NTIRE26  → residual |Δ| gate + HF residual
  SNR-ADDNet           → local-variance gate
  He guided filter     → flat guided boost (alt to bilat)
  MKPN / DualEx HF     → Gaussian freq-split fuse
  NTIRE self-ensemble  → geometric TTA on DN
  AIM25 / edge recover → edge-only unsharp
  Morph gate           → dilate/erode soft edge before blend

Constraints: edge_safe (f10_ef≥0.965, f05_ef≥0.870, f1_ef≥0.960).
Baseline: gen2_deploy (0.9175). Bake only if safe mean > baseline + 1e-4.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost
from .p6_fusion import detail_transfer, multi_cue_edge_map, blend_with_edge_map
from .p7_fusion import (
    edge_unsharp,
    flat_gaussian_boost,
    flat_guided_boost,
    flat_median_boost,
    freq_split_fuse,
    gen2_base_then,
    geometric_self_ensemble,
    local_variance_gate_blend,
    morph_edge_blend,
    residual_gate_blend,
)
from .run_des_loop100 import (
    F1,
    F05,
    F10,
    cache_forwards,
    edge_safe,
    score_image,
)


@dataclass
class Recipe:
    name: str
    family: str = "gen2"
    # shared blend
    temperature: float = 8.0
    harden: float = 16.0
    # residual / var gate
    gate_temp: float = 12.0
    gate_harden: float = 16.0
    gate_pct: float = 90.0
    gate_mix: float = 1.0
    var_win: int = 7
    edge_bias: float = 0.5
    # freq split
    freq_sigma: float = 1.5
    high_edge: float = 1.0
    low_edge: float = 0.0
    # morph
    dilate: int = 0
    erode: int = 0
    # post on gen2 / blend
    post: str = "none"  # none|guided|median|gauss|unsharp|tta|dt
    post_strength: float = 0.0
    post_kw: dict = field(default_factory=dict)
    # fps schedule for non-gen2 families
    low_fps: float = 1.0
    mid_fps: float = 5.0
    low_bilat: float = 0.75
    mid_bilat: float = 0.7
    bilat_harden: float = 40.0
    # detail transfer
    dt_amount: float = 0.0
    # use gen2 bilat schedule after custom blend
    apply_gen2_bilat: bool = True


def build_100_recipes() -> list[Recipe]:
    recipes: list[Recipe] = []
    recipes.append(Recipe(name="gen2_deploy", family="gen2"))

    # 1-20: residual gate (DualExNet)
    for i, (gt, gh, pct, mix) in enumerate(
        [
            (8.0, 12.0, 90.0, 1.0),
            (10.0, 16.0, 90.0, 1.0),
            (12.0, 16.0, 90.0, 1.0),
            (14.0, 20.0, 90.0, 1.0),
            (16.0, 16.0, 85.0, 1.0),
            (12.0, 24.0, 90.0, 1.0),
            (10.0, 16.0, 80.0, 1.0),
            (12.0, 16.0, 95.0, 1.0),
            (8.0, 16.0, 90.0, 0.85),
            (12.0, 16.0, 90.0, 0.7),
            (12.0, 16.0, 90.0, 0.55),
            (6.0, 12.0, 90.0, 1.0),
            (20.0, 16.0, 90.0, 1.0),
            (12.0, 8.0, 90.0, 1.0),
            (12.0, 32.0, 90.0, 0.8),
            (10.0, 16.0, 75.0, 0.9),
            (14.0, 16.0, 92.0, 0.75),
            (12.0, 16.0, 88.0, 0.65),
            (9.0, 18.0, 90.0, 1.0),
            (15.0, 14.0, 90.0, 0.9),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"res_t{gt:g}_h{gh:g}_p{pct:g}_m{mix:g}",
                family="residual",
                gate_temp=gt,
                gate_harden=gh,
                gate_pct=pct,
                gate_mix=mix,
            )
        )

    # 21-35: local variance gate
    for win, gt, bias, pct in [
        (5, 10.0, 0.5, 85.0),
        (7, 10.0, 0.5, 85.0),
        (9, 10.0, 0.5, 85.0),
        (7, 8.0, 0.35, 85.0),
        (7, 12.0, 0.65, 85.0),
        (7, 10.0, 0.5, 75.0),
        (7, 10.0, 0.5, 90.0),
        (5, 12.0, 0.7, 85.0),
        (11, 10.0, 0.4, 85.0),
        (7, 14.0, 0.55, 80.0),
        (7, 10.0, 0.25, 85.0),
        (7, 10.0, 0.8, 85.0),
        (9, 12.0, 0.5, 90.0),
        (5, 8.0, 0.6, 80.0),
        (7, 16.0, 0.5, 85.0),
    ]:
        recipes.append(
            Recipe(
                name=f"var_w{win}_t{gt:g}_b{bias:g}_p{pct:g}",
                family="variance",
                var_win=win,
                gate_temp=gt,
                edge_bias=bias,
                gate_pct=pct,
            )
        )

    # 36-50: freq-split
    for sig, he, le in [
        (1.0, 1.0, 0.0),
        (1.2, 1.0, 0.0),
        (1.5, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (2.5, 1.0, 0.0),
        (1.5, 0.85, 0.0),
        (1.5, 0.7, 0.0),
        (1.5, 1.0, 0.1),
        (1.5, 1.0, 0.2),
        (1.2, 0.9, 0.05),
        (1.8, 0.95, 0.0),
        (1.0, 0.8, 0.0),
        (3.0, 1.0, 0.0),
        (1.5, 0.6, 0.0),
        (2.0, 0.85, 0.1),
    ]:
        recipes.append(
            Recipe(
                name=f"freq_s{sig:g}_he{he:g}_le{le:g}",
                family="freq",
                freq_sigma=sig,
                high_edge=he,
                low_edge=le,
            )
        )

    # 51-60: morph edge
    for d, e, h in [
        (1, 0, 16.0),
        (2, 0, 16.0),
        (1, 1, 16.0),
        (0, 1, 16.0),
        (2, 1, 16.0),
        (1, 0, 24.0),
        (1, 0, 12.0),
        (3, 0, 16.0),
        (2, 0, 24.0),
        (1, 2, 16.0),
    ]:
        recipes.append(
            Recipe(
                name=f"morph_d{d}_e{e}_h{h:g}",
                family="morph",
                dilate=d,
                erode=e,
                harden=h,
            )
        )

    # 61-75: gen2 + guided / median / gauss (alt flat denoisers)
    for post, strength, kw in [
        ("guided", 0.35, {"radius": 3, "eps": 1e-3}),
        ("guided", 0.5, {"radius": 4, "eps": 1e-3}),
        ("guided", 0.65, {"radius": 4, "eps": 5e-4}),
        ("guided", 0.5, {"radius": 6, "eps": 1e-3}),
        ("guided", 0.4, {"radius": 4, "eps": 2e-3}),
        ("median", 0.35, {"ksize": 3}),
        ("median", 0.5, {"ksize": 3}),
        ("median", 0.45, {"ksize": 5}),
        ("gauss", 0.35, {"sigma": 0.8}),
        ("gauss", 0.5, {"sigma": 1.2}),
        ("gauss", 0.4, {"sigma": 1.5}),
        ("guided", 0.55, {"radius": 5, "eps": 8e-4}),
        ("guided", 0.3, {"radius": 3, "eps": 1e-4}),
        ("median", 0.3, {"ksize": 3}),
        ("gauss", 0.55, {"sigma": 1.0}),
    ]:
        tag = "_".join(f"{k}{v}" for k, v in kw.items())
        recipes.append(
            Recipe(
                name=f"g2_{post}_{strength:g}_{tag}",
                family="gen2_post",
                post=post,
                post_strength=strength,
                post_kw=kw,
            )
        )

    # 76-85: edge unsharp / TTA / mild dt on gen2
    for amt, sig in [
        (0.2, 0.8),
        (0.35, 1.0),
        (0.5, 1.0),
        (0.35, 1.5),
        (0.25, 1.2),
        (0.45, 0.9),
        (0.15, 1.0),
        (0.55, 1.2),
    ]:
        recipes.append(
            Recipe(
                name=f"g2_unsharp_a{amt:g}_s{sig:g}",
                family="gen2_post",
                post="unsharp",
                post_strength=amt,
                post_kw={"sigma": sig, "harden": 16.0},
            )
        )
    recipes.append(
        Recipe(name="g2_tta4", family="gen2_post", post="tta", post_kw={"n_aug": 4})
    )
    recipes.append(
        Recipe(
            name="g2_dt0.4",
            family="gen2_post",
            post="dt",
            post_strength=0.4,
            post_kw={"sigma": 1.2, "harden": 16.0},
        )
    )

    # 86-99: hybrids — residual/freq + gen2 bilat schedule, mild combos
    for gt, mix, mid in [
        (12.0, 0.7, 0.7),
        (12.0, 0.55, 0.7),
        (10.0, 0.65, 0.6),
        (14.0, 0.6, 0.7),
        (12.0, 0.8, 0.55),
    ]:
        recipes.append(
            Recipe(
                name=f"hy_res_m{mix:g}_mid{mid:g}",
                family="residual",
                gate_temp=gt,
                gate_mix=mix,
                mid_bilat=mid,
                apply_gen2_bilat=True,
            )
        )
    for sig, he, mid in [
        (1.5, 0.85, 0.7),
        (1.2, 0.9, 0.65),
        (2.0, 0.8, 0.7),
        (1.5, 0.75, 0.55),
    ]:
        recipes.append(
            Recipe(
                name=f"hy_freq_s{sig:g}_he{he:g}_m{mid:g}",
                family="freq",
                freq_sigma=sig,
                high_edge=he,
                mid_bilat=mid,
                apply_gen2_bilat=True,
            )
        )
    recipes.append(
        Recipe(
            name="hy_var_guided",
            family="variance",
            var_win=7,
            edge_bias=0.5,
            post="guided",
            post_strength=0.35,
            post_kw={"radius": 4, "eps": 1e-3},
        )
    )
    recipes.append(
        Recipe(
            name="hy_res_unsharp",
            family="residual",
            gate_temp=12.0,
            gate_mix=0.7,
            post="unsharp",
            post_strength=0.25,
            post_kw={"sigma": 1.0},
        )
    )
    recipes.append(
        Recipe(
            name="hy_morph_guided",
            family="morph",
            dilate=1,
            post="guided",
            post_strength=0.4,
            post_kw={"radius": 4, "eps": 1e-3},
        )
    )
    recipes.append(
        Recipe(
            name="hy_freq_tta",
            family="freq",
            freq_sigma=1.5,
            high_edge=0.9,
            post="tta",
            post_kw={"n_aug": 4},
        )
    )

    recipes = recipes[:100]
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_res_{i}",
                family="residual",
                gate_temp=8.0 + 0.2 * i,
                gate_mix=0.5 + 0.01 * (i % 40),
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


def apply_recipe(recipe: Recipe, sota: np.ndarray, edge: np.ndarray, fps: float) -> np.ndarray:
    if recipe.family == "gen2":
        return gen2_base_then(sota, edge, fps, post="none")

    if recipe.family == "gen2_post":
        if recipe.post == "dt":
            out = gen2_base_then(sota, edge, fps, post="none")
            return detail_transfer(
                out,
                edge,
                amount=float(recipe.post_strength),
                sigma=float(recipe.post_kw.get("sigma", 1.2)),
                harden=float(recipe.post_kw.get("harden", 16.0)),
                edge_only=True,
            )
        return gen2_base_then(
            sota,
            edge,
            fps,
            post=recipe.post,
            post_strength=recipe.post_strength,
            **recipe.post_kw,
        )

    low = float(fps) <= float(recipe.low_fps)
    if low:
        out = sota.copy()
    elif recipe.family == "residual":
        out = residual_gate_blend(
            sota,
            edge,
            temperature=recipe.gate_temp,
            harden=recipe.gate_harden,
            percentile=recipe.gate_pct,
            mix=recipe.gate_mix,
        )
    elif recipe.family == "variance":
        out = local_variance_gate_blend(
            sota,
            edge,
            win=recipe.var_win,
            temperature=recipe.gate_temp,
            harden=recipe.gate_harden,
            percentile=recipe.gate_pct,
            edge_bias=recipe.edge_bias,
        )
    elif recipe.family == "freq":
        out = freq_split_fuse(
            sota,
            edge,
            sigma=recipe.freq_sigma,
            high_edge=recipe.high_edge,
            low_edge=recipe.low_edge,
        )
    elif recipe.family == "morph":
        out = morph_edge_blend(
            sota,
            edge,
            temperature=recipe.temperature,
            harden=recipe.harden,
            dilate=recipe.dilate,
            erode=recipe.erode,
        )
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, temperature=recipe.temperature, harden=recipe.harden
        )

    if recipe.dt_amount > 0.0 and not low:
        out = detail_transfer(out, edge, amount=recipe.dt_amount, edge_only=True)

    if recipe.apply_gen2_bilat:
        if low and recipe.low_bilat > 0:
            out = flat_bilateral_boost(
                out, guide=out, flat_strength=recipe.low_bilat, harden=recipe.bilat_harden
            )
        elif (not low) and recipe.mid_fps > 0 and float(fps) <= recipe.mid_fps:
            if recipe.mid_bilat > 0:
                out = flat_bilateral_boost(
                    out,
                    guide=out,
                    flat_strength=recipe.mid_bilat,
                    harden=recipe.bilat_harden,
                )

    # optional post
    if recipe.post == "guided" and recipe.post_strength > 0:
        out = flat_guided_boost(
            out, flat_strength=recipe.post_strength, **recipe.post_kw
        )
    elif recipe.post == "median" and recipe.post_strength > 0:
        out = flat_median_boost(
            out, flat_strength=recipe.post_strength, **recipe.post_kw
        )
    elif recipe.post == "gauss" and recipe.post_strength > 0:
        out = flat_gaussian_boost(
            out, flat_strength=recipe.post_strength, **recipe.post_kw
        )
    elif recipe.post == "unsharp":
        out = edge_unsharp(
            out, amount=recipe.post_strength or 0.35, **recipe.post_kw
        )
    elif recipe.post == "tta":
        out = geometric_self_ensemble(out, n_aug=int(recipe.post_kw.get("n_aug", 4)))

    return out.astype(np.float32)


def maybe_bake(best_safe: dict, baseline: float, output_dir: Path) -> None:
    """Rewrite infer_blend_deploy only for gen2_post / schedule-compatible wins."""
    if best_safe is None:
        return
    if float(best_safe["mean_des"]) < baseline + 1e-4:
        print("No bake: best_safe not above gen2 baseline", flush=True)
        return
    bake_path = output_dir / "bake_recipe.json"
    bake_path.write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
    print(f"Bake candidate written: {bake_path}", flush=True)

    # Auto-patch deploy only when recipe is gen2_post with guided/median that
    # can be expressed as bilateral alternatives — keep JSON for manual; if
    # family is residual/freq write note only (needs code path).
    recipe = best_safe.get("recipe") or {}
    family = recipe.get("family", "")
    note = output_dir / "BAKE_NOTE.txt"
    note.write_text(
        f"best_safe={best_safe['name']} mean={best_safe['mean_des']:.4f}\n"
        f"family={family}\n"
        f"Deploy patch: see bake_recipe.json; apply in infer_blend_deploy if "
        f"schedule-compatible.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_loop100_p7")
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
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--bake-deploy", action="store_true")
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
    recipes = build_100_recipes()[: max(1, int(args.iters))]
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    history: list[dict] = []
    best_safe: dict | None = None
    best_any: dict | None = None
    baseline_des = 0.0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        des_list: list[float] = []
        f10 = f05 = f1 = None
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
        if recipe.name == "gen2_deploy":
            baseline_des = mean_des
        if best_any is None or mean_des > best_any["mean_des"]:
            best_any = rec
        if ok and (best_safe is None or mean_des > best_safe["mean_des"]):
            best_safe = rec
            print(
                f"[{idx:03d}/{len(recipes)}] NEW SAFE BEST {recipe.name} "
                f"DES={mean_des:.4f} f10_ef={f10_ef:.4f} f05_ng={f05_ng:.4f}",
                flush=True,
            )
        else:
            tag = "ok" if ok else "unsafe"
            print(
                f"[{idx:03d}/{len(recipes)}] {recipe.name}: DES={mean_des:.4f} ({tag})",
                flush=True,
            )
        if (idx + 1) % 10 == 0 or idx + 1 == len(recipes):
            (args.output_dir / "history.json").write_text(
                json.dumps(
                    {"history": history, "best_safe": best_safe, "best_any": best_any},
                    indent=2,
                ),
                encoding="utf-8",
            )

    csv_path = args.output_dir / "metrics_by_recipe.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
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
        for rec in history:
            writer.writerow({k: rec[k] for k in writer.fieldnames})

    summary = {
        "n_iters": len(history),
        "baseline_gen2": baseline_des,
        "best_safe": best_safe,
        "best_any": best_any,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("--- done ---", flush=True)
    if best_safe:
        print(
            f"BEST_SAFE {best_safe['name']} mean={best_safe['mean_des']:.4f}",
            flush=True,
        )
    if best_any:
        print(
            f"BEST_ANY {best_any['name']} mean={best_any['mean_des']:.4f} "
            f"safe={best_any['edge_safe']}",
            flush=True,
        )
    print(f"Wrote {csv_path}", flush=True)
    if args.bake_deploy:
        maybe_bake(best_safe, baseline_des or 0.9175, args.output_dir)


if __name__ == "__main__":
    main()
