"""100-iter zero-train DES ladder on cached SOTA + lap_ms DualHead outputs.

Literature leftovers after P5 (mean DES ~0.9100 deploy):
  - Burt/HDR+ highpass detail transfer (edge-masked)
  - Yue/Hasinoff flat bilateral on mid-fps band
  - Multi-cue (Sobel+LoG) blend gates (Young MoE-lite)
  - Combinations of the above

Loop: for each of 100 recipes, score holdout DES under edge-safe
constraints; keep best; optionally bake into infer_blend_deploy.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .compare_sota_edgekd_blend import _spatial_on_merged
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .fe_schedule import gated_spatial_from_temporal
from .infer import load_model
from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import flat_bilateral_boost, multiband_fuse
from .p6_fusion import blend_with_edge_map, detail_transfer, multi_cue_edge_map
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi
from .wiener_merge import merge_from_memmap

F10 = "74824541"
F05 = "53940814"
F1 = "53741354"


@dataclass
class Recipe:
    name: str
    # blend path (fps > low_fps)
    blend_mode: str = "pixel"  # pixel | multicue | mb
    temperature: float = 8.0
    harden: float = 16.0
    sobel_weight: float = 0.65
    log_sigma: float = 1.0
    mb_weights: tuple[float, ...] | None = None
    # detail transfer onto blend (or sota)
    dt_amount: float = 0.0
    dt_sigma: float = 1.2
    dt_harden: float = 16.0
    dt_edge_only: bool = True
    # flat bilateral
    low_fps: float = 1.0
    mid_fps: float = 0.0  # 0 = disabled mid band
    low_bilat: float = 0.65
    mid_bilat: float = 0.0
    bilat_harden: float = 40.0
    # low-fps base: sota (deploy) or blend
    low_use_blend: bool = False


def build_100_recipes() -> list[Recipe]:
    recipes: list[Recipe] = []
    # 0: current P5 deploy baseline
    recipes.append(Recipe(name="p5_deploy"))

    # 1-18: detail transfer on pixel blend
    for i, (amt, sig, h) in enumerate(
        [
            (0.35, 0.8, 12.0),
            (0.5, 0.8, 16.0),
            (0.65, 1.0, 16.0),
            (0.8, 1.0, 16.0),
            (1.0, 1.2, 16.0),
            (1.15, 1.2, 16.0),
            (1.3, 1.5, 20.0),
            (0.8, 1.5, 12.0),
            (1.0, 0.8, 24.0),
            (1.0, 2.0, 16.0),
            (0.6, 1.2, 8.0),
            (0.9, 1.0, 32.0),
            (1.2, 1.0, 16.0),
            (0.75, 1.2, 16.0),
            (1.0, 1.2, 12.0),
            (0.5, 1.5, 24.0),
            (0.85, 0.9, 18.0),
            (1.1, 1.3, 14.0),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"dt_a{amt:g}_s{sig:g}_h{h:g}",
                dt_amount=amt,
                dt_sigma=sig,
                dt_harden=h,
            )
        )

    # 19-30: mid-fps bilateral
    for mid_fps, mid_s, low_s in [
        (2.5, 0.25, 0.65),
        (2.5, 0.35, 0.65),
        (2.5, 0.45, 0.65),
        (5.0, 0.20, 0.65),
        (5.0, 0.30, 0.65),
        (5.0, 0.40, 0.65),
        (5.0, 0.35, 0.55),
        (5.0, 0.35, 0.75),
        (3.0, 0.30, 0.65),
        (8.0, 0.25, 0.65),
        (5.0, 0.50, 0.65),
        (1.5, 0.40, 0.65),
    ]:
        recipes.append(
            Recipe(
                name=f"mid_f{mid_fps:g}_ms{mid_s:g}_ls{low_s:g}",
                mid_fps=mid_fps,
                mid_bilat=mid_s,
                low_bilat=low_s,
            )
        )

    # 31-45: multi-cue blend
    for sw, t, h, ls in [
        (0.5, 8.0, 16.0, 1.0),
        (0.65, 8.0, 16.0, 1.0),
        (0.8, 8.0, 16.0, 1.0),
        (0.5, 12.0, 16.0, 1.0),
        (0.65, 12.0, 20.0, 1.2),
        (0.35, 8.0, 16.0, 0.8),
        (0.65, 8.0, 24.0, 1.0),
        (0.65, 16.0, 16.0, 1.0),
        (0.7, 10.0, 18.0, 1.0),
        (0.55, 8.0, 12.0, 1.5),
        (0.65, 8.0, 16.0, 0.7),
        (0.9, 8.0, 16.0, 1.0),
        (0.4, 12.0, 16.0, 1.0),
        (0.65, 6.0, 16.0, 1.0),
        (0.75, 8.0, 32.0, 1.0),
    ]:
        recipes.append(
            Recipe(
                name=f"mc_sw{sw:g}_T{t:g}_h{h:g}_ls{ls:g}",
                blend_mode="multicue",
                sobel_weight=sw,
                temperature=t,
                harden=h,
                log_sigma=ls,
            )
        )

    # 46-55: light multi-band leftovers
    for wts in [
        (1.0, 0.6, 0.0),
        (1.0, 0.8, 0.1, 0.0),
        (0.95, 0.55, 0.0),
        (1.0, 1.0, 0.25, 0.0),
        (1.0, 0.7, 0.15, 0.0),
        (0.85, 0.4, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 0.9, 0.35, 0.05),
        (1.0, 0.75, 0.2, 0.0),
        (0.9, 0.7, 0.2, 0.0),
    ]:
        recipes.append(
            Recipe(
                name=f"mb_{'_'.join(f'{w:g}' for w in wts)}",
                blend_mode="mb",
                mb_weights=tuple(wts),
            )
        )

    # 56-75: detail transfer + mid bilateral combos
    for amt, mid_s, mid_fps in [
        (0.5, 0.30, 5.0),
        (0.65, 0.30, 5.0),
        (0.8, 0.30, 5.0),
        (1.0, 0.25, 5.0),
        (0.8, 0.35, 5.0),
        (0.65, 0.25, 2.5),
        (1.0, 0.35, 5.0),
        (0.75, 0.40, 5.0),
        (0.9, 0.20, 8.0),
        (1.15, 0.30, 5.0),
        (0.6, 0.35, 3.0),
        (0.85, 0.28, 5.0),
        (1.0, 0.30, 2.5),
        (0.7, 0.45, 5.0),
        (0.95, 0.32, 5.0),
        (0.55, 0.30, 5.0),
        (1.05, 0.22, 5.0),
        (0.8, 0.30, 8.0),
        (0.65, 0.35, 8.0),
        (1.0, 0.40, 5.0),
    ]:
        recipes.append(
            Recipe(
                name=f"dt{amt:g}_mid{mid_s:g}_f{mid_fps:g}",
                dt_amount=amt,
                dt_sigma=1.2,
                dt_harden=16.0,
                mid_fps=mid_fps,
                mid_bilat=mid_s,
            )
        )

    # 76-90: multicue + detail transfer
    for sw, amt, mid_s in [
        (0.65, 0.7, 0.0),
        (0.65, 0.9, 0.0),
        (0.5, 0.8, 0.0),
        (0.8, 0.8, 0.0),
        (0.65, 1.0, 0.3),
        (0.65, 0.75, 0.35),
        (0.55, 0.85, 0.25),
        (0.7, 0.9, 0.3),
        (0.65, 1.1, 0.25),
        (0.6, 0.8, 0.4),
        (0.75, 0.7, 0.3),
        (0.5, 1.0, 0.3),
        (0.65, 0.6, 0.35),
        (0.8, 1.0, 0.2),
        (0.65, 0.95, 0.28),
    ]:
        recipes.append(
            Recipe(
                name=f"mc{sw:g}_dt{amt:g}_m{mid_s:g}",
                blend_mode="multicue",
                sobel_weight=sw,
                dt_amount=amt,
                mid_fps=5.0 if mid_s > 0 else 0.0,
                mid_bilat=mid_s,
            )
        )

    # 91-99: low-fps blend+bilateral / stronger low bilateral / global mild
    for low_s, mid_s, mid_fps, use_blend in [
        (0.55, 0.0, 0.0, False),
        (0.75, 0.0, 0.0, False),
        (0.85, 0.0, 0.0, False),
        (0.65, 0.35, 5.0, False),
        (0.70, 0.30, 5.0, False),
        (0.65, 0.0, 0.0, True),
        (0.65, 0.30, 5.0, True),
        (0.60, 0.40, 5.0, False),
        (0.65, 0.25, 10.0, False),
    ]:
        recipes.append(
            Recipe(
                name=f"low{low_s:g}_mid{mid_s:g}_f{mid_fps:g}_b{int(use_blend)}",
                low_bilat=low_s,
                mid_bilat=mid_s,
                mid_fps=mid_fps,
                low_use_blend=use_blend,
            )
        )

    # Ensure exactly 100 (pad/truncate)
    if len(recipes) < 100:
        for i in range(len(recipes), 100):
            recipes.append(
                Recipe(
                    name=f"pad_dt_{i}",
                    dt_amount=0.4 + 0.01 * (i - 50),
                    dt_sigma=1.0 + 0.02 * (i % 5),
                )
            )
    recipes = recipes[:100]
    # unique names
    seen: set[str] = set()
    uniq: list[Recipe] = []
    for r in recipes:
        name = r.name
        if name in seen:
            name = f"{name}_dup{len(uniq)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        uniq.append(r)
    return uniq


def apply_recipe(
    recipe: Recipe,
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
) -> np.ndarray:
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    low = float(fps) <= float(recipe.low_fps)

    if low and not recipe.low_use_blend:
        base = sota.copy()
    else:
        if recipe.blend_mode == "multicue":
            emap = multi_cue_edge_map(
                guide,
                temperature=recipe.temperature,
                harden=recipe.harden,
                log_sigma=recipe.log_sigma,
                sobel_weight=recipe.sobel_weight,
            )
            base = blend_with_edge_map(sota, edge, emap)
        elif recipe.blend_mode == "mb" and recipe.mb_weights is not None:
            base = multiband_fuse(
                sota,
                edge,
                band_edge_weights=list(recipe.mb_weights),
                guide=guide,
                spatial_mix=0.0,
            )
        else:
            base, _ = blend_sota_edgekd(
                sota,
                edge,
                guide_dn=guide,
                temperature=recipe.temperature,
                harden=recipe.harden,
                edge_weight=1.0,
            )

    if recipe.dt_amount > 0.0 and not (low and not recipe.low_use_blend):
        # detail transfer mainly for blend path (high/mid fps)
        base = detail_transfer(
            base,
            edge,
            guide=guide,
            amount=recipe.dt_amount,
            sigma=recipe.dt_sigma,
            harden=recipe.dt_harden,
            edge_only=recipe.dt_edge_only,
        )
    elif recipe.dt_amount > 0.0 and low and recipe.low_use_blend:
        base = detail_transfer(
            base,
            edge,
            guide=guide,
            amount=recipe.dt_amount,
            sigma=recipe.dt_sigma,
            harden=recipe.dt_harden,
            edge_only=recipe.dt_edge_only,
        )

    # bilateral schedule
    if low:
        if recipe.low_bilat > 0.0:
            base = flat_bilateral_boost(
                base,
                guide=base,
                flat_strength=recipe.low_bilat,
                harden=recipe.bilat_harden,
            )
    elif recipe.mid_fps > 0.0 and float(fps) <= float(recipe.mid_fps):
        if recipe.mid_bilat > 0.0:
            base = flat_bilateral_boost(
                base,
                guide=base,
                flat_strength=recipe.mid_bilat,
                harden=recipe.bilat_harden,
            )
    return base.astype(np.float32)


def edge_safe(f10_ef: float, f05_ef: float, f1_ef: float) -> bool:
    return f10_ef >= 0.965 and f05_ef >= 0.870 and f1_ef >= 0.960


def cache_forwards(
    files: list[Path],
    cache_dir: Path,
    checkpoint_sota: Path,
    checkpoint_edge: Path,
    tile_size: int,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sota, n_in, _ = load_model(checkpoint_sota, None, device)
    model_edge, n_e, _ = load_model(checkpoint_edge, None, device)
    assert n_in == n_e
    metas: list[dict] = []
    for path in files:
        stem = path.stem
        meta_path = cache_dir / f"{stem}.json"
        sota_path = cache_dir / f"{stem}_sota.npy"
        edge_path = cache_dir / f"{stem}_edge.npy"
        if meta_path.exists() and sota_path.exists() and edge_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metas.append(meta)
            print(f"cache hit {stem}", flush=True)
            continue
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        target_index = frames.shape[0] // 2
        ys, xs = central_roi(height, width)
        input_dn = decode_mono10(frames[target_index])
        temporal_ref = aligned_temporal_trimmed_mean(
            frames,
            target_index,
            ys,
            xs,
            frames.shape[0],
            min(16, frames.shape[0]),
            trim_fraction=0.10,
            exclude_frame_index=target_index,
        )
        mask = reliable_reference_mask(input_dn, temporal_ref)
        print(f"cache miss {stem}: FE+forwards...", flush=True)
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
        sota_dn = _spatial_on_merged(
            model_sota, fe_gated, exposure_ms, device, n_in, tile_size
        )
        edge_dn = _spatial_on_merged(
            model_edge, fe_gated, exposure_ms, device, n_in, tile_size
        )
        np.save(sota_path, sota_dn.astype(np.float32))
        np.save(edge_path, edge_dn.astype(np.float32))
        np.save(cache_dir / f"{stem}_input.npy", input_dn.astype(np.float32))
        np.save(cache_dir / f"{stem}_ref.npy", temporal_ref.astype(np.float32))
        np.save(cache_dir / f"{stem}_mask.npy", mask.astype(np.uint8))
        meta = {
            "file": path.name,
            "stem": stem,
            "path": str(path),
            "fps": float(fps),
            "exposure_ms": float(exposure_ms),
            "target_index": int(target_index),
            "ys0": int(ys.start),
            "ys1": int(ys.stop),
            "xs0": int(xs.start),
            "xs1": int(xs.stop),
            "fe_schedule": params.name,
            "fe_sigma_dn": float(fe_sigma),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        metas.append(meta)
    return metas


def score_image(
    meta: dict,
    cache_dir: Path,
    method: str,
    image: np.ndarray,
) -> dict[str, float]:
    stem = meta["stem"]
    input_dn = np.load(cache_dir / f"{stem}_input.npy")
    temporal_ref = np.load(cache_dir / f"{stem}_ref.npy")
    mask = np.load(cache_dir / f"{stem}_mask.npy").astype(bool)
    ys = slice(meta["ys0"], meta["ys1"])
    xs = slice(meta["xs0"], meta["xs1"])
    path = Path(meta["path"])
    fps = meta["fps"]
    target_index = meta["target_index"]
    row = metric_row(
        path,
        fps,
        target_index,
        method,
        image,
        temporal_ref,
        mask,
        ys,
        xs,
        checkpoint=method,
        input_frames=16,
    )
    input_sigma = float(
        metric_row(
            path,
            fps,
            target_index,
            "input",
            input_dn,
            temporal_ref,
            mask,
            ys,
            xs,
        )["roi_highpass_noise_sigma_dn"]
    )
    retention, edge_mae = edge_metrics_dn(image, temporal_ref)
    des, ng, ef = denoise_edge_score(
        float(row["roi_highpass_noise_sigma_dn"]),
        input_sigma,
        retention,
    )
    return {
        "des": float(des),
        "noise_gain": float(ng),
        "edge_fidelity": float(ef),
        "edge_retention": float(retention),
        "edge_sobel_mae": float(edge_mae),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("nafnet_denoise/cache_dual_holdout"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_loop100"),
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
    parser.add_argument(
        "--bake-deploy",
        action="store_true",
        help="After loop, rewrite infer_blend_deploy defaults if best beats p5.",
    )
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

    # Preload arrays
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
                f"[{idx:03d}/{len(recipes)}] NEW SAFE BEST {recipe.name} "
                f"DES={mean_des:.4f} f10_ef={f10_ef:.4f} f05_ng={f05_ng:.4f}",
                flush=True,
            )
        else:
            tag = "ok" if ok else "unsafe"
            print(
                f"[{idx:03d}/{len(recipes)}] {recipe.name}: "
                f"DES={mean_des:.4f} ({tag})",
                flush=True,
            )

        # checkpoint every 10
        if (idx + 1) % 10 == 0 or idx + 1 == len(recipes):
            (args.output_dir / "history.json").write_text(
                json.dumps(
                    {
                        "history": history,
                        "best_safe": best_safe,
                        "best_any": best_any,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # CSV summary
    csv_path = args.output_dir / "metrics_by_recipe.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iter",
                "name",
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
        "best_safe": best_safe,
        "best_any": best_any,
        "baseline_p5": next((h for h in history if h["name"] == "p5_deploy"), None),
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

    if args.bake_deploy and best_safe is not None:
        base = next((h for h in history if h["name"] == "p5_deploy"), None)
        if base is None or best_safe["mean_des"] > base["mean_des"] + 1e-5:
            bake_path = args.output_dir / "bake_recipe.json"
            bake_path.write_text(
                json.dumps(best_safe, indent=2), encoding="utf-8"
            )
            print(f"Bake candidate written: {bake_path}", flush=True)
        else:
            print("No bake: best_safe not above p5_deploy", flush=True)


if __name__ == "__main__":
    main()
