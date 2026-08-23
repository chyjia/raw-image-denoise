"""P9 training meta: lit levers after postprocess saturation (DES≈0.9221).

Paper map:
  Young / FitNet          → edge-masked highpass + output KD (blend or lap_ms)
  AIM25 MAE-style         → (approx) higher real-fraction + hard-scene focus
  EAGLE / EAFormer        → stronger grad / des_edge_fid weights
  Multi-teacher blend KD  → blend_kd with edge_gain (retry with short FT)
  Noise2Noise leave-burst → mild temporal teacher (prior long FT failed)

Each recipe: short DualHead FT → score holdout with P7 deploy fusion
(Gen2 bilat + unsharp). Edge-safe gate. Bake SOTA/edge ckpt if mean DES
beats 0.9221 + 1e-4.

Up to 100 recipes; early-stop after ``patience`` stagnant.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .infer import load_model
from .infer_blend_deploy import denoise_blend_deploy
from .p8_fusion import deploy_p7_base
from .run_des_loop100 import (
    F1,
    F05,
    F10,
    cache_forwards,
    edge_safe,
    score_image,
)
from .common import memmap_frames, parse_exposure_ms, parse_geometry

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 0.9221


@dataclass
class TrainRecipe:
    name: str
    family: str
    # which arm to FT
    target: str = "sota"  # sota | edge | both→sota resume only
    epochs: int = 4
    patches: int = 128
    lr: float = 2e-6
    highpass_weight: float = 0.50
    edge_weight: float = 0.36
    grad_weight: float = 0.40
    des_edge_fid_weight: float = 1.20
    des_noise_gain_weight: float = 0.60
    flat_boost: float = 2.0
    teacher_mode: str = "partitioned"  # partitioned | temporal
    blend_kd: bool = False
    blend_kd_mix: float = 0.85
    blend_kd_edge_gain: float = 1.2
    freeze_flat: bool = False
    freeze_edge: bool = False
    hard_sample_mult: float = 3.0
    hard_loss_mult: float = 2.0
    mixed_grad_xy: float = 0.30
    teacher_w_start: float = 0.55
    teacher_w_end: float = 0.40


def build_100_recipes() -> list[TrainRecipe]:
    recipes: list[TrainRecipe] = []
    # 0: baseline score only (no train) — handled as skip
    recipes.append(TrainRecipe(name="baseline_p7", family="baseline", epochs=0))

    # 1-20: edge-arm HP + DES fid (Young-lite / absorb blend detail)
    for i, (hp, ef, ew, lr) in enumerate(
        [
            (0.55, 1.40, 0.40, 2e-6),
            (0.65, 1.60, 0.42, 2e-6),
            (0.70, 1.80, 0.45, 1.5e-6),
            (0.50, 2.00, 0.48, 2e-6),
            (0.60, 1.50, 0.38, 3e-6),
            (0.75, 1.70, 0.44, 1e-6),
            (0.55, 1.90, 0.50, 2e-6),
            (0.45, 1.60, 0.36, 2e-6),
            (0.65, 1.40, 0.42, 2.5e-6),
            (0.70, 2.00, 0.40, 1.5e-6),
            (0.60, 1.75, 0.46, 2e-6),
            (0.80, 1.55, 0.40, 1e-6),
            (0.55, 1.65, 0.42, 2e-6),
            (0.65, 1.85, 0.44, 2e-6),
            (0.50, 1.70, 0.50, 2e-6),
            (0.70, 1.50, 0.38, 2e-6),
            (0.60, 2.10, 0.45, 1.5e-6),
            (0.55, 1.45, 0.40, 3e-6),
            (0.75, 1.90, 0.42, 1e-6),
            (0.65, 1.70, 0.48, 2e-6),
        ]
    ):
        recipes.append(
            TrainRecipe(
                name=f"edge_hp_{i}_h{hp:g}_ef{ef:g}",
                family="edge_hp",
                target="edge",
                freeze_flat=True,
                highpass_weight=hp,
                des_edge_fid_weight=ef,
                edge_weight=ew,
                lr=lr,
                blend_kd=True,
                blend_kd_mix=0.9,
                blend_kd_edge_gain=1.3,
            )
        )

    # 21-35: SOTA flat arm noise_gain / highpass (EAGLE flats)
    for i, (hp, ng, fb, lr) in enumerate(
        [
            (0.55, 0.70, 2.2, 2e-6),
            (0.65, 0.75, 2.4, 1.5e-6),
            (0.70, 0.80, 2.5, 1e-6),
            (0.50, 0.65, 2.0, 2e-6),
            (0.60, 0.70, 2.3, 2e-6),
            (0.75, 0.72, 2.6, 1e-6),
            (0.55, 0.85, 2.2, 1.5e-6),
            (0.45, 0.70, 1.8, 2e-6),
            (0.65, 0.68, 2.4, 2e-6),
            (0.70, 0.78, 2.3, 1.5e-6),
            (0.60, 0.75, 2.5, 2e-6),
            (0.80, 0.70, 2.2, 1e-6),
            (0.55, 0.72, 2.4, 2e-6),
            (0.65, 0.80, 2.0, 1.5e-6),
            (0.50, 0.78, 2.5, 2e-6),
        ]
    ):
        recipes.append(
            TrainRecipe(
                name=f"flat_ng_{i}_h{hp:g}_n{ng:g}",
                family="flat_ng",
                target="sota",
                freeze_edge=True,
                highpass_weight=hp,
                des_noise_gain_weight=ng,
                flat_boost=fb,
                lr=lr,
                des_edge_fid_weight=1.0,
            )
        )

    # 36-50: blend-KD multi-teacher (retry short)
    for i, (mix, gain, ef) in enumerate(
        [
            (0.70, 1.0, 1.2),
            (0.85, 1.2, 1.4),
            (0.95, 1.4, 1.6),
            (1.00, 1.5, 1.5),
            (0.80, 1.8, 1.7),
            (0.90, 1.1, 1.3),
            (0.75, 1.6, 1.8),
            (0.85, 1.3, 1.5),
            (0.95, 1.0, 1.4),
            (0.80, 1.5, 1.2),
            (0.70, 1.4, 1.6),
            (0.90, 1.7, 1.5),
            (0.85, 1.0, 1.8),
            (1.00, 1.2, 1.4),
            (0.75, 1.3, 1.5),
        ]
    ):
        recipes.append(
            TrainRecipe(
                name=f"blendkd_{i}_m{mix:g}_g{gain:g}",
                family="blend_kd",
                target="sota",
                blend_kd=True,
                blend_kd_mix=mix,
                blend_kd_edge_gain=gain,
                des_edge_fid_weight=ef,
                highpass_weight=0.45,
                lr=2e-6,
            )
        )

    # 51-65: EAGLE-style grad
    for i, (gw, xy, ef) in enumerate(
        [
            (0.50, 0.40, 1.5),
            (0.60, 0.50, 1.6),
            (0.70, 0.45, 1.7),
            (0.55, 0.55, 1.4),
            (0.65, 0.35, 1.8),
            (0.45, 0.50, 1.5),
            (0.75, 0.40, 1.6),
            (0.50, 0.60, 1.7),
            (0.60, 0.45, 1.9),
            (0.55, 0.40, 1.5),
            (0.70, 0.55, 1.4),
            (0.65, 0.50, 1.6),
            (0.80, 0.35, 1.5),
            (0.50, 0.45, 1.8),
            (0.60, 0.60, 1.7),
        ]
    ):
        recipes.append(
            TrainRecipe(
                name=f"eagle_{i}_g{gw:g}_xy{xy:g}",
                family="eagle",
                target="edge",
                freeze_flat=True,
                grad_weight=gw,
                mixed_grad_xy=xy,
                des_edge_fid_weight=ef,
                edge_weight=0.42,
                blend_kd=True,
                blend_kd_mix=0.85,
                blend_kd_edge_gain=1.25,
            )
        )

    # 66-80: mild N2N temporal on flat (short, avoid prior collapse)
    for i, (hp, ng, hs) in enumerate(
        [
            (0.40, 0.55, 2.0),
            (0.45, 0.60, 2.5),
            (0.50, 0.65, 3.0),
            (0.35, 0.50, 2.0),
            (0.45, 0.55, 3.5),
            (0.40, 0.70, 2.5),
            (0.50, 0.50, 2.0),
            (0.45, 0.60, 4.0),
            (0.55, 0.55, 2.5),
            (0.40, 0.65, 3.0),
            (0.35, 0.60, 2.5),
            (0.50, 0.70, 2.0),
            (0.45, 0.55, 3.0),
            (0.40, 0.50, 3.5),
            (0.55, 0.60, 2.5),
        ]
    ):
        recipes.append(
            TrainRecipe(
                name=f"n2n_{i}_h{hp:g}_n{ng:g}",
                family="n2n",
                target="sota",
                freeze_edge=True,
                teacher_mode="temporal",
                highpass_weight=hp,
                des_noise_gain_weight=ng,
                hard_sample_mult=hs,
                des_edge_fid_weight=0.95,
                lr=1e-6,
                epochs=3,
            )
        )

    # 81-99: hybrids / both-arm light
    for i, (hp, ef, mix) in enumerate(
        [
            (0.55, 1.5, 0.85),
            (0.60, 1.6, 0.90),
            (0.50, 1.7, 0.80),
            (0.65, 1.4, 0.95),
            (0.55, 1.8, 0.85),
            (0.70, 1.5, 0.90),
            (0.45, 1.6, 0.75),
            (0.60, 1.9, 0.85),
            (0.55, 1.55, 1.00),
            (0.65, 1.65, 0.80),
            (0.50, 1.75, 0.90),
            (0.60, 1.45, 0.85),
            (0.70, 1.70, 0.95),
            (0.55, 1.60, 0.70),
            (0.60, 1.80, 0.85),
            (0.50, 1.50, 0.90),
            (0.65, 1.70, 0.85),
            (0.55, 1.90, 0.95),
            (0.60, 1.55, 0.80),
        ]
    ):
        recipes.append(
            TrainRecipe(
                name=f"hy_{i}_h{hp:g}_ef{ef:g}",
                family="hybrid",
                target="sota",
                blend_kd=True,
                blend_kd_mix=mix,
                blend_kd_edge_gain=1.3,
                highpass_weight=hp,
                des_edge_fid_weight=ef,
                grad_weight=0.45,
                lr=1.5e-6,
            )
        )

    recipes = recipes[:100]
    while len(recipes) < 100:
        i = len(recipes)
        recipes.append(
            TrainRecipe(
                name=f"pad_edge_{i}",
                family="edge_hp",
                target="edge",
                freeze_flat=True,
                highpass_weight=0.5 + 0.01 * (i % 20),
                des_edge_fid_weight=1.4 + 0.02 * (i % 15),
                blend_kd=True,
            )
        )
    return recipes[:100]


def _base_train_cmd(out_dir: Path, resume: Path, recipe: TrainRecipe) -> list[str]:
    hard_manifest = Path(
        "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
    )
    sigma_map = Path("nafnet_denoise/cache/hard_wiener_bm3d_sigma_map.json")
    material = Path(r"D:\denoise\val_raw")
    sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    python = sys.executable
    cmd = [
        python, "-u", "-m", "nafnet_denoise.train_distill",
        "--arch", "dual_head", "--use-nonlocal", "--nonlocal-count", "1",
        "--align-soft-gate", "--align-threshold", "0.03",
        "--align-temperature", "0.015", "--align-drop-prob", "0.05",
        "--wiener-front-end", "--wiener-merge-frames", "16",
        "--wiener-tile", "32", "--wiener-overlap", "16", "--wiener-c-factor", "8.0",
        "--wiener-spatial", "--wiener-spatial-c-factor", "1.0",
        "--wiener-spatial-adaptive",
        "--wiener-spatial-flat-c-mult", "1.75",
        "--wiener-spatial-edge-c-mult", "0.45",
        "--wiener-spatial-dark-boost", "0.35",
        "--wiener-spatial-mask-harden", "8.0",
        "--wiener-spatial-freq-gamma", "0.15",
        "--wiener-fe-schedule", "fps_sigma",
        "--teacher-mode", recipe.teacher_mode,
        "--dark-boost", "0.5", "--scene-loss-power", "0.5",
        "--scene-loss-ref-fps", "10.0", "--scene-loss-max", "3.0",
        "--hard-scene-substr", "20260719153940814,20260725174824541",
        "--hard-scene-sample-mult", str(recipe.hard_sample_mult),
        "--hard-scene-loss-mult", str(recipe.hard_loss_mult),
        "--des-edge-fid-weight", str(recipe.des_edge_fid_weight),
        "--des-noise-gain-weight", str(recipe.des_noise_gain_weight),
        "--loss-domain", "dn",
        "--real-manifest", str(hard_manifest),
        "--out-dir", str(out_dir),
        "--resume", str(resume),
        "--fresh-resume",
        "--use-sigma", "--input-frames", "4", "--width", "32",
        "--patch-size", "256", "--micro-batch", "2", "--accum-steps", "4",
        "--epochs", str(recipe.epochs),
        "--patches-per-epoch", str(recipe.patches),
        "--real-fraction", "1.0", "--lr", str(recipe.lr),
        "--teacher-weight-start", str(recipe.teacher_w_start),
        "--teacher-weight-end", str(recipe.teacher_w_end),
        "--highpass-weight", str(recipe.highpass_weight),
        "--flat-boost", str(recipe.flat_boost),
        "--edge-weight", str(recipe.edge_weight),
        "--grad-weight", str(recipe.grad_weight),
        "--mixed-grad-xy-weight", str(recipe.mixed_grad_xy),
        "--validation-every", "99",
        "--validation-dir", str(material),
        "--num-workers", "0",
    ]
    if recipe.teacher_mode == "partitioned":
        cmd.extend([
            "--wiener-bm3d-teacher", "--wiener-bm3d-sigma", "0.35",
            "--wiener-bm3d-prob", "1.0",
            "--distill-edge-harden", "16.0",
        ])
        if sigma_map.exists():
            cmd.extend(["--hard-wiener-bm3d-sigma-map", str(sigma_map)])
    if recipe.freeze_flat:
        cmd.append("--freeze-flat-head")
    if recipe.freeze_edge:
        cmd.append("--freeze-edge-head")
    if recipe.blend_kd:
        cmd.extend([
            "--blend-kd-sota", str(sota),
            "--blend-kd-edgekd", str(edge),
            "--blend-kd-mix", str(recipe.blend_kd_mix),
            "--blend-kd-temperature", "8.0",
            "--blend-kd-harden", "16.0",
            "--blend-kd-edge-gain", str(recipe.blend_kd_edge_gain),
        ])
    if recipe.target == "edge" or recipe.freeze_flat:
        # lap edge helpers if resuming edge arm
        if "lap_edge" in str(resume):
            cmd.extend(["--use-lap-edge", "--use-lap-edge-ms"])
    return cmd


def train_recipe(recipe: TrainRecipe, work_root: Path) -> Path | None:
    if recipe.epochs <= 0:
        return None
    out_dir = work_root / f"ckpts_{recipe.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    edge = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")
    resume = edge if recipe.target == "edge" else sota
    if not resume.exists():
        print(f"Missing resume {resume}", flush=True)
        return None
    cmd = _base_train_cmd(out_dir, resume, recipe)
    log = out_dir / "train.log"
    print(f"TRAIN {recipe.name}: {' '.join(cmd[:8])}...", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    ckpt = out_dir / "best.pt"
    if code != 0 or not ckpt.exists():
        # fallback last.pt
        last = out_dir / "last.pt"
        if last.exists():
            print(f"TRAIN {recipe.name} exit={code}, using last.pt", flush=True)
            return last
        print(f"TRAIN FAILED {recipe.name} exit={code}", flush=True)
        return None
    return ckpt


@torch.no_grad()
def score_deploy_with_ckpts(
    files: list[Path],
    cache_dir: Path,
    ckpt_sota: Path,
    ckpt_edge: Path,
    method: str,
) -> dict:
    """Score P7 deploy fusion; re-forward arms when ckpt differs from cache keys."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sota, n_in, _ = load_model(ckpt_sota, None, device)
    model_edge, n_e, _ = load_model(ckpt_edge, None, device)
    assert n_in == n_e

    # Use cached arrays when ckpts are the stock ones; else live infer via deploy
    stock_sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt").resolve()
    stock_edge = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt").resolve()
    use_cache_sota = ckpt_sota.resolve() == stock_sota
    use_cache_edge = ckpt_edge.resolve() == stock_edge

    des_list = []
    f10 = f05 = f1 = None
    for path in files:
        stem = path.stem
        meta = json.loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        target_index = int(meta["target_index"])
        if use_cache_sota and use_cache_edge:
            sota = np.load(cache_dir / f"{stem}_sota.npy")
            edge = np.load(cache_dir / f"{stem}_edge.npy")
            out = deploy_p7_base(sota, edge, float(fps))
        else:
            out, _ = denoise_blend_deploy(
                model_sota,
                model_edge,
                n_in,
                frames,
                target_index,
                float(fps),
                exposure_ms,
                device,
                tile=256,
            )
        sc = score_image(meta, cache_dir, method, out)
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
    return {
        "mean_des": mean_des,
        "f10_ef": f10_ef,
        "f05_ef": f05_ef,
        "f05_ng": f05_ng,
        "f1_ef": f1_ef,
        "edge_safe": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p9")
    )
    parser.add_argument("--max-recipes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4, help="Override recipe epochs")
    parser.add_argument("--patches", type=int, default=128)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 under {args.input_dir}")
    # ensure cache
    cache_forwards(
        files,
        args.cache_dir,
        Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
        Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt"),
        256,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipes = build_100_recipes()[: int(args.max_recipes)]

    stock_sota = Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt")
    stock_edge = Path("nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt")

    history: list[dict] = []
    best_safe: dict | None = None
    stagnant = 0

    for idx, recipe in enumerate(recipes):
        if idx < int(args.start):
            continue
        if recipe.epochs > 0:
            recipe.epochs = min(recipe.epochs, int(args.epochs))
            recipe.patches = int(args.patches)

        print(f"\n===== P9 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
        ckpt_sota, ckpt_edge = stock_sota, stock_edge
        trained = None
        if recipe.family != "baseline":
            trained = train_recipe(recipe, args.output_dir)
            if trained is None:
                rec = {
                    "iter": idx,
                    "name": recipe.name,
                    "family": recipe.family,
                    "mean_des": 0.0,
                    "edge_safe": False,
                    "failed": True,
                    "recipe": asdict(recipe),
                }
                history.append(rec)
                stagnant += 1
                print(f"SKIP failed train {recipe.name}", flush=True)
                if stagnant >= int(args.patience):
                    print("Early stop after failures/stagnation", flush=True)
                    break
                continue
            if recipe.target == "edge":
                ckpt_edge = trained
            else:
                ckpt_sota = trained

        sc = score_deploy_with_ckpts(
            files, args.cache_dir, ckpt_sota, ckpt_edge, recipe.name
        )
        rec = {
            "iter": idx,
            "name": recipe.name,
            "family": recipe.family,
            "ckpt_sota": str(ckpt_sota),
            "ckpt_edge": str(ckpt_edge),
            "recipe": asdict(recipe),
            **sc,
        }
        history.append(rec)
        improved = False
        if sc["edge_safe"] and (
            best_safe is None or sc["mean_des"] > best_safe["mean_des"] + 1e-4
        ):
            # also require beat baseline for "improved" tracking toward bake
            if best_safe is None or sc["mean_des"] > best_safe["mean_des"] + 1e-4:
                best_safe = rec
                improved = True
                stagnant = 0
                (args.output_dir / "global_best.json").write_text(
                    json.dumps(best_safe, indent=2), encoding="utf-8"
                )
                print(
                    f"NEW SAFE BEST {recipe.name} DES={sc['mean_des']:.4f}",
                    flush=True,
                )
        else:
            if best_safe is not None and sc["mean_des"] <= best_safe["mean_des"] + 1e-4:
                stagnant += 1
            elif best_safe is None and sc["mean_des"] < BASELINE + 1e-4:
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

        if recipe.family == "baseline":
            stagnant = 0  # don't count baseline
        if stagnant >= int(args.patience) and idx > 0:
            print(
                f"Early stop: {stagnant} without +1e-4. "
                f"best={None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P9 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        bake = args.output_dir / "bake_recipe.json"
        bake.write_text(json.dumps(best_safe, indent=2), encoding="utf-8")
        print(f"BAKE candidate {bake} DES={best_safe['mean_des']:.4f}", flush=True)
        # Point deploy defaults via hook file (ckpt paths)
        hook = Path("nafnet_denoise/deploy_p9_hook.json")
        hook.write_text(
            json.dumps(
                {
                    "mean_des": best_safe["mean_des"],
                    "ckpt_sota": best_safe["ckpt_sota"],
                    "ckpt_edge": best_safe["ckpt_edge"],
                    "name": best_safe["name"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Patch infer_blend_deploy default checkpoint paths if different
        dep = Path("nafnet_denoise/infer_blend_deploy.py")
        text = dep.read_text(encoding="utf-8")
        if Path(best_safe["ckpt_sota"]).resolve() != stock_sota.resolve():
            # copy winning ckpt into a stable deploy path
            deploy_sota = Path("nafnet_denoise/checkpoints_4f_p9_deploy_sota")
            deploy_sota.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(best_safe["ckpt_sota"], deploy_sota / "best.pt")
            text = text.replace(
                "nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt",
                "nafnet_denoise/checkpoints_4f_p9_deploy_sota/best.pt",
            )
        if Path(best_safe["ckpt_edge"]).resolve() != stock_edge.resolve():
            deploy_edge = Path("nafnet_denoise/checkpoints_4f_p9_deploy_edge")
            deploy_edge.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(best_safe["ckpt_edge"], deploy_edge / "best.pt")
            text = text.replace(
                "nafnet_denoise/checkpoints_4f_lap_edge_ms/best.pt",
                "nafnet_denoise/checkpoints_4f_p9_deploy_edge/best.pt",
            )
        dep.write_text(text, encoding="utf-8")
        print("Patched infer_blend_deploy checkpoint defaults", flush=True)
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
