"""P17 meta: Fourier LF/HF fuse + output geometric ensemble on P13 D4.

Baseline: P13 d4_r90 DES ≈ 0.9252
Lit (after P16 shift saturation):
  Self-Bootstrapping TTA (ICML'25) — Fourier LF amplitude masking/mix signals
  DualEx / WIFE — HF residual from edge, LF from flat SOTA (freq already in P16)
  NTIRE geometric self-ensemble — cheap D4 on *output* (post-deploy) vs FE

Uses cached ``*_sota_d4.npy`` / ``*_edge_d4.npy`` from P16 (zero re-forward).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .p7_fusion import edge_unsharp
from .p8_fusion import deploy_p7_base, multi_scale_unsharp, clipped_unsharp
from .run_des_loop100 import F1, F05, F10, edge_safe, score_image

BASELINE = 0.9252


@dataclass
class Recipe:
    name: str
    family: str  # baseline | fourier | out_geom | ms_unsharp | clip_u | amp_mix
    # fourier / amp
    keep_lf_frac: float = 0.15
    edge_hf: float = 0.85
    sota_lf: float = 1.0
    # out geom
    n_aug: int = 4
    # unsharp
    amounts: tuple[float, ...] = (0.1, 0.08, 0.05)
    sigmas: tuple[float, ...] = (0.8, 1.4, 2.5)
    u_amt: float = 0.15
    u_sig: float = 1.4
    clip_pct: float = 98.0


def build_100_recipes() -> list[Recipe]:
    recipes = [Recipe(name="p13_d4", family="baseline")]

    for lf, ehf in [
        (0.08, 0.9),
        (0.12, 0.85),
        (0.15, 0.85),
        (0.2, 0.8),
        (0.1, 1.0),
        (0.18, 0.75),
        (0.25, 0.7),
        (0.12, 0.95),
        (0.15, 0.7),
        (0.1, 0.85),
        (0.22, 0.85),
        (0.05, 0.9),
        (0.15, 0.6),
        (0.3, 0.8),
        (0.12, 0.65),
    ]:
        recipes.append(
            Recipe(
                name=f"fourier_lf{lf:g}_ehf{ehf:g}",
                family="fourier",
                keep_lf_frac=lf,
                edge_hf=ehf,
            )
        )

    for lf, ehf, slf in [
        (0.15, 0.85, 1.0),
        (0.15, 0.9, 0.95),
        (0.12, 0.85, 0.9),
        (0.2, 0.8, 1.0),
        (0.1, 0.95, 1.0),
        (0.18, 0.75, 0.95),
        (0.15, 0.85, 0.85),
        (0.25, 0.7, 1.0),
    ]:
        recipes.append(
            Recipe(
                name=f"amp_lf{lf:g}_ehf{ehf:g}_slf{slf:g}",
                family="amp_mix",
                keep_lf_frac=lf,
                edge_hf=ehf,
                sota_lf=slf,
            )
        )

    for n in [2, 4, 8]:
        recipes.append(Recipe(name=f"out_geom_{n}", family="out_geom", n_aug=n))

    for amts, sigs in [
        ((0.08, 0.06, 0.04), (0.8, 1.4, 2.5)),
        ((0.10, 0.08, 0.05), (0.8, 1.4, 2.5)),
        ((0.12, 0.08, 0.04), (1.0, 1.6, 2.8)),
        ((0.06, 0.08, 0.06), (0.7, 1.2, 2.0)),
        ((0.10, 0.05), (1.0, 2.0)),
        ((0.15, 0.05, 0.03), (0.8, 1.5, 3.0)),
        ((0.05, 0.05, 0.05), (0.8, 1.4, 2.5)),
        ((0.12, 0.0, 0.08), (0.9, 1.4, 2.2)),
    ]:
        tag = "_".join(f"{a:g}" for a in amts)
        recipes.append(
            Recipe(
                name=f"msu_{tag}",
                family="ms_unsharp",
                amounts=amts,
                sigmas=sigs[: len(amts)],
            )
        )

    for amt, sig, pct in [
        (0.15, 1.4, 98.0),
        (0.18, 1.2, 97.0),
        (0.12, 1.6, 99.0),
        (0.16, 1.4, 95.0),
        (0.14, 1.5, 98.0),
        (0.20, 1.3, 97.0),
        (0.10, 1.8, 99.0),
        (0.15, 1.4, 96.0),
    ]:
        recipes.append(
            Recipe(
                name=f"clipu_a{amt:g}_s{sig:g}_p{pct:g}",
                family="clip_u",
                u_amt=amt,
                u_sig=sig,
                clip_pct=pct,
            )
        )

    # pad to 100 with fourier grid
    while len(recipes) < 100:
        i = len(recipes)
        lf = 0.08 + (i % 10) * 0.02
        ehf = 0.6 + (i % 8) * 0.05
        recipes.append(
            Recipe(
                name=f"pad_fourier_{i}",
                family="fourier",
                keep_lf_frac=lf,
                edge_hf=min(ehf, 1.0),
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


def _fft2(img: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))


def _ifft2(F: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(F)).real.astype(np.float32)


def fourier_lf_hf_fuse(
    sota: np.ndarray,
    edge: np.ndarray,
    keep_lf_frac: float = 0.15,
    edge_hf: float = 0.85,
) -> np.ndarray:
    """LF from SOTA, mix HF from edge in Fourier domain (Self-Bootstrapping cue)."""
    h, w = sota.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    # radial mask: keep center LF radius
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(np.sqrt(cy**2 + cx**2)) + 1e-6
    lf_r = float(keep_lf_frac) * r_max
    lf_mask = (r <= lf_r).astype(np.float32)
    hf_mask = 1.0 - lf_mask
    Fs = _fft2(sota)
    Fe = _fft2(edge)
    ehf = float(np.clip(edge_hf, 0.0, 1.0))
    F = Fs * lf_mask + ((1.0 - ehf) * Fs + ehf * Fe) * hf_mask
    return _ifft2(F)


def amp_phase_mix(
    sota: np.ndarray,
    edge: np.ndarray,
    keep_lf_frac: float = 0.15,
    edge_hf: float = 0.85,
    sota_lf: float = 1.0,
) -> np.ndarray:
    """Mix amplitudes in LF/HF bands; keep SOTA phase (structure)."""
    h, w = sota.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(np.sqrt(cy**2 + cx**2)) + 1e-6
    lf_mask = (r <= float(keep_lf_frac) * r_max).astype(np.float32)
    hf_mask = 1.0 - lf_mask
    Fs = _fft2(sota)
    Fe = _fft2(edge)
    As, Ps = np.abs(Fs), np.angle(Fs)
    Ae = np.abs(Fe)
    ehf = float(np.clip(edge_hf, 0.0, 1.0))
    slf = float(np.clip(sota_lf, 0.0, 1.0))
    A = (slf * As + (1.0 - slf) * Ae) * lf_mask + (
        (1.0 - ehf) * As + ehf * Ae
    ) * hf_mask
    F = A * np.exp(1j * Ps)
    return _ifft2(F)


def geom_ensemble(img: np.ndarray, n_aug: int = 4) -> np.ndarray:
    """Dihedral self-ensemble: mild unsharp under each geom, map back (NTIRE-style)."""
    x = np.ascontiguousarray(img, dtype=np.float32)

    def forward(a: np.ndarray, name: str) -> np.ndarray:
        if name == "id":
            return a
        if name == "lr":
            return np.fliplr(a)
        if name == "ud":
            return np.flipud(a)
        if name == "udlr":
            return np.flipud(np.fliplr(a))
        if name == "r90":
            return np.rot90(a, 1)
        if name == "r180":
            return np.rot90(a, 2)
        if name == "r270":
            return np.rot90(a, 3)
        if name == "r90_lr":
            return np.fliplr(np.rot90(a, 1))
        return a

    def inverse(a: np.ndarray, name: str) -> np.ndarray:
        if name == "id":
            return a
        if name == "lr":
            return np.fliplr(a)
        if name == "ud":
            return np.flipud(a)
        if name == "udlr":
            return np.fliplr(np.flipud(a))
        if name == "r90":
            return np.rot90(a, -1)
        if name == "r180":
            return np.rot90(a, -2)
        if name == "r270":
            return np.rot90(a, -3)
        if name == "r90_lr":
            return np.rot90(np.fliplr(a), -1)
        return a

    names = ["id"]
    if n_aug >= 2:
        names.append("lr")
    if n_aug >= 4:
        names.extend(["ud", "udlr"])
    if n_aug >= 8:
        names.extend(["r90", "r180", "r270", "r90_lr"])
    outs = []
    for n in names:
        t = forward(x, n)
        t = edge_unsharp(t, amount=0.08, sigma=1.2, harden=16.0)
        outs.append(inverse(t, n))
    return np.mean(np.stack(outs, 0), 0).astype(np.float32)


def apply_recipe(recipe: Recipe, cache_dir: Path, stem: str, fps: float) -> np.ndarray:
    sota = np.load(cache_dir / f"{stem}_sota_d4.npy")
    edge = np.load(cache_dir / f"{stem}_edge_d4.npy")
    if recipe.family == "baseline":
        return deploy_p7_base(sota, edge, fps)

    if recipe.family == "fourier":
        fused = fourier_lf_hf_fuse(
            sota, edge, recipe.keep_lf_frac, recipe.edge_hf
        )
        return deploy_p7_base(fused, fused, fps)

    if recipe.family == "amp_mix":
        fused = amp_phase_mix(
            sota,
            edge,
            recipe.keep_lf_frac,
            recipe.edge_hf,
            recipe.sota_lf,
        )
        return deploy_p7_base(fused, fused, fps)

    if recipe.family == "out_geom":
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0)
        return geom_ensemble(base, n_aug=recipe.n_aug)

    if recipe.family == "ms_unsharp":
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0)
        return multi_scale_unsharp(base, amounts=recipe.amounts, sigmas=recipe.sigmas)

    if recipe.family == "clip_u":
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0)
        return clipped_unsharp(
            base, amount=recipe.u_amt, sigma=recipe.u_sig, clip_pct=recipe.clip_pct
        )

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
    hook = Path("nafnet_denoise/deploy_p17_hook.json")
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
    note = Path("nafnet_denoise/compare_meta100_p17/BAKE_NOTE.txt")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"Winner {best['name']} DES={best['mean_des']:.4f}\nSee deploy_p17_hook.json\n",
        encoding="utf-8",
    )
    print(f"Wrote hook + DES stamp for {best['name']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p17")
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
            raise SystemExit(
                f"Missing D4 cache {path.stem}; run P16 cache_d4_arms first"
            )

    recipes = build_100_recipes()[: int(args.max_recipes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"\n===== P17 [{idx:03d}/{len(recipes)}] {recipe.name} =====", flush=True)
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
                f"Early stop: stagnant={stagnant} "
                f"best={None if best_safe is None else best_safe['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P17 done ---", flush=True)
    if best_safe and best_safe["mean_des"] >= BASELINE + 1e-4:
        (args.output_dir / "bake_recipe.json").write_text(
            json.dumps(best_safe, indent=2), encoding="utf-8"
        )
        print(f"BAKE {best_safe['name']} DES={best_safe['mean_des']:.4f}", flush=True)
        bake_deploy(best_safe)
    else:
        print(
            f"No bake (best="
            f"{None if best_safe is None else best_safe['mean_des']:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
