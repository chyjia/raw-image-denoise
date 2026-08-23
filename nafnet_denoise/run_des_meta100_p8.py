"""P8 meta DES loop — lit families beyond P7 bilat/residual/plain-unsharp saturation.

Baseline: P7 deploy (Gen2 + unsharp a=0.15 σ=1.4) ≈ 0.9221
Families:
  umgf     — Unsharp-Mask Guided Filtering (TIP'21)
  ghgif    — Gaussian highpass guided (arXiv'25)
  ms_unsharp — multi-scale edge unsharp
  clip_u   — clipped unsharp (anti-halo)
  refine   — local refine around P7 unsharp params
  post_umgf — P7 base then UMGF/GH with edge as guide

Up to 100 gens × 100 recipes; early-stop after ``patience`` stagnant gens.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .p8_fusion import (
    clipped_unsharp,
    deploy_p7_base,
    gh_gif,
    multi_scale_unsharp,
    umgf_fuse,
)
from .run_des_loop100 import F1, F05, F10, cache_forwards, edge_safe, score_image


@dataclass
class Recipe:
    name: str
    family: str = "p7"
    # unsharp refine
    u_amt: float = 0.15
    u_sig: float = 1.4
    # umgf / ghgif
    amount: float = 0.5
    sigma: float = 1.4
    edge_only: bool = True
    harden: float = 16.0
    # multi-scale
    ms_amts: tuple[float, ...] = (0.1, 0.08, 0.05)
    ms_sigs: tuple[float, ...] = (0.8, 1.4, 2.5)
    # clip
    clip_pct: float = 98.0
    # post on p7 base
    post: str = "none"  # none|umgf|ghgif|ms|clip|unsharp
    post_kw: dict = field(default_factory=dict)
    # replace unsharp in p7 with different params (family=refine)
    replace_unsharp: bool = False


def build_gen0(n: int = 100) -> list[Recipe]:
    recipes: list[Recipe] = [Recipe(name="p7_deploy", family="p7")]

    # refine unsharp grid around 0.15/1.4
    for amt in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
        for sig in [1.0, 1.2, 1.4, 1.6, 1.8]:
            recipes.append(
                Recipe(
                    name=f"ref_u{amt:g}_s{sig:g}",
                    family="refine",
                    u_amt=amt,
                    u_sig=sig,
                    replace_unsharp=True,
                )
            )

    # UMGF as post on p7 (guide=edge) and as full replace
    for amt, sig in [
        (0.25, 1.2),
        (0.35, 1.4),
        (0.45, 1.4),
        (0.55, 1.6),
        (0.4, 1.0),
        (0.3, 1.8),
        (0.5, 1.4),
        (0.2, 1.4),
        (0.6, 1.2),
        (0.35, 2.0),
    ]:
        recipes.append(
            Recipe(
                name=f"umgf_post_a{amt:g}_s{sig:g}",
                family="post",
                post="umgf",
                amount=amt,
                sigma=sig,
            )
        )
        recipes.append(
            Recipe(
                name=f"gh_post_a{amt:g}_s{sig:g}",
                family="post",
                post="ghgif",
                amount=amt,
                sigma=sig,
            )
        )

    # multi-scale
    for i, (amts, sigs) in enumerate(
        [
            ((0.08, 0.06, 0.04), (0.8, 1.4, 2.5)),
            ((0.12, 0.08, 0.04), (0.8, 1.4, 2.5)),
            ((0.1, 0.1, 0.05), (1.0, 1.5, 2.5)),
            ((0.15, 0.05, 0.0), (1.0, 2.0, 3.0)),
            ((0.06, 0.06, 0.06), (0.8, 1.4, 2.2)),
            ((0.1, 0.05), (1.2, 2.0)),
            ((0.12, 0.08, 0.06, 0.03), (0.7, 1.2, 1.8, 2.8)),
            ((0.05, 0.1, 0.05), (0.8, 1.4, 2.5)),
        ]
    ):
        recipes.append(
            Recipe(
                name=f"ms_{i}",
                family="post",
                post="ms",
                ms_amts=amts,
                ms_sigs=sigs,
            )
        )

    # clipped unsharp
    for amt, pct in [
        (0.15, 95.0),
        (0.15, 98.0),
        (0.18, 97.0),
        (0.12, 99.0),
        (0.20, 96.0),
        (0.15, 90.0),
        (0.22, 98.0),
        (0.10, 98.0),
    ]:
        recipes.append(
            Recipe(
                name=f"clip_a{amt:g}_p{pct:g}",
                family="post",
                post="clip",
                amount=amt,
                sigma=1.4,
                clip_pct=pct,
            )
        )

    # UMGF replacing unsharp entirely (p7 bilat then umgf with edge guide)
    for amt, sig in [
        (0.15, 1.4),
        (0.2, 1.4),
        (0.25, 1.2),
        (0.3, 1.4),
        (0.18, 1.6),
    ]:
        recipes.append(
            Recipe(
                name=f"umgf_repl_a{amt:g}_s{sig:g}",
                family="umgf_repl",
                amount=amt,
                sigma=sig,
                u_amt=0.0,  # no classic unsharp
            )
        )

    recipes = recipes[:n]
    while len(recipes) < n:
        i = len(recipes)
        recipes.append(
            Recipe(
                name=f"pad_ref_{i}",
                family="refine",
                u_amt=0.1 + 0.01 * (i % 15),
                u_sig=1.0 + 0.1 * (i % 10),
                replace_unsharp=True,
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
    return out[:n]


def refine_around(best: Recipe | None, gen: int, n: int = 100) -> list[Recipe]:
    if gen == 0:
        return build_gen0(n)

    fams = ["refine", "umgf", "ghgif", "ms", "clip", "post"]
    fam = fams[gen % len(fams)]
    recipes: list[Recipe] = [Recipe(name="p7_deploy", family="p7")]
    if best is not None:
        recipes.append(Recipe(**{**asdict(best), "name": f"seed_{best.name}"[:55]}))

    if fam == "refine":
        ba = float(best.u_amt) if best else 0.15
        bs = float(best.u_sig) if best else 1.4
        for i in range(50):
            recipes.append(
                Recipe(
                    name=f"g{gen}_ref_{i}",
                    family="refine",
                    u_amt=max(0.05, ba + (i % 11 - 5) * 0.02),
                    u_sig=max(0.6, bs + (i % 9 - 4) * 0.1),
                    replace_unsharp=True,
                )
            )
    elif fam in ("umgf", "ghgif"):
        ba = float(best.amount) if best and best.family != "p7" else 0.35
        bs = float(best.sigma) if best and best.family != "p7" else 1.4
        for i in range(40):
            recipes.append(
                Recipe(
                    name=f"g{gen}_{fam}_{i}",
                    family="post",
                    post=fam if fam != "umgf" else "umgf",
                    amount=max(0.1, ba + (i % 9 - 4) * 0.05),
                    sigma=max(0.6, bs + (i % 7 - 3) * 0.15),
                )
            )
    elif fam == "ms":
        for i in range(30):
            a0 = 0.05 + (i % 8) * 0.015
            recipes.append(
                Recipe(
                    name=f"g{gen}_ms_{i}",
                    family="post",
                    post="ms",
                    ms_amts=(a0, a0 * 0.7, a0 * 0.4),
                    ms_sigs=(0.8 + (i % 3) * 0.2, 1.4, 2.2 + (i % 4) * 0.2),
                )
            )
    elif fam == "clip":
        for i in range(30):
            recipes.append(
                Recipe(
                    name=f"g{gen}_clip_{i}",
                    family="post",
                    post="clip",
                    amount=0.1 + (i % 10) * 0.02,
                    sigma=1.2 + (i % 5) * 0.15,
                    clip_pct=92.0 + (i % 8),
                )
            )
    else:
        for i in range(40):
            recipes.append(
                Recipe(
                    name=f"g{gen}_hy_{i}",
                    family="post",
                    post="umgf" if i % 2 == 0 else "ghgif",
                    amount=0.2 + (i % 8) * 0.05,
                    sigma=1.0 + (i % 6) * 0.2,
                )
            )

    i = 0
    while len(recipes) < n:
        recipes.append(
            Recipe(
                name=f"g{gen}_pad_{i}",
                family="refine",
                u_amt=0.1 + 0.01 * (i % 20),
                u_sig=1.0 + 0.05 * (i % 16),
                replace_unsharp=True,
            )
        )
        i += 1

    seen: set[str] = set()
    uniq: list[Recipe] = []
    for r in recipes[:n]:
        name = r.name
        if name in seen:
            name = f"{name}_x{len(uniq)}"
            r = Recipe(**{**asdict(r), "name": name})
        seen.add(name)
        uniq.append(r)
    return uniq[:n]


def apply_recipe(recipe: Recipe, sota: np.ndarray, edge: np.ndarray, fps: float) -> np.ndarray:
    if recipe.family == "p7":
        return deploy_p7_base(sota, edge, fps)

    if recipe.family == "refine" or recipe.replace_unsharp:
        return deploy_p7_base(
            sota, edge, fps, unsharp_amount=recipe.u_amt, unsharp_sigma=recipe.u_sig
        )

    if recipe.family == "umgf_repl":
        # bilat path without classic unsharp, then UMGF with edge guide
        base = deploy_p7_base(sota, edge, fps, unsharp_amount=0.0)
        return umgf_fuse(
            base, edge, amount=recipe.amount, sigma=recipe.sigma, edge_only=True
        )

    # post on full p7 deploy
    base = deploy_p7_base(sota, edge, fps)
    if recipe.post == "umgf":
        return umgf_fuse(
            base,
            edge,
            amount=recipe.amount,
            sigma=recipe.sigma,
            edge_only=recipe.edge_only,
            harden=recipe.harden,
        )
    if recipe.post == "ghgif":
        return gh_gif(
            base,
            edge,
            amount=recipe.amount,
            sigma=recipe.sigma,
            edge_only=recipe.edge_only,
            harden=recipe.harden,
        )
    if recipe.post == "ms":
        return multi_scale_unsharp(
            base, amounts=recipe.ms_amts, sigmas=recipe.ms_sigs, harden=recipe.harden
        )
    if recipe.post == "clip":
        return clipped_unsharp(
            base,
            amount=recipe.amount if recipe.amount > 0 else recipe.u_amt,
            sigma=recipe.sigma,
            clip_pct=recipe.clip_pct,
            harden=recipe.harden,
        )
    if recipe.post == "unsharp":
        from .p7_fusion import edge_unsharp

        return edge_unsharp(base, amount=recipe.u_amt, sigma=recipe.u_sig)
    return base


def run_generation(
    packs: list[dict],
    cache_dir: Path,
    recipes: list[Recipe],
    out_dir: Path,
    gen: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_safe: dict | None = None
    best_any: dict | None = None
    for idx, recipe in enumerate(recipes):
        des_list: list[float] = []
        f10 = f05 = f1 = None
        for pack in packs:
            meta = pack["meta"]
            out = apply_recipe(recipe, pack["sota"], pack["edge"], meta["fps"])
            sc = score_image(meta, cache_dir, recipe.name, out)
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
            "gen": gen,
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
        if best_any is None or mean_des > best_any["mean_des"]:
            best_any = rec
        if ok and (best_safe is None or mean_des > best_safe["mean_des"]):
            best_safe = rec
            print(
                f"[g{gen:02d} {idx:03d}/{len(recipes)}] NEW SAFE {recipe.name} "
                f"DES={mean_des:.4f}",
                flush=True,
            )
        elif idx == 0 or (idx + 1) % 20 == 0:
            print(
                f"[g{gen:02d} {idx:03d}/{len(recipes)}] {recipe.name}: "
                f"DES={mean_des:.4f} ({'ok' if ok else 'unsafe'})",
                flush=True,
            )

    csv_path = out_dir / f"gen{gen:02d}_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gen",
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
    summary = {"gen": gen, "best_safe": best_safe, "best_any": best_any, "n": len(history)}
    (out_dir / f"gen{gen:02d}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def bake_if_better(best: dict, baseline: float) -> None:
    """Patch infer_blend_deploy unsharp defaults when refine wins."""
    if best is None or float(best["mean_des"]) < baseline + 1e-4:
        print("No bake: not above P7 baseline", flush=True)
        return
    recipe = best.get("recipe") or {}
    path = Path("nafnet_denoise/infer_blend_deploy.py")
    text = path.read_text(encoding="utf-8")
    fam = recipe.get("family")
    post = recipe.get("post", "none")
    bake_dir = Path("nafnet_denoise/compare_meta100_p8")
    bake_dir.mkdir(parents=True, exist_ok=True)
    (bake_dir / "bake_recipe.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )

    if fam == "refine" or recipe.get("replace_unsharp"):
        amt = float(recipe.get("u_amt", 0.15))
        sig = float(recipe.get("u_sig", 1.4))
        # update defaults in function signature and argparse
        import re

        text2 = re.sub(
            r"unsharp_amount: float = [0-9.]+",
            f"unsharp_amount: float = {amt}",
            text,
            count=1,
        )
        text2 = re.sub(
            r"unsharp_sigma: float = [0-9.]+",
            f"unsharp_sigma: float = {sig}",
            text2,
            count=1,
        )
        text2 = re.sub(
            r'default=0\.\d+,?\n(\s*)help=.*unsharp-amount|'
            r'--unsharp-amount", type=float, default=[0-9.]+',
            f'--unsharp-amount", type=float, default={amt}',
            text2,
            count=1,
        )
        text2 = re.sub(
            r'--unsharp-amount", type=float, default=[0-9.]+',
            f'--unsharp-amount", type=float, default={amt}',
            text2,
            count=1,
        )
        text2 = re.sub(
            r'--unsharp-sigma", type=float, default=[0-9.]+',
            f'--unsharp-sigma", type=float, default={sig}',
            text2,
            count=1,
        )
        # docstring mean
        text2 = re.sub(
            r"mean DES ~0\.\d+",
            f"mean DES ~{best['mean_des']:.4f}",
            text2,
            count=1,
        )
        text2 = re.sub(
            r"g4_unsharp_23",
            str(best["name"])[:40],
            text2,
            count=1,
        )
        text2 = re.sub(
            r"amount=0\.\d+, sigma=1\.\d+",
            f"amount={amt}, sigma={sig}",
            text2,
            count=1,
        )
        path.write_text(text2, encoding="utf-8")
        print(f"Baked unsharp a={amt} σ={sig} into {path}", flush=True)
    else:
        note = (
            f"Winner {best['name']} family={fam} post={post} DES={best['mean_des']:.4f}\n"
            f"Needs code path beyond plain unsharp defaults — see bake_recipe.json\n"
        )
        (bake_dir / "BAKE_NOTE.txt").write_text(note, encoding="utf-8")
        print(note, flush=True)
        # For umgf_repl / post that beat baseline: still try to encode as unsharp
        # if it's umgf_repl with small amount — else leave note for deploy patch
        if fam == "umgf_repl" or post in ("umgf", "ghgif", "ms", "clip"):
            # Write a small deploy hook file used by infer if present
            hook = Path("nafnet_denoise/deploy_p8_hook.json")
            hook.write_text(
                json.dumps(
                    {
                        "name": best["name"],
                        "mean_des": best["mean_des"],
                        "recipe": recipe,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Wrote deploy hook {hook}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("nafnet_denoise/cache_dual_holdout")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nafnet_denoise/compare_meta100_p8")
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
    parser.add_argument("--max-gens", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--start-gen", type=int, default=0)
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
    packs = [
        {
            "meta": m,
            "sota": np.load(args.cache_dir / f"{m['stem']}_sota.npy"),
            "edge": np.load(args.cache_dir / f"{m['stem']}_edge.npy"),
        }
        for m in metas
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_best: dict | None = None
    best_recipe: Recipe | None = None
    stagnant = 0
    ledger: list[dict] = []
    baseline = 0.9221

    for gen in range(int(args.start_gen), int(args.max_gens)):
        print(f"\n===== P8 META GEN {gen} =====", flush=True)
        recipes = refine_around(best_recipe, gen, n=int(args.iters))
        summary = run_generation(packs, args.cache_dir, recipes, args.output_dir, gen)
        bs = summary.get("best_safe")
        improved = False
        if bs is not None:
            if global_best is None or bs["mean_des"] > global_best["mean_des"] + 1e-4:
                global_best = bs
                best_recipe = Recipe(**bs["recipe"])
                improved = True
                stagnant = 0
                (args.output_dir / "global_best.json").write_text(
                    json.dumps(global_best, indent=2), encoding="utf-8"
                )
                print(
                    f"GLOBAL BEST ← {bs['name']} DES={bs['mean_des']:.4f}",
                    flush=True,
                )
            else:
                stagnant += 1
        else:
            stagnant += 1
        if gen == 0 and bs is not None and bs["name"] == "p7_deploy":
            baseline = float(bs["mean_des"])
        ledger.append(
            {
                "gen": gen,
                "best_safe_des": None if bs is None else bs["mean_des"],
                "best_safe_name": None if bs is None else bs["name"],
                "improved": improved,
                "stagnant": stagnant,
            }
        )
        (args.output_dir / "ledger.json").write_text(
            json.dumps(
                {"ledger": ledger, "global_best": global_best, "baseline_p7": baseline},
                indent=2,
            ),
            encoding="utf-8",
        )
        if stagnant >= int(args.patience):
            print(
                f"Early stop: {stagnant} gens without +1e-4 DES. "
                f"global_best="
                f"{None if global_best is None else global_best['mean_des']:.4f}",
                flush=True,
            )
            break

    print("--- P8 meta done ---", flush=True)
    if global_best:
        print(
            f"FINAL {global_best['name']} mean={global_best['mean_des']:.4f}",
            flush=True,
        )
        bake_if_better(global_best, baseline)


if __name__ == "__main__":
    main()
