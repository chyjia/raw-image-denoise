"""Score P150 (current SOTA) on holdout cache and save preview PNGs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .des_cycle_common import P19_TAG, load_arms
from .infer import save_mono10_png
from .p8_fusion import deploy_p150
from .run_des_loop100 import F05, F1, F10, edge_safe, score_image
from .validate import save_preview

P150_KW = dict(
    k_list=(0.8, 1.0, 1.2),
    n_iters=5,
    thr_decay=0.7,
    ans_sigma=1.8,
    residual_scale=0.3,
)


def main() -> None:
    cache = Path("nafnet_denoise/cache_dual_holdout")
    out = Path("nafnet_denoise/compare_p150_demo")
    out.mkdir(parents=True, exist_ok=True)

    metas = sorted(cache.glob("*.json"), key=lambda p: p.name)
    rows: list[dict] = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stem = meta["stem"]
        fps = float(meta["fps"])
        sota, edge = load_arms(cache, stem, P19_TAG)
        den = deploy_p150(sota, edge, fps, **P150_KW)
        sc = score_image(meta, cache, "p150", den)
        rows.append({"file": stem, "fps": fps, **sc})
        print(
            f"{stem[-28:]} fps={fps:g} DES={sc['des']:.4f} "
            f"ng={sc['noise_gain']:.4f} ef={sc['edge_fidelity']:.4f}",
            flush=True,
        )

    f10 = next(r for r in rows if F10 in r["file"])
    f05 = next(r for r in rows if F05 in r["file"])
    f1 = next(r for r in rows if F1 in r["file"])
    summary = {
        "mean_des": float(sum(r["des"] for r in rows) / len(rows)),
        "f10_ef": f10["edge_fidelity"],
        "f05_ef": f05["edge_fidelity"],
        "f05_ng": f05["noise_gain"],
        "f1_ef": f1["edge_fidelity"],
        "edge_safe": edge_safe(
            f10["edge_fidelity"], f05["edge_fidelity"], f1["edge_fidelity"]
        ),
        "recipe": "bi_0.8_1_1.2_n5_d0.7",
        "model": "P150 BiShrink+SURE-k",
    }
    print("\n=== SUMMARY ===", flush=True)
    print(summary, flush=True)

    with (out / "per_clip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "fps",
                "des",
                "noise_gain",
                "edge_fidelity",
                "edge_retention",
                "edge_sobel_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preview_stems = [
        "Video_20260719153940814_w1920_h1200_pMono10_f0.5",
        "Video_20260725174703305_w1920_h1200_pMono10_f5",
        "Video_20260725174824541_w1920_h1200_pMono10_f10",
    ]
    for stem in preview_stems:
        meta = json.loads((cache / f"{stem}.json").read_text(encoding="utf-8"))
        fps = float(meta["fps"])
        sota, edge = load_arms(cache, stem, P19_TAG)
        inp = np.load(cache / f"{stem}_input.npy")
        ref = np.load(cache / f"{stem}_ref.npy")
        den = deploy_p150(sota, edge, fps, **P150_KW)
        d = out / stem
        d.mkdir(exist_ok=True)
        save_mono10_png(d / "input.png", inp)
        save_mono10_png(d / "sota_arm.png", sota)
        save_mono10_png(d / "p150_deploy.png", den)
        save_mono10_png(d / "temporal_ref.png", ref)
        save_preview(
            d / "comparison.png",
            [inp, sota, den, ref],
            ["Input", "SOTA arm", "P150 deploy", "Temporal ref"],
        )
        print(f"preview -> {d / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
