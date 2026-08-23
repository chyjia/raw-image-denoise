"""P3-b: train learned BlendGate for SOTA↔edge soft fusion (freeze both nets)."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .blend_gate import BlendGate, blend_with_gate
from .burst_dataset import RealBurstDataset
from .common import center_frame_index, decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .compare_adaptive_vs_bm3d import vst_bm3d
from .compare_sota_edgekd_blend import _spatial_on_merged, blend_edge_soft
from .fe_schedule import gated_spatial_from_temporal
from .infer import load_model
from .train import charbonnier_loss
from .train_distill import (
    RealNAFNetDistillDataset,
    collate_distill,
    denoise_edge_score,
    edge_metrics_dn,
    soft_edge_mask,
)
from .benchmark_multiframe import metric_row, reliable_reference_mask
from .validate import aligned_temporal_trimmed_mean, central_roi
from .wiener_merge import merge_from_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edge",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_lap_edge/best.pt"),
    )
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=Path(
            "nafnet_denoise/burst_cache_denoise_material/manifest_hard_flat.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_blend_gate"),
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path("nafnet_denoise/compare_blend_gate"),
    )
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\denoise\val_raw"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patches-per-epoch", type=int, default=192)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--blend-weight", type=float, default=1.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_sota, n_in, _ = load_model(args.checkpoint_sota, None, device)
    model_edge, n_e, _ = load_model(args.checkpoint_edge, None, device)
    if n_in != n_e:
        raise SystemExit(f"Frame mismatch {n_in} vs {n_e}")
    model_sota.eval()
    model_edge.eval()
    for p in list(model_sota.parameters()) + list(model_edge.parameters()):
        p.requires_grad_(False)
    model_sota.wiener_front_end = False
    model_edge.wiener_front_end = False

    gate = BlendGate(temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    if not args.skip_train:
        if not args.real_manifest.exists():
            raise SystemExit(f"Missing {args.real_manifest}")
        dataset = RealNAFNetDistillDataset(
            RealBurstDataset(
                args.real_manifest,
                patch_size=256,
                input_frames=n_in,
                samples_per_epoch=args.patches_per_epoch,
                wiener_front_end=True,
                wiener_merge_frames=16,
                wiener_tile=32,
                wiener_overlap=16,
                wiener_c_factor=8.0,
                wiener_spatial=True,
                wiener_spatial_c_factor=1.0,
                wiener_spatial_adaptive=True,
                wiener_spatial_flat_c_mult=1.75,
                wiener_spatial_edge_c_mult=0.45,
                wiener_spatial_dark_boost=0.35,
                wiener_spatial_mask_harden=8.0,
                wiener_spatial_freq_gamma=0.15,
                wiener_fe_schedule="fps_sigma",
                align_soft_gate=True,
                align_threshold=0.03,
                align_temperature=0.015,
                seed=7,
            ),
            use_sigma=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.micro_batch,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_distill,
        )
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            gate.train()
            running = 0.0
            n_steps = 0
            t0 = time.time()
            for batch in loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                with torch.no_grad():
                    sota = model_sota(inputs).float()
                    edge = model_edge(inputs).float()
                guide = 0.5 * (sota + edge)
                edge_map = gate(guide)
                blended = blend_with_gate(sota, edge, edge_map)
                # Supervise gate toward temporal Sobel mask; blend toward temporal.
                with torch.no_grad():
                    mask_tgt = soft_edge_mask(targets, temperature=12.0)
                bce = F.binary_cross_entropy(edge_map.clamp(1e-4, 1 - 1e-4), mask_tgt)
                blend_l = charbonnier_loss(blended, targets)
                loss = args.mask_bce_weight * bce + args.blend_weight * blend_l
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                n_steps += 1
            mean_loss = running / max(n_steps, 1)
            print(
                f"epoch {epoch}/{args.epochs} loss={mean_loss:.5f} "
                f"sec={time.time()-t0:.1f}",
                flush=True,
            )
            payload = {
                "model": gate.state_dict(),
                "args": {
                    "checkpoint_sota": str(args.checkpoint_sota),
                    "checkpoint_edge": str(args.checkpoint_edge),
                    "temperature": args.temperature,
                },
                "epoch": epoch,
                "loss": mean_loss,
            }
            torch.save(payload, args.out_dir / "last.pt")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(payload, args.out_dir / "best.pt")
                print(f"  saved best loss={best_loss:.5f}", flush=True)

    if args.skip_compare:
        return

    ckpt = args.out_dir / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Missing gate ckpt {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    gate.load_state_dict(state["model"])
    gate.eval()

    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No RAW under {args.input_dir}")
    args.compare_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for path in files:
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        target_index = frames.shape[0] // 2
        ys, xs = central_roi(height, width)
        input_dn = decode_mono10(frames[target_index])
        temporal_ref = aligned_temporal_trimmed_mean(
            frames, target_index, ys, xs, frames.shape[0],
            min(16, frames.shape[0]), trim_fraction=0.10,
            exclude_frame_index=target_index,
        )
        mask = reliable_reference_mask(input_dn, temporal_ref)
        temporal, _ = merge_from_memmap(
            frames, target_index, input_frames=16, tile_size=32, overlap=16,
            c_factor=8.0, align=True, spatial_wiener=False,
        )
        fe_gated, params, fe_sigma = gated_spatial_from_temporal(
            temporal, fps=float(fps), n_frames_averaged=16, tile_size=32, overlap=16,
        )
        sota_dn = _spatial_on_merged(
            model_sota, fe_gated, exposure_ms, device, n_in, 256
        )
        edge_dn = _spatial_on_merged(
            model_edge, fe_gated, exposure_ms, device, n_in, 256
        )
        soft_dn, _ = blend_edge_soft(
            sota_dn, edge_dn, (0.5 * sota_dn + 0.5 * edge_dn).astype(np.float32)
        )
        guide_t = torch.from_numpy(
            np.ascontiguousarray(0.5 * (sota_dn + edge_dn), dtype=np.float32)
        )[None, None].to(device)
        with torch.no_grad():
            emap = gate(guide_t)[0, 0].cpu().numpy()
        gate_dn = (emap * edge_dn + (1.0 - emap) * sota_dn).astype(np.float32)

        outputs = {
            "input": input_dn,
            "temporal_reference": temporal_ref,
            "blend_soft": soft_dn,
            "blend_gate": gate_dn,
            "sota_gated": sota_dn,
            "edge_gated": edge_dn,
            "vst_bm3d": vst_bm3d(input_dn, exposure_ms, tile_size=512),
        }
        input_sigma = float(
            metric_row(
                path, fps, target_index, "input", input_dn, temporal_ref, mask, ys, xs
            )["roi_highpass_noise_sigma_dn"]
        )
        des_by = {}
        for method, image in outputs.items():
            row = metric_row(
                path, fps, target_index, method, image, temporal_ref, mask, ys, xs,
                checkpoint=method, input_frames=16,
            )
            retention, edge_mae = edge_metrics_dn(image, temporal_ref)
            des, ng, ef = denoise_edge_score(
                float(row["roi_highpass_noise_sigma_dn"]), input_sigma, retention
            )
            row["edge_retention"] = f"{retention:.6f}"
            row["edge_sobel_mae"] = f"{edge_mae:.6f}"
            row["noise_gain"] = f"{ng:.6f}"
            row["edge_fidelity"] = f"{ef:.6f}"
            row["des"] = f"{des:.6f}"
            row["gated_schedule"] = params.name
            rows.append(row)
            des_by[method] = des
        print(
            f"{path.name}: soft={des_by.get('blend_soft',0):.4f} "
            f"gate={des_by.get('blend_gate',0):.4f} "
            f"bm3d={des_by.get('vst_bm3d',0):.4f}",
            flush=True,
        )

    metrics_path = args.compare_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by = defaultdict(list)
    for row in rows:
        by[row["method"]].append(float(row["des"]))
    print("--- mean DES ---", flush=True)
    for method, vals in sorted(by.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        if method in ("input", "temporal_reference"):
            continue
        print(f"{method}: {sum(vals)/len(vals):.4f}", flush=True)
    print(f"Wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
