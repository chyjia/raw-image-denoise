"""Run current best deploy (P150 BiShrink+SURE-k) on material folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .benchmark_multiframe import metric_row, reliable_reference_mask
from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import load_model, save_mono10_png
from .infer_blend_deploy import denoise_blend_deploy
from .train_distill import denoise_edge_score, edge_metrics_dn
from .validate import aligned_temporal_trimmed_mean, central_roi, save_preview


def _clip_done(sequence_dir: Path) -> bool:
    return any(sequence_dir.glob("frame_*_comparison.png"))


def _save_clip_metrics(sequence_dir: Path, clip: dict) -> None:
    (sequence_dir / "clip_metrics.json").write_text(
        json.dumps(clip, indent=2), encoding="utf-8"
    )


def _load_all_clip_metrics(output_dir: Path) -> list[dict]:
    clips: list[dict] = []
    for metrics_path in sorted(output_dir.glob("Video_*/clip_metrics.json")):
        clips.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    return clips


def _process_clip(
    path: Path,
    *,
    output_dir: Path,
    model_sota,
    model_edge,
    n_in: int,
    device: torch.device,
    frame_index: int,
    temporal_window: int,
    tile_size: int,
) -> tuple[dict[str, str], dict]:
    width, height, fps = parse_geometry(path.name)
    exposure_ms = parse_exposure_ms(path.name)
    frames = memmap_frames(path, width, height)
    target_index = frames.shape[0] // 2 if frame_index < 0 else frame_index
    ys, xs = central_roi(height, width)
    input_dn = decode_mono10(frames[target_index])
    temporal = aligned_temporal_trimmed_mean(
        frames,
        target_index,
        ys,
        xs,
        frames.shape[0],
        min(temporal_window, frames.shape[0]),
        trim_fraction=0.10,
        exclude_frame_index=target_index,
    )
    mask = reliable_reference_mask(input_dn, temporal)

    print(f"\n{path.name} fps={fps} frames={frames.shape[0]} ...", flush=True)
    deploy_dn, deploy_meta = denoise_blend_deploy(
        model_sota,
        model_edge,
        n_in,
        frames,
        target_index,
        float(fps),
        exposure_ms,
        device,
        tile=tile_size,
        temporal_window=temporal_window,
    )
    print(
        f"  route={deploy_meta.get('route')} ans_mode={deploy_meta.get('ans_mode')}",
        flush=True,
    )

    sequence_dir = output_dir / path.stem
    sequence_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"frame_{target_index:03d}"
    save_mono10_png(sequence_dir / f"{prefix}_input.png", input_dn)
    save_mono10_png(sequence_dir / f"{prefix}_p150_deploy.png", deploy_dn)
    save_preview(
        sequence_dir / f"{prefix}_comparison.png",
        [input_dn, deploy_dn],
        ["Input", "P150 deploy"],
    )

    input_sigma = float(
        metric_row(
            path, float(fps), target_index, "input", input_dn, temporal, mask, ys, xs
        )["roi_highpass_noise_sigma_dn"]
    )
    row = metric_row(
        path,
        float(fps),
        target_index,
        "p150_deploy",
        deploy_dn,
        temporal,
        mask,
        ys,
        xs,
        checkpoint="p150",
        input_frames=16,
    )
    retention, edge_mae = edge_metrics_dn(deploy_dn, temporal)
    sigma = float(row["roi_highpass_noise_sigma_dn"])
    des, noise_gain, edge_fidelity = denoise_edge_score(
        sigma, input_sigma, retention
    )
    row.update(
        {
            "edge_retention": f"{retention:.6f}",
            "edge_sobel_mae": f"{edge_mae:.6f}",
            "noise_gain": f"{noise_gain:.6f}",
            "edge_fidelity": f"{edge_fidelity:.6f}",
            "des": f"{des:.6f}",
        }
    )
    clip = {
        "file": path.name,
        "fps": float(fps),
        "des": float(des),
        "noise_gain": float(noise_gain),
        "edge_fidelity": float(edge_fidelity),
    }
    _save_clip_metrics(sequence_dir, clip)
    print(
        f"  DES={des:.4f} ng={noise_gain:.4f} ef={edge_fidelity:.4f} "
        f"saved {sequence_dir / f'{prefix}_comparison.png'}",
        flush=True,
    )
    return row, clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"D:\denoise\素材\降噪素材"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/output_material_p150"),
    )
    parser.add_argument(
        "--checkpoint-sota",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_split_edge_fid/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-edgekd",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_p12_deploy_edge/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--temporal-window", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, only process first N files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips that already have frame_*_comparison.png.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.raw"))
    if args.max_files > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"No .raw under {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} files={len(files)} input={args.input_dir}", flush=True)
    model_sota, n_in, _ = load_model(args.checkpoint_sota, None, device)
    model_edge, n_edge, _ = load_model(args.checkpoint_edgekd, None, device)
    if n_in != n_edge:
        raise SystemExit(f"Frame mismatch {n_in} vs {n_edge}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # backfill clip_metrics.json for f1 from legacy metrics.csv if needed
    f1_dir = args.output_dir / "Video_20260719153741354_w1920_h1200_pMono10_f1"
    f1_metrics = f1_dir / "clip_metrics.json"
    if f1_dir.exists() and not f1_metrics.exists():
        legacy = args.output_dir / "metrics.csv"
        if legacy.exists():
            with legacy.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("file", "").endswith("_f1.raw"):
                        _save_clip_metrics(
                            f1_dir,
                            {
                                "file": row["file"],
                                "fps": float(row["fps"]),
                                "des": float(row["des"]),
                                "noise_gain": float(row["noise_gain"]),
                                "edge_fidelity": float(row["edge_fidelity"]),
                            },
                        )
                        break

    rows: list[dict[str, str]] = []

    for path in files:
        sequence_dir = args.output_dir / path.stem
        if args.skip_existing and _clip_done(sequence_dir):
            print(f"skip existing {path.name}", flush=True)
            continue
        row, _clip = _process_clip(
            path,
            output_dir=args.output_dir,
            model_sota=model_sota,
            model_edge=model_edge,
            n_in=n_in,
            device=device,
            frame_index=args.frame_index,
            temporal_window=args.temporal_window,
            tile_size=args.tile_size,
        )
        rows.append(row)

    # backfill metrics for PNG-only folders (comparison exists, no clip_metrics.json)
    for path in files:
        sequence_dir = args.output_dir / path.stem
        if _clip_done(sequence_dir) and not (sequence_dir / "clip_metrics.json").exists():
            print(f"backfill metrics {path.name}", flush=True)
            row, _clip = _process_clip(
                path,
                output_dir=args.output_dir,
                model_sota=model_sota,
                model_edge=model_edge,
                n_in=n_in,
                device=device,
                frame_index=args.frame_index,
                temporal_window=args.temporal_window,
                tile_size=args.tile_size,
            )
            rows.append(row)

    summary = _load_all_clip_metrics(args.output_dir)
    mean_des = float(sum(s["des"] for s in summary) / max(len(summary), 1))
    out_summary = {"mean_des": mean_des, "n_clips": len(summary), "clips": summary}
    (args.output_dir / "summary.json").write_text(
        json.dumps(out_summary, indent=2), encoding="utf-8"
    )

    all_rows: list[dict[str, str]] = []
    metrics_csv = args.output_dir / "metrics.csv"
    if metrics_csv.exists() and not rows:
        with metrics_csv.open(encoding="utf-8", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
    if rows:
        all_rows.extend(rows)
    if all_rows:
        with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(
        f"\nDone {len(summary)}/11 clips mean_DES={mean_des:.4f} -> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
