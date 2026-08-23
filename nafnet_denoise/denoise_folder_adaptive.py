"""Denoise all Mono10 RAW files in a folder with the adaptive 1f/4f/16f pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .infer import save_mono10_png
from .infer_adaptive import denoise_adaptive, load_adaptive_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-1f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_1f_ptc_black60/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-4f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_4f_ptc_black60/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-16f",
        type=Path,
        default=Path("nafnet_denoise/checkpoints_16f_ptc_black60_mixed/best.pt"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    models = load_adaptive_models(
        args.checkpoint_1f,
        args.checkpoint_4f,
        args.checkpoint_16f,
        device,
    )
    files = sorted(args.input_dir.rglob("*_pMono10_f*.raw"))
    if not files:
        raise SystemExit(f"No Mono10 RAW files under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"files={len(files)}", flush=True)

    for path in files:
        width, height, fps = parse_geometry(path.name)
        exposure_ms = parse_exposure_ms(path.name)
        frames = memmap_frames(path, width, height)
        frame_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index
        if frame_index < 0 or frame_index >= frames.shape[0]:
            raise SystemExit(f"{path.name}: frame-index out of range")

        out_dir = args.output_dir / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        input_dn = decode_mono10(frames[frame_index])
        output_dn, meta = denoise_adaptive(
            models,
            frames,
            frame_index,
            fps,
            exposure_ms,
            device,
            tile_size=args.tile_size,
        )
        route = str(meta["route"])
        save_mono10_png(out_dir / f"frame_{frame_index:03d}_input.png", input_dn)
        save_mono10_png(out_dir / f"frame_{frame_index:03d}_denoised_{route}.png", output_dn)
        print(
            f"{path.name}: frame={frame_index} fps={fps:g} exp={exposure_ms:.2f}ms "
            f"route={route} used={meta['used_frames']} "
            f"fallback={meta['fallback_used']} conf={meta['alignment_confidence']:.4f}",
            flush=True,
        )

    print(f"done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
