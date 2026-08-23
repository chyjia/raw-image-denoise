"""Inference-side Adaptive ⊕ Dual-head ensemble (flat→dual, edge→adaptive)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import decode_mono10, memmap_frames, parse_exposure_ms, parse_geometry
from .dual_head_nafnet import soft_edge_mask
from .infer import denoise_frame, load_model, save_mono10_png
from .infer_adaptive import AdaptiveModels, denoise_adaptive, load_adaptive_models


def soft_edge_mask_dn(
    image_dn: np.ndarray,
    temperature: float = 8.0,
) -> np.ndarray:
    """Soft Sobel edge map in [0, 1]; flats near 0, edges near 1."""
    tensor = torch.from_numpy(np.ascontiguousarray(image_dn, dtype=np.float32))[
        None, None
    ]
    return soft_edge_mask(tensor, temperature=temperature)[0, 0].cpu().numpy()


def blend_adaptive_dual(
    adaptive_dn: np.ndarray,
    dual_dn: np.ndarray,
    guide_dn: np.ndarray,
    temperature: float = 8.0,
    flat_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend: ``edge * adaptive + flat * dual``.

    ``flat_bias`` > 0 shifts more weight toward dual on near-flat regions
    (``edge_map = sigmoid(T*(scaled-0.5) - flat_bias)``).
    """
    edge_map = soft_edge_mask_dn(guide_dn, temperature=temperature)
    if flat_bias != 0.0:
        logit = np.log(
            edge_map.clip(1e-6, 1.0 - 1e-6) / (1.0 - edge_map.clip(1e-6, 1.0 - 1e-6))
        )
        edge_map = 1.0 / (1.0 + np.exp(-(logit - flat_bias)))
    flat_map = 1.0 - edge_map
    blended = edge_map * adaptive_dn + flat_map * dual_dn
    return blended.astype(np.float32), edge_map.astype(np.float32)


def harden_edge_map_dn(edge_map: np.ndarray, harden: float) -> np.ndarray:
    """Sharpen soft edge weights toward 0/1 (P4-a)."""
    if harden <= 0.0:
        return edge_map
    return 1.0 / (1.0 + np.exp(-float(harden) * (edge_map - 0.5)))


def blend_sota_edgekd(
    sota_dn: np.ndarray,
    edgekd_dn: np.ndarray,
    guide_dn: np.ndarray | None = None,
    temperature: float = 8.0,
    edge_weight: float = 1.0,
    harden: float = 0.0,
    edge_power: float = 1.0,
    edge_gain: float = 1.0,
    residual_mix: float = 0.0,
    residual_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deploy blend: ``(w*edge)*edgeKD + (1-w*edge)*SOTA``.

    P4-a holdout winner: ``temperature=8, harden=16`` (mean DES ~0.9060).
    ``edge_weight`` in [0, 1] scales edge arm (0 → pure SOTA).
    """
    if guide_dn is None:
        guide_dn = (0.5 * sota_dn + 0.5 * edgekd_dn).astype(np.float32)
    edge_map = soft_edge_mask_dn(guide_dn, temperature=temperature)
    edge_map = harden_edge_map_dn(edge_map, harden)
    if edge_power != 1.0:
        edge_map = np.power(np.clip(edge_map, 0.0, 1.0), float(edge_power))
    if edge_gain != 1.0:
        edge_map = np.clip(edge_map * float(edge_gain), 0.0, 1.0)
    if residual_mix > 0.0:
        diff = np.abs(sota_dn - edgekd_dn)
        scale = float(np.percentile(diff, 90.0)) + 1e-6
        disagree = np.clip(diff / scale, 0.0, 1.0)
        if residual_power != 1.0:
            disagree = np.power(disagree, float(residual_power))
        edge_map = (1.0 - residual_mix) * edge_map + residual_mix * np.maximum(
            edge_map, disagree
        )
        edge_map = np.clip(edge_map, 0.0, 1.0)
    w = float(np.clip(edge_weight, 0.0, 1.0))
    edge_eff = w * edge_map
    blended = edge_eff * edgekd_dn + (1.0 - edge_eff) * sota_dn
    return blended.astype(np.float32), edge_map.astype(np.float32)


@torch.no_grad()
def denoise_ensemble(
    adaptive_models: AdaptiveModels,
    dual_model: torch.nn.Module,
    dual_frames: int,
    frames: np.memmap,
    target_index: int,
    fps: float,
    exposure_ms: float,
    device: torch.device,
    tile_size: int = 256,
    temperature: float = 8.0,
    flat_bias: float = 0.0,
    guide: str = "dual",
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    adaptive_dn, adaptive_meta = denoise_adaptive(
        adaptive_models,
        frames,
        target_index,
        fps,
        exposure_ms,
        device,
        tile_size=tile_size,
    )
    dual_dn = denoise_frame(
        dual_model,
        frames,
        target_index,
        exposure_ms,
        device,
        input_frames=dual_frames,
        tile=tile_size,
    )
    if guide == "adaptive":
        guide_dn = adaptive_dn
    elif guide == "input":
        guide_dn = decode_mono10(frames[target_index])
    elif guide == "mean":
        guide_dn = 0.5 * (adaptive_dn + dual_dn)
    else:
        guide_dn = dual_dn

    blended, edge_map = blend_adaptive_dual(
        adaptive_dn,
        dual_dn,
        guide_dn,
        temperature=temperature,
        flat_bias=flat_bias,
    )
    meta: dict[str, float | int | bool | str] = {
        **adaptive_meta,
        "route": f"ensemble({adaptive_meta.get('route', 'adaptive')}+dual)",
        "used_frames": int(adaptive_meta.get("used_frames", dual_frames)),
        "edge_map_mean": float(np.mean(edge_map)),
        "flat_bias": float(flat_bias),
        "guide": guide,
    }
    return blended, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--checkpoint-1f", type=Path, required=True)
    parser.add_argument("--checkpoint-4f", type=Path, required=True)
    parser.add_argument("--checkpoint-16f", type=Path, required=True)
    parser.add_argument("--checkpoint-dual", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nafnet_denoise/inference_ensemble"),
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--flat-bias", type=float, default=0.0)
    parser.add_argument(
        "--guide",
        choices=("dual", "adaptive", "input", "mean"),
        default="dual",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptive_models = load_adaptive_models(
        args.checkpoint_1f,
        args.checkpoint_4f,
        args.checkpoint_16f,
        device,
    )
    dual_model, dual_frames, _ = load_model(args.checkpoint_dual, None, device)
    width, height, fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, width, height)
    target_index = frames.shape[0] // 2 if args.frame_index < 0 else args.frame_index

    output, metadata = denoise_ensemble(
        adaptive_models,
        dual_model,
        dual_frames,
        frames,
        target_index,
        fps,
        exposure_ms,
        device,
        tile_size=args.tile_size,
        temperature=args.temperature,
        flat_bias=args.flat_bias,
        guide=args.guide,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_input.png",
        decode_mono10(frames[target_index]),
    )
    save_mono10_png(
        args.output_dir / f"{stem}_frame{target_index:03d}_ensemble.png",
        output,
    )
    print(f"Saved ensemble to {args.output_dir}: {metadata}")


if __name__ == "__main__":
    main()
