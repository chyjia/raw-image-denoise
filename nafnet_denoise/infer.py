"""Run NAFNet inference on Mono10 RAW frame bursts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import (
    DEFAULT_DARK_VARIANCE_PER_S,
    DEFAULT_INPUT_FRAMES,
    center_frame_index,
    decode_mono10,
    frames_to_model_input,
    gather_frame_indices,
    memmap_frames,
    model_output_to_raw,
    parse_exposure_ms,
    parse_geometry,
)
from .nafnet import build_nafnet


def model_uses_sigma(model: torch.nn.Module, input_frames: int) -> bool:
    use_sigma = getattr(model, "use_sigma", None)
    if use_sigma is not None:
        return bool(use_sigma)
    intro = getattr(model, "intro", None)
    if intro is None or not hasattr(intro, "in_channels"):
        return False
    return int(intro.in_channels) == input_frames + 2


def load_model(
    checkpoint: Path,
    width: int | None,
    device: torch.device,
    input_frames: int | None = None,
) -> tuple[torch.nn.Module, int, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args: dict = {}
    if isinstance(payload, dict) and "model" in payload:
        saved_args = payload.get("args", {}) or {}
        if width is None:
            width = int(saved_args.get("width", 32))
        if input_frames is None:
            input_frames = saved_args.get("input_frames")
        state = payload["model"]
    else:
        if input_frames is None:
            input_frames = DEFAULT_INPUT_FRAMES
        if width is None:
            width = 32
        state = payload

    arch = str(saved_args.get("arch", "nafnet"))
    if any(key.startswith("restormer.") for key in state):
        arch = "stacked_restormer"
    if "ending_flat.weight" in state:
        arch = "dual_head"

    use_nonlocal = bool(saved_args.get("use_nonlocal", False))
    if any(key.startswith("bottleneck_nl.") for key in state):
        use_nonlocal = True
    nonlocal_count = int(saved_args.get("nonlocal_count", 1))
    if use_nonlocal and isinstance(state.get("bottleneck_nl"), dict) is False:
        # Count Sequential children from state keys like bottleneck_nl.0.gamma
        indices = {
            int(key.split(".")[1])
            for key in state
            if key.startswith("bottleneck_nl.") and key.split(".")[1].isdigit()
        }
        if indices:
            nonlocal_count = max(indices) + 1
    learnable_mask = bool(saved_args.get("learnable_mask", False))
    if any(key.startswith("mask_head.") for key in state):
        learnable_mask = True
    use_deformable = bool(saved_args.get("use_deformable", False))
    if any(key.startswith("deform_align.") for key in state):
        use_deformable = True
    deform_feat_width = int(saved_args.get("deform_feat_width", 16))
    deform_max_flow = float(saved_args.get("deform_max_flow", 2.0))
    use_burst_merge = bool(saved_args.get("use_burst_merge", False))
    if any(key.startswith("burst_merge.") for key in state):
        use_burst_merge = True
    merge_feat_width = int(saved_args.get("merge_feat_width", 16))
    region_towers = int(saved_args.get("region_towers", 0))
    if any(key.startswith("tower_flat.") or key.startswith("tower_edge.") for key in state):
        # Infer tower depth from Sequential indices if args missing.
        tower_idx = {
            int(key.split(".")[1])
            for key in state
            if key.startswith("tower_flat.") and key.split(".")[1].isdigit()
        }
        if tower_idx:
            region_towers = max(region_towers, max(tower_idx) + 1)
    fusion_edge_harden = float(saved_args.get("fusion_edge_harden", 0.0))

    intro_weight = state.get("intro.weight")
    use_sigma = bool(saved_args.get("use_sigma", False))
    if input_frames is None:
        input_frames = saved_args.get("input_frames")
    if input_frames is None:
        if intro_weight is not None:
            extras = 2 if use_sigma else 1
            input_frames = int(intro_weight.shape[1]) - extras
        else:
            input_frames = DEFAULT_INPUT_FRAMES
    input_frames = int(input_frames)
    if intro_weight is not None:
        use_sigma = int(intro_weight.shape[1]) == input_frames + 2
    elif "restormer.patch_embed.proj.weight" in state:
        in_ch = int(state["restormer.patch_embed.proj.weight"].shape[1])
        use_sigma = in_ch == input_frames + 2

    if arch == "stacked_restormer":
        from .stacked_restormer import build_stacked_restormer

        def _parse_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                return tuple(int(item) for item in value)
            return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())

        model = build_stacked_restormer(
            input_frames=input_frames,
            use_sigma=use_sigma,
            dim=int(saved_args.get("dim", 48)),
            num_blocks=_parse_tuple(saved_args.get("num_blocks"), (2, 3, 3, 4)),
            num_refinement_blocks=int(saved_args.get("num_refinement_blocks", 2)),
            use_checkpoint=False,
        )
    elif arch == "dual_head":
        from .dual_head_nafnet import build_dual_head_nafnet

        model = build_dual_head_nafnet(
            width=width,
            input_frames=input_frames,
            use_sigma=use_sigma,
            use_nonlocal=use_nonlocal,
            nonlocal_count=nonlocal_count,
            learnable_mask=learnable_mask,
            use_deformable=use_deformable,
            deform_feat_width=deform_feat_width,
            deform_max_flow=deform_max_flow,
            use_burst_merge=use_burst_merge,
            merge_feat_width=merge_feat_width,
            region_towers=region_towers,
            fusion_edge_harden=fusion_edge_harden,
            use_sigma_film=bool(saved_args.get("use_sigma_film", False))
            or any(
                key.startswith("sigma_film.") and not key.startswith("sigma_film_flat.")
                for key in state
            ),
            use_sigma_film_flat=bool(saved_args.get("use_sigma_film_flat", False))
            or any(key.startswith("sigma_film_flat.") for key in state),
            use_lap_edge=bool(saved_args.get("use_lap_edge", False))
            or any(
                key.startswith("lap_to_edge.") and not key.startswith("lap_to_edge_ms.")
                for key in state
            ),
            use_lap_edge_ms=bool(saved_args.get("use_lap_edge_ms", False))
            or any(key.startswith("lap_to_edge_ms.") for key in state),
        )
    else:
        model = build_nafnet(
            width=width,
            input_frames=input_frames,
            use_sigma=use_sigma,
            use_nonlocal=use_nonlocal,
            nonlocal_count=nonlocal_count,
        )
    model.load_state_dict(state, strict=False)
    model.use_sigma = use_sigma
    model.use_sigma_film = bool(getattr(model, "use_sigma_film", False)) or bool(
        saved_args.get("use_sigma_film", False)
    )
    model.wiener_front_end = bool(saved_args.get("wiener_front_end", False))
    model.wiener_merge_frames = int(saved_args.get("wiener_merge_frames", input_frames))
    model.wiener_tile = int(saved_args.get("wiener_tile", 32))
    model.wiener_overlap = int(saved_args.get("wiener_overlap", 16))
    model.wiener_c_factor = float(saved_args.get("wiener_c_factor", 8.0))
    model.wiener_spatial = bool(saved_args.get("wiener_spatial", False))
    model.wiener_spatial_c_factor = saved_args.get("wiener_spatial_c_factor", None)
    model.wiener_spatial_adaptive = bool(
        saved_args.get("wiener_spatial_adaptive", False)
    )
    model.wiener_spatial_flat_c_mult = float(
        saved_args.get("wiener_spatial_flat_c_mult", 2.5)
    )
    model.wiener_spatial_dark_boost = float(
        saved_args.get("wiener_spatial_dark_boost", 0.5)
    )
    model.wiener_spatial_edge_c_mult = float(
        saved_args.get("wiener_spatial_edge_c_mult", 1.0)
    )
    model.wiener_spatial_mask_harden = float(
        saved_args.get("wiener_spatial_mask_harden", 0.0)
    )
    model.wiener_spatial_freq_gamma = float(
        saved_args.get("wiener_spatial_freq_gamma", 0.0)
    )
    model.wiener_fe_schedule = saved_args.get("wiener_fe_schedule")
    postmerge_path = saved_args.get("postmerge_noise_calib")
    model.postmerge_calib = None
    if postmerge_path:
        from .postmerge_noise import load_postmerge_calib

        calib_path = Path(postmerge_path)
        if not calib_path.is_absolute():
            # Resolve relative to repo root (parent of nafnet_denoise/).
            repo = Path(__file__).resolve().parents[1]
            candidate = repo / calib_path
            if candidate.exists():
                calib_path = candidate
        if calib_path.exists():
            model.postmerge_calib = load_postmerge_calib(calib_path)
    model.measured_fe_sigma = bool(saved_args.get("measured_fe_sigma", False))
    return model.eval().to(device), input_frames, width


def denoise_from_dn_frames(
    model: torch.nn.Module,
    dn_frames: list[np.ndarray],
    exposure_ms: float,
    device: torch.device,
    tile: int = 512,
    overlap: int = 32,
    residual_model: torch.nn.Module | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Tile-wise denoise from a list of DN frames (already ordered / aligned)."""
    if getattr(model, "wiener_front_end", False):
        from .wiener_merge import merge_burst_dn

        intro = getattr(model, "intro", None)
        if intro is not None and hasattr(intro, "in_channels"):
            extras = 2 if getattr(model, "use_sigma", False) else 1
            model_frames = int(intro.in_channels) - extras
        else:
            model_frames = len(dn_frames)
        schedule = getattr(model, "wiener_fe_schedule", None)
        tile_size = int(getattr(model, "wiener_tile", 32))
        overlap_w = int(getattr(model, "wiener_overlap", 16))
        c_factor = float(getattr(model, "wiener_c_factor", 8.0))
        if schedule in ("fps_sigma", "gated"):
            # Temporal merge first, then fps/σ-gated spatial (P0-a).
            temporal, _meta = merge_burst_dn(
                dn_frames,
                reference_index=center_frame_index(len(dn_frames)),
                align=True,
                tile_size=tile_size,
                overlap=overlap_w,
                c_factor=c_factor,
                spatial_wiener=False,
            )
            from .fe_schedule import gated_spatial_from_temporal

            fps_use = fps if fps is not None else getattr(model, "infer_fps", None)
            merged, params, fe_sigma = gated_spatial_from_temporal(
                temporal,
                fps=fps_use,
                n_frames_averaged=len(dn_frames),
                tile_size=tile_size,
                overlap=overlap_w,
            )
            model.last_fe_schedule = params.name
            model.last_fe_sigma_dn = float(fe_sigma)
        else:
            merged, _meta = merge_burst_dn(
                dn_frames,
                reference_index=center_frame_index(len(dn_frames)),
                align=True,
                tile_size=tile_size,
                overlap=overlap_w,
                c_factor=c_factor,
                spatial_wiener=bool(getattr(model, "wiener_spatial", False)),
                spatial_c_factor=getattr(model, "wiener_spatial_c_factor", None),
                spatial_adaptive=bool(getattr(model, "wiener_spatial_adaptive", False)),
                spatial_flat_c_mult=float(
                    getattr(model, "wiener_spatial_flat_c_mult", 2.5)
                ),
                spatial_dark_boost=float(
                    getattr(model, "wiener_spatial_dark_boost", 0.5)
                ),
                spatial_edge_c_mult=float(
                    getattr(model, "wiener_spatial_edge_c_mult", 1.0)
                ),
                spatial_mask_harden=float(
                    getattr(model, "wiener_spatial_mask_harden", 0.0)
                ),
                spatial_freq_gamma=float(
                    getattr(model, "wiener_spatial_freq_gamma", 0.0)
                ),
            )
        dn_frames = [merged.copy() for _ in range(max(model_frames, 1))]

    input_frames = len(dn_frames)
    center = center_frame_index(input_frames)
    use_sigma = model_uses_sigma(model, input_frames)
    height, width = dn_frames[center].shape
    output = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    step = tile - overlap

    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile)
        y0 = max(0, y1 - tile)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile)
            x0 = max(0, x1 - tile)
            patches = [frame[y0:y1, x0:x1] for frame in dn_frames]
            postmerge_calib = (
                getattr(model, "postmerge_calib", None)
                if getattr(model, "wiener_front_end", False)
                else None
            )
            # Flag alone: caller may already have merged (wiener_front_end off).
            measured_fe = bool(getattr(model, "measured_fe_sigma", False))
            # Match real-train / calib PTC feature (dark current term).
            use_dark = postmerge_calib is not None or measured_fe
            model_input = frames_to_model_input(
                patches,
                exposure_ms,
                reference_index=center,
                use_sigma=use_sigma,
                dark_variance_per_s=(
                    DEFAULT_DARK_VARIANCE_PER_S if use_dark else 0.0
                ),
                postmerge_calib=postmerge_calib,
                measured_fe_sigma=measured_fe,
            )
            tensor = torch.from_numpy(model_input[None]).float().to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                pred_vst = model(tensor).float()
                if residual_model is not None:
                    from .detail_residual import residual_conditioning_input

                    cond = residual_conditioning_input(tensor, pred_vst, input_frames)
                    pred_vst = pred_vst + residual_model(cond).float()
                pred_np = pred_vst.cpu().numpy()[0, 0]
            pred_dn = model_output_to_raw(pred_np)
            window = np.outer(np.hanning(y1 - y0), np.hanning(x1 - x0)).astype(np.float32)
            output[y0:y1, x0:x1] += pred_dn * window
            weights[y0:y1, x0:x1] += window

    return output / np.maximum(weights, 1e-6)


def denoise_frame(
    model: torch.nn.Module,
    frames: np.memmap,
    target_index: int,
    exposure_ms: float,
    device: torch.device,
    input_frames: int = DEFAULT_INPUT_FRAMES,
    tile: int = 512,
    overlap: int = 32,
    residual_model: torch.nn.Module | None = None,
    fps: float | None = None,
) -> np.ndarray:
    gather_frames = input_frames
    if getattr(model, "wiener_front_end", False):
        gather_frames = int(getattr(model, "wiener_merge_frames", input_frames))
    frame_indices = gather_frame_indices(target_index, frames.shape[0], gather_frames)
    decoded = [decode_mono10(frames[index]) for index in frame_indices]
    return denoise_from_dn_frames(
        model,
        decoded,
        exposure_ms,
        device,
        tile=tile,
        overlap=overlap,
        residual_model=residual_model,
        fps=fps,
    )


def save_mono10_png(path: Path, image: np.ndarray) -> None:
    output = np.clip(np.rint(image), 0, 1023).astype(np.uint16) << 6
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("nafnet_denoise/inference"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--input-frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--tile", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_frames, width = load_model(
        args.checkpoint,
        args.width,
        device,
        input_frames=args.input_frames,
    )

    width, height, _fps = parse_geometry(args.raw_path.name)
    exposure_ms = parse_exposure_ms(args.raw_path.name)
    frames = memmap_frames(args.raw_path, width, height)
    if args.frame_index < 0 or args.frame_index >= frames.shape[0]:
        raise SystemExit(f"frame-index out of range: 0-{frames.shape[0] - 1}")

    input_dn = decode_mono10(frames[args.frame_index])
    output_dn = denoise_frame(
        model,
        frames,
        args.frame_index,
        exposure_ms,
        device,
        input_frames=input_frames,
        tile=args.tile,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.raw_path.stem
    save_mono10_png(args.output_dir / f"{stem}_frame{args.frame_index:03d}_input.png", input_dn)
    save_mono10_png(args.output_dir / f"{stem}_frame{args.frame_index:03d}_denoised.png", output_dn)
    print(
        f"Saved denoised frame to {args.output_dir} "
        f"(input_frames={input_frames}, width={width})"
    )


if __name__ == "__main__":
    main()
