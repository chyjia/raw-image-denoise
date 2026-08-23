"""Evaluate the trained Restormer against VST + BM3D on held-out videos.

For each Mono10 video it exports frame 10 and compares:
* input                  : raw noisy frame;
* temporal_trimmed_mean  : leave-one-out multi-frame reference (proxy ground truth);
* vst_bm3d               : PTC-guided VST + BM3D;
* restormer              : the trained conditional Restormer.

Metrics are computed against the leave-one-out temporal reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import bm3d
import cv2
import numpy as np
import torch

import common
from restormer import Restormer


def leave_one_out_reference(
    data: np.memmap, target_index: int, max_frames: int, trim_fraction: float
) -> np.ndarray:
    frame_count, height, width = data.shape
    indices = list(np.linspace(0, frame_count - 1, min(frame_count, max_frames), dtype=np.int64))
    indices = [int(i) for i in indices if int(i) != target_index]

    roi_h, roi_w = int(height * 0.4), int(width * 0.4)
    y0, x0 = (height - roi_h) // 2, (width - roi_w) // 2
    frames, roi_means = [], []
    for index in indices:
        frame = common.decode_frame(data[index])
        frames.append(frame)
        roi_means.append(float(frame[y0 : y0 + roi_h, x0 : x0 + roi_w].mean()))
    series_mean = float(np.mean(roi_means))
    aligned = np.stack([f - m + series_mean for f, m in zip(frames, roi_means)], axis=0)
    trim = int(len(frames) * trim_fraction)
    if trim > 0 and len(frames) - 2 * trim >= 3:
        aligned.sort(axis=0)
        aligned = aligned[trim:-trim]
    return np.clip(aligned.mean(axis=0), 0.0, common.RAW_MAX).astype(np.float32)


def brightness_correct_to_input(
    denoised: np.ndarray, reference_input: np.ndarray, roi_fraction: float = 0.4
) -> np.ndarray:
    """Shift denoised output so its central flat ROI mean matches the noisy input."""
    h, w = denoised.shape
    roi_h, roi_w = int(h * roi_fraction), int(w * roi_fraction)
    y0, x0 = (h - roi_h) // 2, (w - roi_w) // 2
    delta = float(reference_input[y0 : y0 + roi_h, x0 : x0 + roi_w].mean()) - float(
        denoised[y0 : y0 + roi_h, x0 : x0 + roi_w].mean()
    )
    return np.clip(denoised + delta, 0.0, common.RAW_MAX).astype(np.float32)


def vst_bm3d(image_dn: np.ndarray) -> np.ndarray:
    transformed = common.vst_forward(image_dn).astype(np.float64)
    denoised = bm3d.bm3d(transformed, sigma_psd=1.0)
    return common.vst_inverse(denoised).astype(np.float32)


def third_channel_condition(
    image_dn: np.ndarray,
    exposure_ms: float,
    ckpt_args: dict,
    priors: dict[float, float],
) -> float | bool:
    """Return the 3rd input channel value, or False when unused."""
    if not ckpt_args.get("brightness_condition", False):
        return False
    roi_mean = common.central_roi_mean(image_dn)
    if ckpt_args.get("brightness_offset", False):
        return common.brightness_offset_to_condition(roi_mean, exposure_ms, priors)
    return common.brightness_to_condition(roi_mean)


@torch.no_grad()
def restormer_infer(
    model: Restormer,
    image_dn: np.ndarray,
    exposure_condition: float,
    device: torch.device,
    brightness_condition: float | None = None,
    tile: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    normalized = common.raw_to_model_input(image_dn).astype(np.float32)
    height, width = normalized.shape
    step = tile - overlap
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    window = np.maximum(window, 1e-3)
    accumulation = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)

    if brightness_condition is None:
        brightness_condition = common.brightness_to_condition(
            common.central_roi_mean(image_dn)
        )

    ys = list(range(0, max(1, height - tile + 1), step))
    xs = list(range(0, max(1, width - tile + 1), step))
    if ys[-1] != height - tile:
        ys.append(height - tile)
    if xs[-1] != width - tile:
        xs.append(width - tile)

    for y0 in ys:
        for x0 in xs:
            patch = normalized[y0 : y0 + tile, x0 : x0 + tile]
            exposure_channel = np.full_like(patch, exposure_condition)
            channels = [patch, exposure_channel]
            if brightness_condition is not False:
                brightness_channel = np.full_like(patch, brightness_condition)
                channels.append(brightness_channel)
            tensor = torch.from_numpy(np.stack(channels, axis=0))[None].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(tensor)
            out_np = out.float().cpu().numpy()[0, 0]
            accumulation[y0 : y0 + tile, x0 : x0 + tile] += out_np * window
            weights[y0 : y0 + tile, x0 : x0 + tile] += window

    blended = (accumulation / np.maximum(weights, 1e-8)).astype(np.float32)
    return common.model_output_to_raw(blended).astype(np.float32)


def highpass_noise_sigma(image: np.ndarray, y0: int, x0: int, h: int, w: int) -> float:
    roi = image[y0 : y0 + h, x0 : x0 + w]
    lowpass = cv2.GaussianBlur(roi, (0, 0), 1.2)
    residual = roi - lowpass
    return float(np.median(np.abs(residual - np.median(residual))) / 0.6745)


def gradient_mae(image: np.ndarray, reference: np.ndarray) -> float:
    gi = cv2.magnitude(cv2.Sobel(image, cv2.CV_32F, 1, 0), cv2.Sobel(image, cv2.CV_32F, 0, 1))
    gr = cv2.magnitude(cv2.Sobel(reference, cv2.CV_32F, 1, 0), cv2.Sobel(reference, cv2.CV_32F, 0, 1))
    return float(np.mean(np.abs(gi - gr)))


def save_preview(path: Path, images: list[np.ndarray], labels: list[str]) -> None:
    tiles = []
    for image, label in zip(images, labels):
        vis = cv2.cvtColor(np.clip(image / common.RAW_MAX * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
        tiles.append(vis)
    cv2.imwrite(str(path), np.hstack(tiles))


def save_mono10(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), (np.clip(np.rint(image), 0, common.RAW_MAX).astype(np.uint16) << 6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("restormer/checkpoints/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("restormer/eval_results"))
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument(
        "--exposure-priors",
        type=Path,
        default=Path("cache/exposure_brightness_priors.json"),
    )
    args = parser.parse_args()

    priors = common.load_exposure_brightness_priors(args.exposure_priors)

    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = state.get("args", {})
    inp_channels = 3 if ckpt_args.get("brightness_condition", False) else 2
    model = Restormer(
        inp_channels=inp_channels, out_channels=1, dim=args.dim, use_checkpoint=False
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(
        f"Loaded {args.checkpoint} (epoch {state.get('epoch', '?')}, "
        f"inp_channels={inp_channels})"
    )

    files = common.list_mono10_files(args.input_dir, exclude_contains=None)
    if not files:
        raise SystemExit(f"No Mono10 files under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    scene_summaries: dict[str, list[dict]] = {}
    for path in files:
        meta = common.parse_meta(path)
        data = common.open_raw(path, meta)
        if data.shape[0] <= args.frame_index:
            print(f"skip {path.name}: only {data.shape[0]} frames")
            continue
        exposure_condition = common.exposure_to_condition(meta.exposure_ms)
        print(f"{path.name}: exposure {meta.exposure_ms:.1f} ms, frame {args.frame_index}")

        noisy = common.decode_frame(data[args.frame_index])
        scene_type = common.classify_scene_type(noisy)
        print(f"  scene_type={scene_type}")
        reference = leave_one_out_reference(data, args.frame_index, args.max_frames, 0.10)
        bm3d_out = vst_bm3d(noisy)
        aux_condition = third_channel_condition(noisy, meta.exposure_ms, ckpt_args, priors)
        model_out = restormer_infer(
            model,
            noisy,
            exposure_condition,
            device,
            brightness_condition=aux_condition,
        )
        model_bc = brightness_correct_to_input(model_out, noisy)

        outputs = {
            "input": noisy,
            "temporal_trimmed_mean": reference,
            "vst_bm3d": bm3d_out,
            "restormer": model_out,
            "restormer_bc": model_bc,
        }
        out_dir = args.output_dir / "__".join(path.relative_to(args.input_dir).with_suffix("").parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image in outputs.items():
            save_mono10(out_dir / f"frame_{args.frame_index:03d}_{name}.png", image)
        save_preview(
            out_dir / f"frame_{args.frame_index:03d}_preview.png",
            list(outputs.values()),
            ["Input", "Temporal mean", "VST+BM3D", "Restormer", "Restormer+BC"],
        )

        h, w = noisy.shape
        roi_h, roi_w = int(h * 0.4), int(w * 0.4)
        y0, x0 = (h - roi_h) // 2, (w - roi_w) // 2
        for name, image in outputs.items():
            row = {
                "file": path.name,
                "scene_type": scene_type,
                "exposure_ms": f"{meta.exposure_ms:.1f}",
                "method": name,
                "roi_mean_dn": f"{float(image[y0:y0+roi_h, x0:x0+roi_w].mean()):.5f}",
                "roi_highpass_noise_sigma_dn": f"{highpass_noise_sigma(image, y0, x0, roi_h, roi_w):.5f}",
                "mae_to_reference_dn": f"{float(np.mean(np.abs(image - reference))):.5f}",
                "gradient_mae_to_reference": f"{gradient_mae(image, reference):.5f}",
            }
            rows.append(row)
            if name in {"vst_bm3d", "restormer"}:
                scene_summaries.setdefault(scene_type, []).append(row)

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {metrics_path}")

    summary_path = args.output_dir / "metrics_by_scene.csv"
    summary_rows = []
    for scene_type, scene_rows in sorted(scene_summaries.items()):
        for method in ("vst_bm3d", "restormer"):
            method_rows = [r for r in scene_rows if r["method"] == method]
            if not method_rows:
                continue
            summary_rows.append(
                {
                    "scene_type": scene_type,
                    "method": method,
                    "clip_count": str(len(method_rows)),
                    "mae_to_reference_dn_mean": f"{np.mean([float(r['mae_to_reference_dn']) for r in method_rows]):.5f}",
                    "roi_highpass_noise_sigma_dn_mean": f"{np.mean([float(r['roi_highpass_noise_sigma_dn']) for r in method_rows]):.5f}",
                    "gradient_mae_to_reference_mean": f"{np.mean([float(r['gradient_mae_to_reference']) for r in method_rows]):.5f}",
                }
            )
            print(
                f"  [{scene_type}] {method}: "
                f"MAE={summary_rows[-1]['mae_to_reference_dn_mean']} "
                f"flat_noise={summary_rows[-1]['roi_highpass_noise_sigma_dn_mean']}"
            )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved scene summary to {summary_path}")


if __name__ == "__main__":
    main()
