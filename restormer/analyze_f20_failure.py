"""Diagnose why Restormer fails on the f20 validation clip."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import common
from evaluate import leave_one_out_reference, restormer_infer, vst_bm3d
from restormer import Restormer

VAL_DIR = Path("/mnt/d/denoise/素材/训练素材/不参加训练，用来验证训练模型效果")
MANIFEST = Path("cache/train_manifest.json")
FRAME = 10


def spatial_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def roi_stats(image: np.ndarray) -> dict:
    y0, x0, h, w = common.central_roi_bounds(*image.shape, 0.4)
    roi = image[y0 : y0 + h, x0 : x0 + w]
    return {
        "roi_mean": float(roi.mean()),
        "roi_std": float(roi.std()),
        "global_mean": float(image.mean()),
    }


def per_frame_roi_series(data: np.memmap) -> np.ndarray:
    y0, x0, h, w = common.central_roi_bounds(data.shape[1], data.shape[2], 0.4)
    means = []
    for i in range(data.shape[0]):
        frame = common.decode_frame(data[i])
        means.append(float(frame[y0 : y0 + h, x0 : x0 + w].mean()))
    return np.array(means, dtype=np.float64)


def training_brightness_distribution(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref_means = [e["ref_mean_dn"] for e in manifest["entries"]]
    flicker = [e.get("frame_roi_mean_std", 0.0) for e in manifest["entries"]]
    f20_entries = [e for e in manifest["entries"] if abs(e["fps"] - 20.0) < 0.5]
    f20_ref_means = [e["ref_mean_dn"] for e in f20_entries]
    return {
        "count": len(manifest["entries"]),
        "f20_count": len(f20_entries),
        "ref_mean_min": float(np.min(ref_means)),
        "ref_mean_max": float(np.max(ref_means)),
        "ref_mean_median": float(np.median(ref_means)),
        "f20_ref_mean_min": float(np.min(f20_ref_means)) if f20_ref_means else None,
        "f20_ref_mean_max": float(np.max(f20_ref_means)) if f20_ref_means else None,
        "f20_ref_mean_median": float(np.median(f20_ref_means)) if f20_ref_means else None,
        "flicker_std_median": float(np.median(flicker)),
        "flicker_std_max": float(np.max(flicker)),
    }


def load_model(ckpt_path: Path, device: torch.device) -> tuple[Restormer, int]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    inp_channels = 3 if state.get("args", {}).get("brightness_condition", False) else 2
    model = Restormer(inp_channels=inp_channels, out_channels=1, dim=48, use_checkpoint=False).to(
        device
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model, inp_channels


def analyze_clip(path: Path, model_v1: Restormer, model_v2: Restormer, device: torch.device):
    meta = common.parse_meta(path)
    data = common.open_raw(path, meta)
    noisy = common.decode_frame(data[FRAME])
    reference = leave_one_out_reference(data, FRAME, 64, 0.10)

    exposure_cond = common.exposure_to_condition(meta.exposure_ms)
    brightness_cond = common.brightness_to_condition(common.central_roi_mean(noisy))

    v1_out = restormer_infer(model_v1, noisy, exposure_cond, device, brightness_condition=False)
    v2_out = restormer_infer(
        model_v2, noisy, exposure_cond, device, brightness_condition=brightness_cond
    )
    bm3d_out = vst_bm3d(noisy)

    roi_series = per_frame_roi_series(data)
    frame10_delta_from_median = float(noisy.mean() - np.median(roi_series))

    # Spatial error decomposition on central ROI vs edges
    h, w = noisy.shape
    y0, x0, rh, rw = common.central_roi_bounds(h, w, 0.4)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0 : y0 + rh, x0 : x0 + rw] = True

    def roi_vs_edge_mae(pred, ref):
        roi_mae = spatial_mae(pred[mask], ref[mask])
        edge_mae = spatial_mae(pred[~mask], ref[~mask])
        return roi_mae, edge_mae

    report = {
        "file": path.name,
        "fps": meta.fps,
        "exposure_ms": meta.exposure_ms,
        "exposure_condition": exposure_cond,
        "brightness_condition": brightness_cond,
        "frame_count": int(data.shape[0]),
        "noisy": roi_stats(noisy),
        "reference": roi_stats(reference),
        "input_minus_ref_roi_mean": float(noisy[y0 : y0 + rh, x0 : x0 + rw].mean()
                                        - reference[y0 : y0 + rh, x0 : x0 + rw].mean()),
        "flicker_roi_std_all_frames": float(roi_series.std()),
        "flicker_roi_std_sampled80": float(
            common.frame_roi_mean_series(data, 80)[1]
        ),
        "frame10_roi_mean_vs_clip_median": frame10_delta_from_median,
    }

    for name, pred in [("vst_bm3d", bm3d_out), ("v1", v1_out), ("v2", v2_out)]:
        roi_mae, edge_mae = roi_vs_edge_mae(pred, reference)
        report[name] = {
            **roi_stats(pred),
            "mae_all": spatial_mae(pred, reference),
            "mae_roi": roi_mae,
            "mae_edge": edge_mae,
            "roi_mean_error_vs_ref": float(
                pred[y0 : y0 + rh, x0 : x0 + rw].mean()
                - reference[y0 : y0 + rh, x0 : x0 + rw].mean()
            ),
        }

    # Check if v2 is collapsing toward a training-set typical brightness
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train_ref_means = np.array([e["ref_mean_dn"] for e in manifest["entries"]])
    report["train_ref_mean_median"] = float(np.median(train_ref_means))
    report["v2_global_mean_minus_train_median"] = float(v2_out.mean() - np.median(train_ref_means))

    # Gradient magnitude ratio (oversmoothing detector)
    def grad_energy(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        return float(np.mean(np.sqrt(gx * gx + gy * gy)))

    report["grad_energy"] = {
        "input": grad_energy(noisy),
        "reference": grad_energy(reference),
        "vst_bm3d": grad_energy(bm3d_out),
        "v1": grad_energy(v1_out),
        "v2": grad_energy(v2_out),
    }

    return report


def main() -> None:
    device = torch.device("cuda")
    model_v1, _ = load_model(Path("checkpoints/best.pt"), device)
    model_v2, _ = load_model(Path("checkpoints_v2/best.pt"), device)

    print("=== Training distribution ===")
    print(json.dumps(training_brightness_distribution(MANIFEST), indent=2))

    files = common.list_mono10_files(VAL_DIR, exclude_contains=None)
    for path in files:
        print(f"\n=== {path.name} ===")
        print(json.dumps(analyze_clip(path, model_v1, model_v2, device), indent=2))


if __name__ == "__main__":
    main()
