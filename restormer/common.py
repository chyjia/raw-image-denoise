"""Shared helpers for the Restormer denoising pipeline.

The pipeline works in the generalized-Anscombe VST domain using the current
Mono10 grayscale-chart PTC. Training happens on VST values linearly normalized to
roughly [0, 1] so a network sees an approximately homoscedastic signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Current Mono10 PTC: variance_dn2 = PTC_SLOPE * mean_dn + PTC_INTERCEPT
PTC_SLOPE = 0.10618515
PTC_INTERCEPT = 2.00145577
RAW_MAX = 1023.0

WIDTH_HEIGHT_FPS_RE = re.compile(
    r"_w(?P<width>\d+)_h(?P<height>\d+)_pMono10_f(?P<fps>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
EXPOSURE_MS_RE = re.compile(r"exposure_(?P<ms>\d+)ms", re.IGNORECASE)


@dataclass(frozen=True)
class RawMeta:
    width: int
    height: int
    fps: float
    exposure_ms: float


def parse_meta(path: Path) -> RawMeta:
    match = WIDTH_HEIGHT_FPS_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse Mono10 geometry from {path.name}")
    width = int(match.group("width"))
    height = int(match.group("height"))
    fps = float(match.group("fps"))

    exposure_match = EXPOSURE_MS_RE.search(path.name)
    if exposure_match:
        exposure_ms = float(exposure_match.group("ms"))
    elif fps > 0:
        # No explicit exposure: use the frame interval as a proxy.
        exposure_ms = 1000.0 / fps
    else:
        raise ValueError(f"Cannot derive exposure for {path.name}")
    return RawMeta(width=width, height=height, fps=fps, exposure_ms=exposure_ms)


def list_mono10_files(root: Path, exclude_contains: str | None) -> list[Path]:
    files = sorted(root.rglob("*_pMono10_f*.raw"))
    if exclude_contains:
        files = [p for p in files if exclude_contains not in str(p)]
    return files


def open_raw(path: Path, meta: RawMeta) -> np.memmap:
    bytes_per_frame = meta.width * meta.height * 2
    file_size = path.stat().st_size
    if file_size % bytes_per_frame:
        raise ValueError(f"{path.name} is not an integer number of Mono10 frames.")
    frame_count = file_size // bytes_per_frame
    return np.memmap(
        path, mode="r", dtype="<u2", shape=(frame_count, meta.height, meta.width)
    )


def decode_frame(raw_frame: np.ndarray) -> np.ndarray:
    """Decode unpacked low-aligned Mono10 stored as little-endian uint16."""
    return (raw_frame & 0x03FF).astype(np.float32)


def vst_forward(image: np.ndarray) -> np.ndarray:
    radicand = PTC_SLOPE * image + 0.375 * PTC_SLOPE**2 + PTC_INTERCEPT
    return (2.0 / PTC_SLOPE) * np.sqrt(np.maximum(radicand, 1e-8))


def vst_inverse(transformed: np.ndarray) -> np.ndarray:
    image = (
        (PTC_SLOPE / 4.0) * transformed**2
        - 0.375 * PTC_SLOPE
        - PTC_INTERCEPT / PTC_SLOPE
    )
    return np.clip(image, 0.0, RAW_MAX)


_VST_MIN = float(vst_forward(np.array(0.0, dtype=np.float32)))
_VST_MAX = float(vst_forward(np.array(RAW_MAX, dtype=np.float32)))


def vst_normalize(transformed: np.ndarray) -> np.ndarray:
    return (transformed - _VST_MIN) / (_VST_MAX - _VST_MIN)


def vst_denormalize(normalized: np.ndarray) -> np.ndarray:
    return normalized * (_VST_MAX - _VST_MIN) + _VST_MIN


def raw_to_model_input(image_dn: np.ndarray) -> np.ndarray:
    """DN image -> normalized VST domain used by the network."""
    return vst_normalize(vst_forward(image_dn))


def model_output_to_raw(normalized: np.ndarray) -> np.ndarray:
    """Normalized VST output -> DN image."""
    return vst_inverse(vst_denormalize(normalized))


# Exposure is log-scaled so 10 ms and 400 ms map to a compact, well-spread range.
_EXPOSURE_LOG_MIN = float(np.log(5.0))
_EXPOSURE_LOG_MAX = float(np.log(500.0))


def exposure_to_condition(exposure_ms: float) -> float:
    value = (np.log(max(exposure_ms, 1e-3)) - _EXPOSURE_LOG_MIN) / (
        _EXPOSURE_LOG_MAX - _EXPOSURE_LOG_MIN
    )
    return float(np.clip(value, 0.0, 1.0))


# Frame flat-field brightness (central ROI mean DN) for flicker conditioning.
_BRIGHTNESS_LOG_MIN = float(np.log(80.0))
_BRIGHTNESS_LOG_MAX = float(np.log(900.0))


def central_roi_bounds(
    height: int, width: int, roi_fraction: float = 0.4
) -> tuple[int, int, int, int]:
    roi_h = int(height * roi_fraction)
    roi_w = int(width * roi_fraction)
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    return y0, x0, roi_h, roi_w


def central_roi_mean(image_dn: np.ndarray, roi_fraction: float = 0.4) -> float:
    height, width = image_dn.shape
    y0, x0, roi_h, roi_w = central_roi_bounds(height, width, roi_fraction)
    return float(image_dn[y0 : y0 + roi_h, x0 : x0 + roi_w].mean())


def brightness_to_condition(mean_dn: float) -> float:
    """Map central flat ROI mean DN to [0, 1] for an auxiliary input channel."""
    value = (np.log(max(mean_dn, 1.0)) - _BRIGHTNESS_LOG_MIN) / (
        _BRIGHTNESS_LOG_MAX - _BRIGHTNESS_LOG_MIN
    )
    return float(np.clip(value, 0.0, 1.0))


# log(roi_mean) - log(expected_at_exposure); 0 -> channel 0.5 (typical for that exposure).
_OFFSET_LOG_MIN = -1.2
_OFFSET_LOG_MAX = 1.5


def build_exposure_brightness_priors(entries: list[dict]) -> dict[float, float]:
    """Median reference brightness per exposure (ms) from the training manifest."""
    buckets: dict[float, list[float]] = {}
    for entry in entries:
        exposure_ms = round(float(entry["exposure_ms"]), 1)
        buckets.setdefault(exposure_ms, []).append(float(entry["ref_mean_dn"]))
    return {exposure_ms: float(np.median(values)) for exposure_ms, values in buckets.items()}


def load_exposure_brightness_priors(path: Path) -> dict[float, float]:
    import json

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {float(k): float(v) for k, v in data.items()}


def expected_brightness_dn(exposure_ms: float, priors: dict[float, float]) -> float:
    if not priors:
        return 300.0
    keys = np.array(list(priors.keys()), dtype=np.float64)
    nearest = float(keys[np.argmin(np.abs(keys - exposure_ms))])
    return priors[nearest]


def brightness_offset_to_condition(
    roi_mean_dn: float, exposure_ms: float, priors: dict[float, float]
) -> float:
    """Brightness relative to the exposure-specific training prior; 0.5 = typical."""
    expected = expected_brightness_dn(exposure_ms, priors)
    offset = float(np.log(max(roi_mean_dn, 1.0)) - np.log(max(expected, 1.0)))
    value = (offset - _OFFSET_LOG_MIN) / (_OFFSET_LOG_MAX - _OFFSET_LOG_MIN)
    return float(np.clip(value, 0.0, 1.0))


def classify_scene_type(image_dn: np.ndarray, roi_fraction: float = 0.4) -> str:
    """Split validation clips into flat-uniform vs high-dynamic scenes."""
    height, width = image_dn.shape
    y0, x0, roi_h, roi_w = central_roi_bounds(height, width, roi_fraction)
    roi = image_dn[y0 : y0 + roi_h, x0 : x0 + roi_w]
    roi_mean = float(roi.mean())
    global_mean = float(image_dn.mean())
    roi_std = float(roi.std())
    ratio = roi_mean / max(global_mean, 1.0)
    if ratio < 1.15 and roi_std < 120.0:
        return "flat_uniform"
    return "high_dynamic"


def align_reference_to_frame_brightness(
    reference_dn: np.ndarray,
    frame_dn: np.ndarray,
    roi_fraction: float = 0.4,
) -> np.ndarray:
    """Shift the clean reference so its central ROI mean matches the noisy frame."""
    ref_mean = central_roi_mean(reference_dn, roi_fraction)
    frame_mean = central_roi_mean(frame_dn, roi_fraction)
    return np.clip(reference_dn + (frame_mean - ref_mean), 0.0, RAW_MAX).astype(
        np.float32
    )


def frame_roi_mean_series(
    data: np.memmap,
    max_frames: int,
    roi_fraction: float = 0.4,
) -> tuple[list[float], float]:
    """Per-frame central ROI means and their standard deviation (flicker proxy)."""
    frame_count = data.shape[0]
    indices = np.linspace(
        0, frame_count - 1, min(frame_count, max_frames), dtype=np.int64
    )
    height, width = data.shape[1], data.shape[2]
    y0, x0, roi_h, roi_w = central_roi_bounds(height, width, roi_fraction)
    roi_means = []
    for index in indices:
        frame = decode_frame(data[int(index)])
        roi_means.append(float(frame[y0 : y0 + roi_h, x0 : x0 + roi_w].mean()))
    return roi_means, float(np.std(roi_means))


def robust_temporal_reference(
    data: np.memmap,
    max_frames: int,
    trim_fraction: float,
    roi_fraction: float = 0.4,
) -> np.ndarray:
    """Brightness-aligned trimmed temporal mean used as a clean target (DN)."""
    frame_count = data.shape[0]
    indices = np.linspace(
        0, frame_count - 1, min(frame_count, max_frames), dtype=np.int64
    )
    height, width = data.shape[1], data.shape[2]
    y0, x0, roi_h, roi_w = central_roi_bounds(height, width, roi_fraction)

    frames = []
    roi_means = []
    for index in indices:
        frame = decode_frame(data[int(index)])
        roi_mean = float(frame[y0 : y0 + roi_h, x0 : x0 + roi_w].mean())
        roi_means.append(roi_mean)
        frames.append(frame)
    # Median is more robust than mean when lights flicker across the clip.
    series_roi_mean = float(np.median(roi_means))

    aligned = np.empty((len(frames), height, width), dtype=np.float32)
    for i, (frame, roi_mean) in enumerate(zip(frames, roi_means)):
        aligned[i] = frame - roi_mean + series_roi_mean

    trim = int(len(frames) * trim_fraction)
    if trim > 0 and len(frames) - 2 * trim >= 3:
        aligned.sort(axis=0)
        aligned = aligned[trim:-trim]
    reference = aligned.mean(axis=0)
    return np.clip(reference, 0.0, RAW_MAX).astype(np.float32)
