"""Shared helpers for the Restormer VST-domain denoising pipeline.

The PTC constants come from the current independent Mono10 grayscale-chart
calibrations in ``泊松分布/ptc_grayscale_videos``. Training and inference happen in a
normalized generalized-Anscombe VST domain so the Poisson-Gaussian noise is
approximately variance-stabilized before it reaches the network.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PTC_SLOPE = 0.10618515
PTC_INTERCEPT = 2.00145577
RAW_MAX = 1023.0

_META_RE = re.compile(
    r"_w(?P<width>\d+)_h(?P<height>\d+)_pMono10_f(?P<fps>\d+(?:\.\d+)?)\.raw$",
    re.IGNORECASE,
)
_EXPOSURE_RE = re.compile(r"exposure_(\d+)ms", re.IGNORECASE)
_FPS_RE = re.compile(r"_f(\d+(?:\.\d+)?)(?:\.raw)?$", re.IGNORECASE)


def parse_geometry(name: str) -> tuple[int, int, float]:
    match = _META_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse Mono10 geometry from {name}")
    return int(match["width"]), int(match["height"]), float(match["fps"])


def parse_exposure_ms(name: str) -> float:
    """Exposure priority: explicit exposure tag, else frame interval 1000/fps."""
    match = _EXPOSURE_RE.search(name)
    if match:
        return float(match.group(1))
    fps_match = _FPS_RE.search(name)
    if fps_match:
        fps = float(fps_match.group(1))
        if fps > 0:
            return 1000.0 / fps
    raise ValueError(f"Cannot parse exposure from {name}")


# Exposure condition is normalized so 200 ms maps to 1.0; keeps the channel O(1).
EXPOSURE_NORM_MS = 200.0


def normalize_exposure(exposure_ms: float) -> float:
    return float(exposure_ms) / EXPOSURE_NORM_MS


def decode_mono10(raw: np.ndarray) -> np.ndarray:
    """Decode unpacked low-aligned Mono10 stored as little-endian uint16."""
    return (raw & 0x03FF).astype(np.float32)


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


# Normalization so the VST domain roughly lands in [0, 1] before the network.
VST_MIN = float(vst_forward(np.array(0.0, dtype=np.float32)))
VST_MAX = float(vst_forward(np.array(RAW_MAX, dtype=np.float32)))
VST_SCALE = VST_MAX - VST_MIN


def vst_normalize(transformed: np.ndarray) -> np.ndarray:
    return (transformed - VST_MIN) / VST_SCALE


def vst_denormalize(normalized: np.ndarray) -> np.ndarray:
    return normalized * VST_SCALE + VST_MIN


def raw_to_model_input(image_dn: np.ndarray) -> np.ndarray:
    return vst_normalize(vst_forward(image_dn))


def model_output_to_raw(normalized: np.ndarray) -> np.ndarray:
    return vst_inverse(vst_denormalize(normalized))


def memmap_frames(path: Path, width: int, height: int) -> np.memmap:
    bytes_per_frame = width * height * 2
    file_size = path.stat().st_size
    if file_size % bytes_per_frame:
        raise ValueError(
            f"{path.name}: {file_size} bytes is not a whole number of "
            f"{width}x{height} unpacked Mono10 frames."
        )
    frame_count = file_size // bytes_per_frame
    return np.memmap(path, mode="r", dtype="<u2", shape=(frame_count, height, width))
