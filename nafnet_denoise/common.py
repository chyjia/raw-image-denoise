"""Shared helpers for the NAFNet Mono10 VST-domain denoising pipeline.

PTC constants are the mean of the two independent Mono10 grayscale-chart
calibrations in ``泊松分布/ptc_grayscale_videos``:
    variance_dn2 = 0.10618515 * mean_dn + 2.00145577
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PTC_SLOPE = 0.10618515
PTC_INTERCEPT = 2.00145577
RAW_MAX = 1023.0
READ_NOISE_DN = float(np.sqrt(max(PTC_INTERCEPT, 0.0)))

_META_RE = re.compile(
    r"_w(?P<width>\d+)_h(?P<height>\d+)_pMono10_f(?P<fps>\d+(?:\.\d+)?)\.raw$",
    re.IGNORECASE,
)
_EXPOSURE_RE = re.compile(r"exposure_(\d+)ms", re.IGNORECASE)
_FPS_RE = re.compile(r"_f(\d+(?:\.\d+)?)(?:\.raw)?$", re.IGNORECASE)

EXPOSURE_NORM_MS = 200.0
TARGET_MEAN_DN_MIN = 150.0
TARGET_MEAN_DN_MAX = 950.0
DEFAULT_INPUT_FRAMES = 4


def center_frame_index(input_frames: int) -> int:
    """Index of the denoised frame inside a temporal window."""
    if input_frames == 1:
        return 0
    return input_frames // 2 - 1


def gather_frame_indices(target_index: int, frame_count: int, input_frames: int) -> list[int]:
    """Return ``input_frames`` indices centered on ``target_index`` with edge clamping."""
    center = center_frame_index(input_frames)
    start = target_index - center
    indices = [start + offset for offset in range(input_frames)]
    return [min(frame_count - 1, max(0, index)) for index in indices]


def align_frames_to_reference(
    frames: list[np.ndarray],
    reference_index: int,
) -> list[np.ndarray]:
    """Remove per-frame brightness offsets relative to the reference frame."""
    reference = frames[reference_index]
    reference_mean = float(reference.mean())
    aligned = []
    for index, frame in enumerate(frames):
        if index == reference_index:
            aligned.append(frame.astype(np.float32, copy=False))
            continue
        offset = float(frame.mean()) - reference_mean
        aligned.append(np.clip(frame - offset, 0.0, RAW_MAX).astype(np.float32))
    return aligned


def frames_to_model_input(
    frames_dn: list[np.ndarray],
    exposure_ms: float,
    reference_index: int | None = None,
) -> np.ndarray:
    """Stack aligned VST frames and an exposure-conditioning channel."""
    if reference_index is None:
        reference_index = center_frame_index(len(frames_dn))
    aligned = align_frames_to_reference(frames_dn, reference_index)
    vst_channels = [raw_to_model_input(frame) for frame in aligned]
    exposure = normalize_exposure(exposure_ms)
    exposure_channel = np.full_like(vst_channels[0], exposure, dtype=np.float32)
    return np.ascontiguousarray(np.stack([*vst_channels, exposure_channel], axis=0))


def model_input_channels(input_frames: int, use_exposure: bool = True) -> int:
    return input_frames + (1 if use_exposure else 0)


def parse_geometry(name: str) -> tuple[int, int, float]:
    match = _META_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse Mono10 geometry from {name}")
    return int(match["width"]), int(match["height"]), float(match["fps"])


def parse_exposure_ms(name: str) -> float:
    match = _EXPOSURE_RE.search(name)
    if match:
        return float(match.group(1))
    fps_match = _FPS_RE.search(name)
    if fps_match:
        fps = float(fps_match.group(1))
        if fps > 0:
            return 1000.0 / fps
    raise ValueError(f"Cannot parse exposure from {name}")


def normalize_exposure(exposure_ms: float) -> float:
    return float(exposure_ms) / EXPOSURE_NORM_MS


def decode_mono10(raw: np.ndarray) -> np.ndarray:
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


def noise_sigma_map(
    image_dn: np.ndarray,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """Normalized raw-domain sigma map for noise-conditioned models."""
    sigma = np.sqrt(np.maximum(slope * image_dn + intercept, 1e-6))
    sigma_max = np.sqrt(max(slope * RAW_MAX + intercept, 1e-6))
    return (sigma / sigma_max).astype(np.float32)


def poisson_gaussian_noise(
    clean_dn: np.ndarray,
    rng: np.random.Generator,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """Add signal-dependent Gaussian noise with optional perturbed PTC parameters."""
    variance = np.maximum(slope * clean_dn + intercept, 1e-6)
    noisy = clean_dn + rng.normal(0.0, 1.0, size=clean_dn.shape).astype(np.float32) * np.sqrt(variance)
    return np.clip(noisy, 0.0, RAW_MAX).astype(np.float32)


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


def ps4k_to_mono_luminance(ps4k: np.ndarray) -> np.ndarray:
    """Convert PixelShift200 ``ps4k`` (H, W, 4) uint16 to float32 luminance."""
    image = ps4k.astype(np.float32)
    if image.ndim == 3 and image.shape[-1] == 4:
        return image.mean(axis=-1)
    if image.ndim == 2:
        return image
    raise ValueError(f"Unexpected ps4k shape: {ps4k.shape}")
