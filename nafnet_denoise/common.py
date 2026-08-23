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
DEFAULT_BLACK_LEVEL_DN = 60.0
DEFAULT_DARK_VARIANCE_PER_S = 3.0
READ_VARIANCE_DN2 = PTC_INTERCEPT + PTC_SLOPE * DEFAULT_BLACK_LEVEL_DN
READ_NOISE_DN = float(np.sqrt(max(READ_VARIANCE_DN2, 0.0)))

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
    dark_variance_per_s: float = 0.0,
    use_sigma: bool = False,
    use_exposure: bool = True,
    postmerge_calib=None,
    measured_fe_sigma: bool = False,
) -> np.ndarray:
    """Stack aligned VST frames, optional sigma map, and exposure channel.

    Channel layout when ``use_sigma`` and ``use_exposure`` are both True:
    ``[vst_0, ..., vst_{N-1}, sigma_ref, exposure]``.

    Sigma priority when ``use_sigma``:
    1. ``measured_fe_sigma`` — flat highpass MAD on the reference (FFDNet-style)
    2. ``postmerge_calib`` — affine residual post-Wiener model
    3. single-frame PTC
    """
    if reference_index is None:
        reference_index = center_frame_index(len(frames_dn))
    aligned = align_frames_to_reference(frames_dn, reference_index)
    vst_channels = [
        raw_to_model_input(
            frame,
            exposure_ms=exposure_ms,
            dark_variance_per_s=dark_variance_per_s,
        )
        for frame in aligned
    ]
    channels = list(vst_channels)
    if use_sigma:
        ref = aligned[reference_index]
        if measured_fe_sigma:
            from .postmerge_noise import measured_fe_sigma_map

            channels.append(
                measured_fe_sigma_map(
                    ref,
                    exposure_ms=exposure_ms,
                    dark_variance_per_s=dark_variance_per_s,
                )
            )
        elif postmerge_calib is not None:
            from .postmerge_noise import residual_noise_sigma_map

            channels.append(
                residual_noise_sigma_map(
                    ref,
                    postmerge_calib,
                    exposure_ms=exposure_ms,
                    dark_variance_per_s=dark_variance_per_s,
                )
            )
        else:
            channels.append(
                noise_sigma_map(
                    ref,
                    exposure_ms=exposure_ms,
                    dark_variance_per_s=dark_variance_per_s,
                )
            )
    if use_exposure:
        exposure = normalize_exposure(exposure_ms)
        channels.append(np.full_like(vst_channels[0], exposure, dtype=np.float32))
    return np.ascontiguousarray(np.stack(channels, axis=0))


def model_input_channels(
    input_frames: int,
    use_exposure: bool = True,
    use_sigma: bool = False,
) -> int:
    return input_frames + (1 if use_sigma else 0) + (1 if use_exposure else 0)


def expand_intro_state_for_sigma(
    state: dict[str, torch.Tensor],
    input_frames: int,
) -> dict[str, torch.Tensor]:
    """Expand ``intro`` conv from ``[VST*N, exp]`` to ``[VST*N, sigma, exp]``."""
    import torch

    weight = state["intro.weight"]
    old_c = int(weight.shape[1])
    new_c = input_frames + 2
    if old_c == new_c:
        return state
    if old_c != input_frames + 1:
        raise ValueError(
            f"Cannot expand intro for sigma: expected {input_frames + 1} "
            f"input channels, got {old_c}"
        )
    expanded = dict(state)
    new_weight = weight.new_zeros(weight.shape[0], new_c, weight.shape[2], weight.shape[3])
    new_weight[:, :input_frames] = weight[:, :input_frames]
    # sigma channel starts at 0; copy exposure weights to the last channel.
    new_weight[:, input_frames + 1] = weight[:, input_frames]
    expanded["intro.weight"] = new_weight
    return expanded


def expand_intro_state_for_input_frames(
    state: dict[str, torch.Tensor],
    new_frames: int,
    use_sigma: bool = True,
    use_exposure: bool = True,
) -> dict[str, torch.Tensor]:
    """Expand ``intro`` VST channels when increasing ``input_frames`` (center-aligned).

    Old VST kernels are placed so their center frame lands on the new center;
    newly added outer frame channels are zero-initialized. Sigma/exposure
    kernels (if present) are copied to the trailing channels.
    """
    weight = state["intro.weight"]
    old_c = int(weight.shape[1])
    new_extras = (1 if use_sigma else 0) + (1 if use_exposure else 0)
    new_c = int(new_frames) + new_extras
    if old_c == new_c:
        return state

    # Infer old frame count: try (sigma+exp), then exp-only, then frames-only.
    old_frames = None
    old_has_sigma = False
    old_has_exposure = False
    for has_sigma, has_exp in ((True, True), (False, True), (True, False), (False, False)):
        extras = (1 if has_sigma else 0) + (1 if has_exp else 0)
        candidate = old_c - extras
        if candidate >= 1:
            old_frames = candidate
            old_has_sigma = has_sigma
            old_has_exposure = has_exp
            break
    if old_frames is None or old_frames > new_frames:
        raise ValueError(
            f"Cannot expand intro frames: old_c={old_c} -> new_frames={new_frames} "
            f"(new_c={new_c})"
        )

    expanded = dict(state)
    new_weight = weight.new_zeros(weight.shape[0], new_c, weight.shape[2], weight.shape[3])
    old_center = center_frame_index(old_frames)
    new_center = center_frame_index(new_frames)
    shift = new_center - old_center
    for src in range(old_frames):
        dst = src + shift
        if 0 <= dst < new_frames:
            new_weight[:, dst] = weight[:, src]
    # Fill remaining outer VST channels with nearest copied frame weights.
    for dst in range(new_frames):
        if float(new_weight[:, dst].abs().sum()) > 0:
            continue
        nearest = min(range(old_frames), key=lambda src: abs((src + shift) - dst))
        new_weight[:, dst] = weight[:, nearest]

    cursor = old_frames
    dest = new_frames
    if old_has_sigma and use_sigma:
        new_weight[:, dest] = weight[:, cursor]
        cursor += 1
        dest += 1
    elif use_sigma:
        dest += 1  # leave zeros
    if old_has_exposure and use_exposure:
        new_weight[:, new_frames + new_extras - 1] = weight[:, cursor]
    expanded["intro.weight"] = new_weight
    return expanded


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


def exposure_intercept(
    exposure_ms: float | None = None,
    intercept: float = PTC_INTERCEPT,
    dark_variance_per_s: float = 0.0,
) -> float:
    exposure_s = max(float(exposure_ms or 0.0), 0.0) / 1000.0
    return float(intercept + exposure_s * max(dark_variance_per_s, 0.0))


def signal_variance_dn2(
    image_dn: np.ndarray,
    exposure_ms: float | None = None,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
) -> np.ndarray:
    signal = np.maximum(image_dn - black_level_dn, 0.0)
    read_variance = intercept + slope * black_level_dn
    dark_variance = max(float(exposure_ms or 0.0), 0.0) / 1000.0 * max(
        dark_variance_per_s,
        0.0,
    )
    return np.maximum(slope * signal + read_variance + dark_variance, 1e-6)


def vst_forward(
    image: np.ndarray,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    radicand = PTC_SLOPE * image + 0.375 * PTC_SLOPE**2 + intercept
    return (2.0 / PTC_SLOPE) * np.sqrt(np.maximum(radicand, 1e-8))


def vst_inverse(
    transformed: np.ndarray,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    image = (
        (PTC_SLOPE / 4.0) * transformed**2
        - 0.375 * PTC_SLOPE
        - intercept / PTC_SLOPE
    )
    return np.clip(image, 0.0, RAW_MAX)


VST_MIN = float(vst_forward(np.array(0.0, dtype=np.float32)))
VST_MAX = float(vst_forward(np.array(RAW_MAX, dtype=np.float32)))
VST_SCALE = VST_MAX - VST_MIN


def vst_normalize(transformed: np.ndarray) -> np.ndarray:
    return (transformed - VST_MIN) / VST_SCALE


def vst_denormalize(normalized: np.ndarray) -> np.ndarray:
    return normalized * VST_SCALE + VST_MIN


def raw_to_model_input(
    image_dn: np.ndarray,
    exposure_ms: float | None = None,
    dark_variance_per_s: float = 0.0,
) -> np.ndarray:
    intercept = exposure_intercept(
        exposure_ms,
        dark_variance_per_s=dark_variance_per_s,
    )
    return vst_normalize(vst_forward(image_dn, intercept=intercept))


def model_output_to_raw(
    normalized: np.ndarray,
    exposure_ms: float | None = None,
    dark_variance_per_s: float = 0.0,
) -> np.ndarray:
    intercept = exposure_intercept(
        exposure_ms,
        dark_variance_per_s=dark_variance_per_s,
    )
    return vst_inverse(vst_denormalize(normalized), intercept=intercept)


def noise_sigma_map(
    image_dn: np.ndarray,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
    exposure_ms: float | None = None,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
) -> np.ndarray:
    """Normalized raw-domain sigma map for noise-conditioned models."""
    sigma = np.sqrt(
        signal_variance_dn2(
            image_dn,
            exposure_ms,
            slope,
            intercept,
            black_level_dn,
            dark_variance_per_s,
        )
    )
    sigma_max = np.sqrt(
        float(
            signal_variance_dn2(
                np.asarray(RAW_MAX),
                exposure_ms,
                slope,
                intercept,
                black_level_dn,
                dark_variance_per_s,
            )
        )
    )
    return (sigma / sigma_max).astype(np.float32)


def poisson_gaussian_noise(
    clean_dn: np.ndarray,
    rng: np.random.Generator,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
    exposure_ms: float | None = None,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
) -> np.ndarray:
    """Add signal-dependent Gaussian noise with optional perturbed PTC parameters."""
    variance = signal_variance_dn2(
        clean_dn,
        exposure_ms,
        slope,
        intercept,
        black_level_dn,
        dark_variance_per_s,
    )
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
