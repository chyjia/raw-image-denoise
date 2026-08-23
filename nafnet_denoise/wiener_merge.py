"""HDR+-style pairwise tile Wiener merge for Mono10 bursts.

Follows Hasinoff et al. (SIGGRAPH Asia 2016) robust pairwise temporal merge:
for each frequency bin, shrink misaligned alternate-frame energy toward the
reference using a Wiener-like weight, then average across the burst.
"""

from __future__ import annotations

import numpy as np

from .common import (
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    center_frame_index,
    gather_frame_indices,
)
from .alignment import soft_gate_burst


def _tile_noise_variance(
    tile: np.ndarray,
    window: np.ndarray,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> float:
    """Signal-independent noise variance proxy for one windowed tile (HDR+)."""
    windowed = tile.astype(np.float64) * window
    # RMS biased toward brighter content, as in the paper.
    rms = float(np.sqrt(np.mean(windowed * windowed) / max(float(np.mean(window * window)), 1e-12)))
    signal = max(rms, 0.0)
    return float(max(slope * signal + intercept, 1e-3))


def soft_flat_mask_dn(
    image: np.ndarray,
    temperature: float = 8.0,
    lo_q: float = 0.50,
    hi_q: float = 0.85,
    harden: float = 0.0,
) -> np.ndarray:
    """Soft flat weight in [0, 1] from Sobel magnitude (1 on flats, 0 on edges).

    Uses DualHead-style per-image percentiles so texture/edge regions are not
    misclassified as flat (the old fixed DN threshold bled into f10 structure).
    """
    image64 = image.astype(np.float64, copy=False)
    gx = np.zeros_like(image64)
    gy = np.zeros_like(image64)
    gx[:, 1:-1] = image64[:, 2:] - image64[:, :-2]
    gy[1:-1, :] = image64[2:, :] - image64[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    flat_vals = mag.reshape(-1)
    lo = float(np.quantile(flat_vals, lo_q))
    hi = float(np.quantile(flat_vals, hi_q))
    scaled = (mag - lo) / max(hi - lo, 1e-6)
    edge = 1.0 / (1.0 + np.exp(-float(temperature) * (scaled - 0.5)))
    if harden > 1.0:
        edge = np.clip(edge, 1e-4, 1.0 - 1e-4)
        logit = np.log(edge / (1.0 - edge))
        edge = 1.0 / (1.0 + np.exp(-float(harden) * logit))
    return (1.0 - edge).astype(np.float32)


def adaptive_spatial_wiener(
    image: np.ndarray,
    n_frames_averaged: int = 1,
    tile_size: int = 32,
    overlap: int = 16,
    spatial_c_factor: float = 1.0,
    flat_c_mult: float = 2.5,
    dark_boost: float = 0.5,
    dark_ref_dn: float = 400.0,
    edge_c_mult: float = 1.0,
    mask_harden: float = 0.0,
    freq_gamma: float = 0.0,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """Region-aware spatial Wiener: flat c↑, edge c↓ (HDR+ γ + DRA blend).

    Unlike the old global adaptive (flat-only boost with a leaky mask), edges can
    use ``edge_c_mult < 1`` so structure gets *less* spatial shrinkage than the
    baseline pass. Optional ``freq_gamma`` applies IPOL-style ``f(ω)=γ|ω|``
    noise shaping on the flat/strong pass only.
    """
    base_c = float(max(spatial_c_factor, 1e-3))
    mean_dn = float(np.mean(image))
    dark_scale = 1.0 + float(max(dark_boost, 0.0)) * max(
        0.0, (float(dark_ref_dn) - mean_dn) / max(float(dark_ref_dn), 1.0)
    )
    flat_c = base_c * float(max(flat_c_mult, 1.0)) * dark_scale
    edge_c = base_c * float(max(edge_c_mult, 1e-3))
    edge_pass = spatial_wiener_tiles(
        image,
        n_frames_averaged=n_frames_averaged,
        tile_size=tile_size,
        overlap=overlap,
        c_factor=edge_c,
        freq_gamma=0.0,
        slope=slope,
        intercept=intercept,
    )
    if flat_c <= edge_c * 1.01 and float(freq_gamma) <= 0.0:
        return edge_pass
    flat_pass = spatial_wiener_tiles(
        image,
        n_frames_averaged=n_frames_averaged,
        tile_size=tile_size,
        overlap=overlap,
        c_factor=flat_c,
        freq_gamma=float(max(freq_gamma, 0.0)),
        slope=slope,
        intercept=intercept,
    )
    flat = soft_flat_mask_dn(image, harden=mask_harden)
    out = flat * flat_pass + (1.0 - flat) * edge_pass
    return np.clip(out, 0.0, RAW_MAX).astype(np.float32)


def spatial_wiener_tiles(
    image: np.ndarray,
    n_frames_averaged: int = 1,
    tile_size: int = 32,
    overlap: int = 16,
    c_factor: float = 8.0,
    freq_gamma: float = 0.0,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """HDR+ post-merge 2D DFT spatial Wiener shrinkage.

    Applies ``T * |T|^2 / (|T|^2 + c σ^2)`` per tile. Noise variance is scaled
    by ``1/N`` assuming the temporal merge averaged ``N`` frames perfectly.
    With ``freq_gamma > 0``, inflate σ² by ``(1 + γ|ω|)`` (IPOL HDR+ shaping).
    """
    height, width = image.shape
    n_avg = max(int(n_frames_averaged), 1)
    pad = tile_size
    padded = np.pad(
        image.astype(np.float64, copy=False),
        ((pad, pad), (pad, pad)),
        mode="reflect",
    )
    padded_h, padded_w = padded.shape
    window_1d = np.hanning(tile_size).astype(np.float64)
    window = np.outer(window_1d, window_1d)
    window_power = float(np.mean(window * window))
    n_fft = tile_size * tile_size
    # Single-image DFT noise scale (no *2 difference factor).
    noise_scale = n_fft / max(window_power, 1e-12)
    gamma = float(max(freq_gamma, 0.0))
    freq_shape = None
    if gamma > 0.0:
        fy = np.fft.fftfreq(tile_size)
        fx = np.fft.fftfreq(tile_size)
        wy, wx = np.meshgrid(fy, fx, indexing="ij")
        # |ω| in cycles/sample; scale so corner ≈ 1.
        freq_shape = 1.0 + gamma * np.sqrt(wx * wx + wy * wy) / np.sqrt(0.5)

    step = max(tile_size - overlap, 1)
    accum = np.zeros((padded_h, padded_w), dtype=np.float64)
    weight = np.zeros((padded_h, padded_w), dtype=np.float64)

    for y0 in range(0, padded_h - tile_size + 1, step):
        for x0 in range(0, padded_w - tile_size + 1, step):
            y1 = y0 + tile_size
            x1 = x0 + tile_size
            tile = padded[y0:y1, x0:x1]
            sigma2 = _tile_noise_variance(tile, window, slope=slope, intercept=intercept)
            sigma2_eff = sigma2 / float(n_avg)
            c_sigma2 = float(c_factor) * noise_scale * sigma2_eff
            coeffs = np.fft.fft2(tile * window)
            power = coeffs.real * coeffs.real + coeffs.imag * coeffs.imag
            denom = c_sigma2 if freq_shape is None else c_sigma2 * freq_shape
            shrink = power / (power + denom)
            # Keep DC (mean) unshrunk to avoid brightness drift.
            shrink[0, 0] = 1.0
            filtered = np.fft.ifft2(coeffs * shrink).real
            accum[y0:y1, x0:x1] += filtered
            weight[y0:y1, x0:x1] += window

    safe = weight > 1e-4
    out_padded = padded.copy()
    out_padded[safe] = accum[safe] / weight[safe]
    out = out_padded[pad : pad + height, pad : pad + width]
    return np.clip(out, 0.0, RAW_MAX).astype(np.float32)


def pairwise_wiener_merge_tiles(
    frames: list[np.ndarray],
    reference_index: int | None = None,
    tile_size: int = 32,
    overlap: int = 16,
    c_factor: float = 8.0,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """Merge aligned DN frames with HDR+ pairwise frequency-domain Wiener."""
    if not frames:
        raise ValueError("frames must be non-empty")
    if reference_index is None:
        reference_index = center_frame_index(len(frames))
    height, width = frames[reference_index].shape
    n_frames = len(frames)

    # Pad so Hann windows have full support at the image borders.
    pad = tile_size
    stack = [
        np.pad(
            frame.astype(np.float64, copy=False),
            ((pad, pad), (pad, pad)),
            mode="reflect",
        )
        for frame in frames
    ]
    padded_h, padded_w = stack[reference_index].shape
    reference_padded = stack[reference_index]

    window_1d = np.hanning(tile_size).astype(np.float64)
    window = np.outer(window_1d, window_1d)
    window_power = float(np.mean(window * window))
    # Dz = T0-Tz scales noise by: n_fft^2 (DFT), 1/window_power, and *2 for difference.
    n_fft = tile_size * tile_size
    noise_scale = (n_fft / max(window_power, 1e-12)) * 2.0

    step = max(tile_size - overlap, 1)
    accum = np.zeros((padded_h, padded_w), dtype=np.float64)
    weight = np.zeros((padded_h, padded_w), dtype=np.float64)

    for y0 in range(0, padded_h - tile_size + 1, step):
        for x0 in range(0, padded_w - tile_size + 1, step):
            y1 = y0 + tile_size
            x1 = x0 + tile_size
            tiles = [frame[y0:y1, x0:x1] for frame in stack]
            ref = tiles[reference_index]
            sigma2 = _tile_noise_variance(ref, window, slope=slope, intercept=intercept)
            c_sigma2 = float(c_factor) * noise_scale * sigma2

            t0 = np.fft.fft2(ref * window)
            merged_freq = np.zeros_like(t0)
            for tile in tiles:
                tz = np.fft.fft2(tile * window)
                dz = t0 - tz
                dz_power = dz.real * dz.real + dz.imag * dz.imag
                az = dz_power / (dz_power + c_sigma2)
                # (1-Az)*Tz + Az*T0
                merged_freq += (1.0 - az) * tz + az * t0
            merged_freq /= float(n_frames)
            merged_tile = np.fft.ifft2(merged_freq).real
            accum[y0:y1, x0:x1] += merged_tile
            weight[y0:y1, x0:x1] += window

    safe = weight > 1e-4
    merged_padded = reference_padded.copy()
    merged_padded[safe] = accum[safe] / weight[safe]
    merged = merged_padded[pad : pad + height, pad : pad + width]
    return np.clip(merged, 0.0, RAW_MAX).astype(np.float32)


def merge_burst_dn(
    frames_dn: list[np.ndarray],
    reference_index: int | None = None,
    align: bool = True,
    tile_size: int = 32,
    overlap: int = 16,
    c_factor: float = 8.0,
    spatial_wiener: bool = False,
    spatial_c_factor: float | None = None,
    spatial_adaptive: bool = False,
    spatial_flat_c_mult: float = 2.5,
    spatial_dark_boost: float = 0.5,
    spatial_edge_c_mult: float = 1.0,
    spatial_mask_harden: float = 0.0,
    spatial_freq_gamma: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Brightness/ECC soft-gate align (optional) then pairwise Wiener merge.

    When ``spatial_wiener`` is True, apply HDR+ 2D spatial Wiener after the
    temporal merge (noise variance scaled by 1/N). With ``spatial_adaptive``,
    flats get a stronger ``c`` (and optional freq γ) while edges can use a
    milder ``spatial_edge_c_mult``.
    """
    if reference_index is None:
        reference_index = center_frame_index(len(frames_dn))
    aligned = frames_dn
    responses = np.ones(len(frames_dn), dtype=np.float32)
    if align and len(frames_dn) > 1:
        aligned, responses = soft_gate_burst(
            frames_dn,
            reference_index,
            apply_residual_warp=True,
        )
    merged = pairwise_wiener_merge_tiles(
        aligned,
        reference_index=reference_index,
        tile_size=tile_size,
        overlap=overlap,
        c_factor=c_factor,
    )
    if spatial_wiener:
        spatial_c = 1.0 if spatial_c_factor is None else float(spatial_c_factor)
        if spatial_adaptive:
            merged = adaptive_spatial_wiener(
                merged,
                n_frames_averaged=len(frames_dn),
                tile_size=tile_size,
                overlap=overlap,
                spatial_c_factor=spatial_c,
                flat_c_mult=spatial_flat_c_mult,
                dark_boost=spatial_dark_boost,
                edge_c_mult=spatial_edge_c_mult,
                mask_harden=spatial_mask_harden,
                freq_gamma=spatial_freq_gamma,
            )
        else:
            merged = spatial_wiener_tiles(
                merged,
                n_frames_averaged=len(frames_dn),
                tile_size=tile_size,
                overlap=overlap,
                c_factor=spatial_c,
                freq_gamma=spatial_freq_gamma,
            )
    meta = {
        "input_frames": len(frames_dn),
        "alignment_response_median": float(np.median(responses)),
        "tile_size": tile_size,
        "c_factor": c_factor,
        "spatial_wiener": bool(spatial_wiener),
        "spatial_c_factor": (
            None if not spatial_wiener else (1.0 if spatial_c_factor is None else float(spatial_c_factor))
        ),
        "spatial_adaptive": bool(spatial_adaptive and spatial_wiener),
        "spatial_flat_c_mult": float(spatial_flat_c_mult),
        "spatial_dark_boost": float(spatial_dark_boost),
        "spatial_edge_c_mult": float(spatial_edge_c_mult),
        "spatial_mask_harden": float(spatial_mask_harden),
        "spatial_freq_gamma": float(spatial_freq_gamma),
    }
    return merged, meta


def merge_from_memmap(
    frames,
    target_index: int,
    input_frames: int = 8,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Gather a burst window from a Mono10 memmap and Wiener-merge it."""
    from .common import decode_mono10

    indices = gather_frame_indices(target_index, frames.shape[0], input_frames)
    decoded = [decode_mono10(frames[index]) for index in indices]
    reference_index = center_frame_index(input_frames)
    return merge_burst_dn(decoded, reference_index=reference_index, **kwargs)
