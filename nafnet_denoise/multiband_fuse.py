"""Burt–Adelson Laplacian multi-band fusion helpers (P5-a)."""

from __future__ import annotations

import cv2
import numpy as np

from .infer_ensemble import blend_sota_edgekd, soft_edge_mask_dn


def _even_pad(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    ph = h % 2
    pw = w % 2
    if ph == 0 and pw == 0:
        return image, (0, 0)
    padded = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
    return padded, (ph, pw)


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    levels = int(max(levels, 1))
    pyr: list[np.ndarray] = []
    current, _ = _even_pad(image.astype(np.float32, copy=False))
    pyr.append(current)
    for _ in range(levels - 1):
        current, _ = _even_pad(current)
        current = cv2.pyrDown(current, dstsize=(current.shape[1] // 2, current.shape[0] // 2))
        pyr.append(current.astype(np.float32, copy=False))
    return pyr


def laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = gaussian_pyramid(image, levels)
    laps: list[np.ndarray] = []
    for i in range(len(gauss) - 1):
        size = (gauss[i].shape[1], gauss[i].shape[0])
        up = cv2.pyrUp(gauss[i + 1], dstsize=size).astype(np.float32, copy=False)
        # pyrUp size may still mismatch by 1 on odd dims after pad path
        if up.shape != gauss[i].shape:
            up = cv2.resize(up, (gauss[i].shape[1], gauss[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        laps.append(gauss[i] - up)
    laps.append(gauss[-1])
    return laps


def collapse_laplacian(laps: list[np.ndarray], out_shape: tuple[int, int]) -> np.ndarray:
    current = laps[-1].astype(np.float32, copy=False)
    for band in reversed(laps[:-1]):
        size = (band.shape[1], band.shape[0])
        up = cv2.pyrUp(current, dstsize=size).astype(np.float32, copy=False)
        if up.shape != band.shape:
            up = cv2.resize(up, (band.shape[1], band.shape[0]), interpolation=cv2.INTER_LINEAR)
        current = up + band
    if current.shape != out_shape:
        current = current[: out_shape[0], : out_shape[1]]
    return current.astype(np.float32, copy=False)


def multiband_fuse(
    sota: np.ndarray,
    edge: np.ndarray,
    band_edge_weights: list[float],
    guide: np.ndarray | None = None,
    spatial_mix: float = 0.0,
    temperature: float = 8.0,
    harden: float = 16.0,
) -> np.ndarray:
    """Fuse two DN images with Laplacian band weights (1=take edge band).

    ``band_edge_weights`` length = number of Laplacian levels (last is residual LF).
    Optional ``spatial_mix`` blends each band weight toward a downsampled soft
    edge map (0=pure band schedule, 1=fully spatial).
    """
    sota32 = np.ascontiguousarray(sota, dtype=np.float32)
    edge32 = np.ascontiguousarray(edge, dtype=np.float32)
    levels = len(band_edge_weights)
    if levels < 2:
        raise ValueError("need >=2 band weights")
    ls = laplacian_pyramid(sota32, levels)
    le = laplacian_pyramid(edge32, levels)
    if len(ls) != levels or len(le) != levels:
        raise RuntimeError("pyramid level mismatch")

    emap_full = None
    if spatial_mix > 0.0:
        if guide is None:
            guide = 0.5 * sota32 + 0.5 * edge32
        emap_full = soft_edge_mask_dn(guide, temperature=temperature)
        if harden > 0.0:
            emap_full = 1.0 / (1.0 + np.exp(-float(harden) * (emap_full - 0.5)))

    fused_bands: list[np.ndarray] = []
    for i, (a, b) in enumerate(zip(ls, le)):
        w = float(np.clip(band_edge_weights[i], 0.0, 1.0))
        if emap_full is not None and spatial_mix > 0.0:
            emap = cv2.resize(
                emap_full,
                (a.shape[1], a.shape[0]),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            # Interpolate constant band weight ↔ spatial edge map.
            w_map = (1.0 - spatial_mix) * w + spatial_mix * emap
            fused_bands.append(w_map * b + (1.0 - w_map) * a)
        else:
            fused_bands.append(w * b + (1.0 - w) * a)
    return collapse_laplacian(fused_bands, sota32.shape)


def _edge_flat_maps(
    guide: np.ndarray,
    temperature: float = 8.0,
    harden: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    emap = soft_edge_mask_dn(guide, temperature=temperature)
    if harden > 0.0:
        emap = 1.0 / (1.0 + np.exp(-float(harden) * (emap - 0.5)))
    return emap.astype(np.float32), (1.0 - emap).astype(np.float32)


def flat_half_scale_boost(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    flat_strength: float = 0.65,
    temperature: float = 8.0,
    harden: float = 16.0,
) -> np.ndarray:
    """P5-b: replace flat regions with a 1/2-scale smoothed version.

    High frequencies on edges are preserved from ``image``; flats take more of
    the upsampled low-pass (stronger grain suppression).
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    if guide is None:
        guide = img
    _, flat = _edge_flat_maps(guide, temperature=temperature, harden=harden)
    small = cv2.pyrDown(img)
    # light extra blur on the coarse scale (flat grain)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=0.8)
    up = cv2.pyrUp(small, dstsize=(img.shape[1], img.shape[0])).astype(np.float32)
    if up.shape != img.shape:
        up = cv2.resize(up, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    s = float(np.clip(flat_strength, 0.0, 1.0))
    out = img * (1.0 - s * flat) + up * (s * flat)
    return out.astype(np.float32)


def flat_bilateral_boost(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    flat_strength: float = 0.5,
    temperature: float = 8.0,
    harden: float = 24.0,
    d: int = 7,
    sigma_color: float = 12.0,
    sigma_space: float = 7.0,
) -> np.ndarray:
    """Edge-aware flat-only bilateral blend (safer than pyramid blur)."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    if guide is None:
        guide = img
    _, flat = _edge_flat_maps(guide, temperature=temperature, harden=harden)
    # bilateralFilter expects 8U/32F; DN values are fine as 32F
    smooth = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    s = float(np.clip(flat_strength, 0.0, 1.0))
    out = img * (1.0 - s * flat) + smooth * (s * flat)
    return out.astype(np.float32)


def pixel_blend_baseline(
    sota: np.ndarray,
    edge: np.ndarray,
    temperature: float = 8.0,
    harden: float = 16.0,
) -> np.ndarray:
    out, _ = blend_sota_edgekd(
        sota, edge, temperature=temperature, harden=harden, edge_weight=1.0
    )
    return out
