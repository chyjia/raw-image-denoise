"""P6 post-fusion helpers: detail transfer + multi-cue edge maps."""

from __future__ import annotations

import cv2
import numpy as np

from .infer_ensemble import soft_edge_mask_dn
from .multiband_fuse import _edge_flat_maps, flat_bilateral_boost


def highpass_dn(image: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
    return (img - blur).astype(np.float32)


def detail_transfer(
    base: np.ndarray,
    detail_src: np.ndarray,
    guide: np.ndarray | None = None,
    amount: float = 1.0,
    sigma: float = 1.2,
    temperature: float = 8.0,
    harden: float = 16.0,
    edge_only: bool = True,
) -> np.ndarray:
    """Add highpass(detail_src)−highpass(base), optionally only on edges.

    Burt/HDR+-style: keep base low-frequency, pull structure highs from
    ``detail_src`` (lap_ms) where the soft edge map is high.
    """
    base32 = np.ascontiguousarray(base, dtype=np.float32)
    src32 = np.ascontiguousarray(detail_src, dtype=np.float32)
    delta = highpass_dn(src32, sigma=sigma) - highpass_dn(base32, sigma=sigma)
    a = float(amount)
    if edge_only:
        if guide is None:
            guide = 0.5 * base32 + 0.5 * src32
        emap, _ = _edge_flat_maps(guide, temperature=temperature, harden=harden)
        out = base32 + (a * emap) * delta
    else:
        out = base32 + a * delta
    return out.astype(np.float32)


def multi_cue_edge_map(
    guide: np.ndarray,
    temperature: float = 8.0,
    harden: float = 16.0,
    log_sigma: float = 1.0,
    sobel_weight: float = 0.65,
) -> np.ndarray:
    """Combine Sobel soft-edge with LoG magnitude (multi-cue gate)."""
    sobel = soft_edge_mask_dn(guide, temperature=temperature)
    img = np.ascontiguousarray(guide, dtype=np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(log_sigma))
    log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    mag = np.abs(log)
    scale = float(np.percentile(mag, 90.0)) + 1e-6
    log_n = np.clip(mag / scale, 0.0, 1.0)
    # Match sobel dynamic range with same temperature-style sigmoid
    log_s = 1.0 / (1.0 + np.exp(-float(temperature) * (log_n - 0.5)))
    w = float(np.clip(sobel_weight, 0.0, 1.0))
    emap = w * sobel + (1.0 - w) * log_s
    if harden > 0.0:
        emap = 1.0 / (1.0 + np.exp(-float(harden) * (emap - 0.5)))
    return emap.astype(np.float32)


def blend_with_edge_map(
    sota: np.ndarray,
    edge: np.ndarray,
    edge_map: np.ndarray,
) -> np.ndarray:
    emap = np.clip(edge_map, 0.0, 1.0).astype(np.float32)
    return (emap * edge + (1.0 - emap) * sota).astype(np.float32)


def apply_flat_bilateral_if(
    image: np.ndarray,
    fps: float,
    low_fps: float = 1.0,
    mid_fps: float = 5.0,
    low_strength: float = 0.65,
    mid_strength: float = 0.35,
    harden: float = 40.0,
) -> np.ndarray:
    """P6-b: bilateral on flats for fps<=low (strong) or low<fps<=mid (mild)."""
    if float(fps) <= float(low_fps) and low_strength > 0.0:
        return flat_bilateral_boost(
            image, guide=image, flat_strength=low_strength, harden=harden
        )
    if float(fps) <= float(mid_fps) and mid_strength > 0.0:
        return flat_bilateral_boost(
            image, guide=image, flat_strength=mid_strength, harden=harden
        )
    return image
