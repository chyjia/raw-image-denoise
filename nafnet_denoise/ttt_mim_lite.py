"""TTT-MIM output-lite (ECCV'24): blind-spot patch consistency without weight updates."""

from __future__ import annotations

import cv2
import numpy as np

from .multiband_fuse import _edge_flat_maps


def checker_blindspot_fill(img: np.ndarray, period: int = 2) -> np.ndarray:
    """Fill masked checker pixels with local median (blind-spot proxy)."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    p = max(int(period), 1)
    yy, xx = np.mgrid[: x.shape[0], : x.shape[1]]
    mask = ((yy // p + xx // p) % 2 == 0).astype(np.float32)
    med = cv2.medianBlur(x, 3)
    return (mask * x + (1.0 - mask) * med).astype(np.float32)


def mim_flat_consistency(
    image: np.ndarray,
    *,
    strength: float = 0.5,
    sigma: float = 1.2,
    period: int = 2,
    flat_pct: float = 40.0,
    harden: float = 40.0,
) -> np.ndarray:
    """Pull flat HP toward checkerboard blind-spot self-consistent version."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    if strength <= 0:
        return img
    _, flat = _edge_flat_maps(img, temperature=8.0, harden=harden)
    if flat_pct > 0:
        sob = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            img, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = (flat > 0.5) & (sob < np.percentile(sob, flat_pct))
        flat = flat.astype(np.float32)
    filled = checker_blindspot_fill(img, period=period)
    low = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
    low_f = cv2.GaussianBlur(filled, (0, 0), sigmaX=float(sigma))
    hp = img - low
    hp_f = filled - low_f
    s = float(np.clip(strength, 0.0, 1.0))
    out = img - s * flat * (hp - hp_f)
    return out.astype(np.float32)
