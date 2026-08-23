"""P7 fusion ops from DualExNet / SNR-gate / guided-filter / freq-split lit."""

from __future__ import annotations

import cv2
import numpy as np

from .infer_ensemble import blend_sota_edgekd, soft_edge_mask_dn
from .multiband_fuse import _edge_flat_maps, flat_bilateral_boost


def residual_gate_blend(
    sota: np.ndarray,
    edge: np.ndarray,
    temperature: float = 12.0,
    harden: float = 16.0,
    percentile: float = 90.0,
    mix: float = 1.0,
) -> np.ndarray:
    """DualExNet-style content gate from |edge−sota| magnitude."""
    s = np.ascontiguousarray(sota, dtype=np.float32)
    e = np.ascontiguousarray(edge, dtype=np.float32)
    delta = np.abs(e - s)
    scale = float(np.percentile(delta, float(percentile))) + 1e-6
    n = np.clip(delta / scale, 0.0, 1.0)
    gate = 1.0 / (1.0 + np.exp(-float(temperature) * (n - 0.5)))
    if harden > 0.0:
        gate = 1.0 / (1.0 + np.exp(-float(harden) * (gate - 0.5)))
    m = float(np.clip(mix, 0.0, 1.0))
    gate = m * gate
    return (gate * e + (1.0 - gate) * s).astype(np.float32)


def local_variance_gate_blend(
    sota: np.ndarray,
    edge: np.ndarray,
    guide: np.ndarray | None = None,
    win: int = 7,
    temperature: float = 10.0,
    harden: float = 16.0,
    percentile: float = 85.0,
    edge_bias: float = 0.5,
) -> np.ndarray:
    """SNR-ADDNet-lite: high local variance → more edge arm."""
    s = np.ascontiguousarray(sota, dtype=np.float32)
    e = np.ascontiguousarray(edge, dtype=np.float32)
    g = np.ascontiguousarray(guide if guide is not None else 0.5 * s + 0.5 * e, dtype=np.float32)
    k = int(win) | 1
    mean = cv2.blur(g, (k, k))
    mean2 = cv2.blur(g * g, (k, k))
    var = np.clip(mean2 - mean * mean, 0.0, None)
    scale = float(np.percentile(var, float(percentile))) + 1e-6
    n = np.clip(var / scale, 0.0, 1.0)
    gate = 1.0 / (1.0 + np.exp(-float(temperature) * (n - 0.5)))
    if harden > 0.0:
        gate = 1.0 / (1.0 + np.exp(-float(harden) * (gate - 0.5)))
    # Mix with sobel soft edge so pure texture grain does not always pick edge arm
    sobel = soft_edge_mask_dn(g, temperature=8.0)
    b = float(np.clip(edge_bias, 0.0, 1.0))
    gate = b * gate + (1.0 - b) * sobel
    return (gate * e + (1.0 - gate) * s).astype(np.float32)


def guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter (single-channel)."""
    I = np.ascontiguousarray(guide, dtype=np.float32)
    p = np.ascontiguousarray(src, dtype=np.float32)
    r = max(1, int(radius))
    k = 2 * r + 1
    mean_I = cv2.boxFilter(I, -1, (k, k), normalize=True)
    mean_p = cv2.boxFilter(p, -1, (k, k), normalize=True)
    mean_Ip = cv2.boxFilter(I * p, -1, (k, k), normalize=True)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, -1, (k, k), normalize=True)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (k, k), normalize=True)
    mean_b = cv2.boxFilter(b, -1, (k, k), normalize=True)
    return (mean_a * I + mean_b).astype(np.float32)


def flat_guided_boost(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    flat_strength: float = 0.6,
    radius: int = 4,
    eps: float = 1e-3,
    harden: float = 40.0,
) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    g = img if guide is None else np.ascontiguousarray(guide, dtype=np.float32)
    _, flat = _edge_flat_maps(g, temperature=8.0, harden=harden)
    smooth = guided_filter_gray(g, img, radius=radius, eps=eps)
    s = float(np.clip(flat_strength, 0.0, 1.0))
    return (img * (1.0 - s * flat) + smooth * (s * flat)).astype(np.float32)


def freq_split_fuse(
    sota: np.ndarray,
    edge: np.ndarray,
    sigma: float = 1.5,
    high_edge: float = 1.0,
    low_edge: float = 0.0,
) -> np.ndarray:
    """Gaussian freq split: LF from sota, HF from edge (MKPN / DualExNet HF residual)."""
    s = np.ascontiguousarray(sota, dtype=np.float32)
    e = np.ascontiguousarray(edge, dtype=np.float32)
    s_low = cv2.GaussianBlur(s, (0, 0), sigmaX=float(sigma))
    e_low = cv2.GaussianBlur(e, (0, 0), sigmaX=float(sigma))
    s_high = s - s_low
    e_high = e - e_low
    he = float(np.clip(high_edge, 0.0, 1.0))
    le = float(np.clip(low_edge, 0.0, 1.0))
    low = (1.0 - le) * s_low + le * e_low
    high = (1.0 - he) * s_high + he * e_high
    return (low + high).astype(np.float32)


def geometric_self_ensemble(image: np.ndarray, n_aug: int = 4) -> np.ndarray:
    """NTIRE-style geometric self-ensemble on a single DN image (cheap TTA)."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    outs = [img]
    # 0 identity already; add flips / 180
    outs.append(np.flipud(img))
    outs.append(np.fliplr(img))
    outs.append(np.flipud(np.fliplr(img)))
    if n_aug >= 8:
        rot = np.rot90(img, 1)
        outs.append(np.rot90(rot, 3))  # back → same as identity path after undo
        # Proper undo for 90/270 needs careful handling; skip odd rotations for non-square
        # Holdout is 1920x1200 — skip 90/270 to avoid shape issues
    n = min(int(n_aug), len(outs))
    stack = np.stack([outs[i] for i in range(n)], axis=0)
    # Undo flips when averaging
    fixed = [stack[0]]
    if n >= 2:
        fixed.append(np.flipud(stack[1]))
    if n >= 3:
        fixed.append(np.fliplr(stack[2]))
    if n >= 4:
        fixed.append(np.flipud(np.fliplr(stack[3])))
    return np.mean(np.stack(fixed, axis=0), axis=0).astype(np.float32)


def edge_unsharp(
    image: np.ndarray,
    amount: float = 0.35,
    sigma: float = 1.0,
    harden: float = 16.0,
) -> np.ndarray:
    """Mild edge-only unsharp to recover f10 edge_fidelity after flat denoise."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
    detail = img - blur
    emap, _ = _edge_flat_maps(img, temperature=8.0, harden=harden)
    return (img + float(amount) * emap * detail).astype(np.float32)


def flat_median_boost(
    image: np.ndarray,
    flat_strength: float = 0.5,
    ksize: int = 3,
    harden: float = 40.0,
) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    _, flat = _edge_flat_maps(img, temperature=8.0, harden=harden)
    k = int(ksize) | 1
    # medianBlur needs 8U or C1 32F with k<=5 typically; scale to 8U path for safety
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(u8, k).astype(np.float32) / 255.0
    s = float(np.clip(flat_strength, 0.0, 1.0))
    return (img * (1.0 - s * flat) + med * (s * flat)).astype(np.float32)


def flat_gaussian_boost(
    image: np.ndarray,
    flat_strength: float = 0.45,
    sigma: float = 1.2,
    harden: float = 40.0,
) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    _, flat = _edge_flat_maps(img, temperature=8.0, harden=harden)
    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
    s = float(np.clip(flat_strength, 0.0, 1.0))
    return (img * (1.0 - s * flat) + smooth * (s * flat)).astype(np.float32)


def morph_edge_blend(
    sota: np.ndarray,
    edge: np.ndarray,
    temperature: float = 8.0,
    harden: float = 16.0,
    dilate: int = 0,
    erode: int = 0,
) -> np.ndarray:
    """Morphologically thicken/thin Sobel gate before blend."""
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    emap = soft_edge_mask_dn(guide, temperature=temperature)
    if harden > 0.0:
        emap = 1.0 / (1.0 + np.exp(-float(harden) * (emap - 0.5)))
    u8 = np.clip(emap * 255.0, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if dilate > 0:
        u8 = cv2.dilate(u8, k, iterations=int(dilate))
    if erode > 0:
        u8 = cv2.erode(u8, k, iterations=int(erode))
    emap = u8.astype(np.float32) / 255.0
    return (emap * edge + (1.0 - emap) * sota).astype(np.float32)


def gen2_base_then(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    post: str = "none",
    post_strength: float = 0.0,
    **post_kw,
) -> np.ndarray:
    """Start from Gen2 deploy path, then optional non-bilat post."""
    low_fps, mid_fps = 1.0, 5.0
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=0.75, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0, edge_weight=1.0
        )
        if float(fps) <= mid_fps:
            out = flat_bilateral_boost(out, guide=out, flat_strength=0.7, harden=40.0)
    if post == "guided" and post_strength > 0:
        out = flat_guided_boost(out, flat_strength=post_strength, **post_kw)
    elif post == "median" and post_strength > 0:
        out = flat_median_boost(out, flat_strength=post_strength, **post_kw)
    elif post == "gauss" and post_strength > 0:
        out = flat_gaussian_boost(out, flat_strength=post_strength, **post_kw)
    elif post == "unsharp":
        out = edge_unsharp(out, amount=post_strength or 0.35, **post_kw)
    elif post == "tta":
        out = geometric_self_ensemble(out, n_aug=int(post_kw.get("n_aug", 4)))
    return out.astype(np.float32)
