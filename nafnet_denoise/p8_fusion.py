"""P8 fusion: Unsharp-Mask Guided Filtering + GH-GIF + multi-scale edge recovery.

Lit (beyond P7 residual/bilat saturation):
  - TIP'21 Unsharp Mask Guided Filtering (Shi): out = low(src) + a * high(guide)
  - arXiv'25 Gaussian highpass GIF: explicit structure transfer via Gaussian HP
  - Multi-scale unsharp (classic + NTIRE detail recovery)
"""

from __future__ import annotations

import cv2
import numpy as np

from .infer_ensemble import blend_sota_edgekd
from .multiband_fuse import _edge_flat_maps, flat_bilateral_boost
from .p7_fusion import edge_unsharp


def gaussian_highpass(image: np.ndarray, sigma: float) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    low = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
    return (img - low).astype(np.float32)


def umgf_fuse(
    base: np.ndarray,
    guide: np.ndarray,
    amount: float = 0.5,
    sigma: float = 1.4,
    edge_only: bool = True,
    harden: float = 16.0,
) -> np.ndarray:
    """Unsharp-Mask Guided Filtering (single-coeff structure transfer).

    out ≈ base + amount * edge_mask * (hp(guide) − hp(base))
    """
    b = np.ascontiguousarray(base, dtype=np.float32)
    g = np.ascontiguousarray(guide, dtype=np.float32)
    b_low = cv2.GaussianBlur(b, (0, 0), sigmaX=float(sigma))
    g_hp = gaussian_highpass(g, sigma)
    b_hp = b - b_low
    a = float(amount)
    if edge_only:
        emap, _ = _edge_flat_maps(0.5 * b + 0.5 * g, temperature=8.0, harden=harden)
        out = b + (a * emap) * (g_hp - b_hp)
    else:
        out = b_low + a * g_hp
    return out.astype(np.float32)


def gh_gif(
    src: np.ndarray,
    guide: np.ndarray,
    amount: float = 0.6,
    sigma: float = 1.5,
    edge_only: bool = True,
    harden: float = 16.0,
) -> np.ndarray:
    """Gaussian-highpass guided: smooth(src) + amount * highpass(guide)."""
    s = np.ascontiguousarray(src, dtype=np.float32)
    g = np.ascontiguousarray(guide, dtype=np.float32)
    s_low = cv2.GaussianBlur(s, (0, 0), sigmaX=float(sigma))
    g_hp = gaussian_highpass(g, sigma)
    a = float(amount)
    if edge_only:
        emap, _ = _edge_flat_maps(g, temperature=8.0, harden=harden)
        out = s_low + (a * emap) * g_hp
    else:
        out = s_low + a * g_hp
    return out.astype(np.float32)


def multi_scale_unsharp(
    image: np.ndarray,
    amounts: tuple[float, ...] = (0.1, 0.08, 0.05),
    sigmas: tuple[float, ...] = (0.8, 1.4, 2.5),
    harden: float = 16.0,
) -> np.ndarray:
    """Sum of edge-only unsharps at multiple scales (detail pyramid lite)."""
    out = np.ascontiguousarray(image, dtype=np.float32)
    emap, _ = _edge_flat_maps(out, temperature=8.0, harden=harden)
    for amt, sig in zip(amounts, sigmas):
        if amt <= 0:
            continue
        hp = gaussian_highpass(out, sig)
        out = out + float(amt) * emap * hp
    return out.astype(np.float32)


def clipped_unsharp(
    image: np.ndarray,
    amount: float = 0.15,
    sigma: float = 1.4,
    clip_pct: float = 98.0,
    harden: float = 16.0,
) -> np.ndarray:
    """Edge unsharp with highpass magnitude clipping (anti-halo)."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    hp = gaussian_highpass(img, sigma)
    thr = float(np.percentile(np.abs(hp), float(clip_pct))) + 1e-6
    hp = np.clip(hp, -thr, thr)
    emap, _ = _edge_flat_maps(img, temperature=8.0, harden=harden)
    return (img + float(amount) * emap * hp).astype(np.float32)


def flat_noise_proxy(
    img: np.ndarray,
    *,
    mode: str = "mad",
    flat_pct: float = 30.0,
) -> float:
    g = cv2.GaussianBlur(img, (0, 0), 1.2)
    hp = np.abs(img - g)
    sob = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        g, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    flat = sob < float(np.percentile(sob, float(flat_pct)))
    vals = hp[flat] if np.any(flat) else hp.ravel()
    mode_l = str(mode).lower()
    if mode_l == "mad":
        med = float(np.median(vals))
        return float(np.median(np.abs(vals - med)) * 1.4826)
    if mode_l == "p75":
        return float(np.percentile(vals, 75.0))
    if mode_l == "p90":
        return float(np.percentile(vals, 90.0))
    return float(np.std(vals))


def deploy_p68_noise(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    noise_lo: float = 0.002,
    noise_hi: float = 0.012,
    bilat_lo: float = 0.72,
    bilat_hi: float = 1.0,
    u_lo: float = 0.10,
    u_hi: float = 0.22,
    low_fps: float = 1.5,
    unsharp_sigma: float = 1.4,
    noise_mode: str = "mad",
    flat_pct: float = 30.0,
) -> np.ndarray:
    """P74/P71: robust flat-noise proxy gates bilat/unsharp (RPG-VST-lite)."""
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide, mode=noise_mode, flat_pct=flat_pct)
    t = (n - float(noise_lo)) / max(float(noise_hi) - float(noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    bilat = float(bilat_lo) + t * (float(bilat_hi) - float(bilat_lo))
    u = float(u_lo) + t * (float(u_hi) - float(u_lo))
    if float(fps) <= float(low_fps):
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0, edge_weight=1.0
        )
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=unsharp_sigma, harden=16.0)
    return out.astype(np.float32)


def gat_eui_inverse(z: np.ndarray, *, read_sigma: float = 0.0) -> np.ndarray:
    """Makitalo–Foi closed-form EUI of generalized Anscombe (TIP'12)."""
    w = (2.0 * np.maximum(z.astype(np.float32), 1e-6)).astype(np.float32)
    sig2 = float(read_sigma) ** 2
    y = 0.25 * w * w
    y += 0.25 * np.sqrt(1.5 + 2.0 * sig2) / w
    y -= (11.0 / 8.0 + sig2) / (w * w)
    y += (5.0 / 8.0) * np.sqrt(1.5 + 2.0 * sig2) / (w ** 3)
    y -= 0.125 + sig2
    return np.clip(y, 0.0, None).astype(np.float32)


def _hp_shrink(
    hp: np.ndarray,
    thr: float,
    *,
    kind: str = "soft",
    firm_ratio: float = 2.0,
) -> np.ndarray:
    """Soft / non-negative garrote / firm (Gao) shrinkage of highpass coeffs."""
    a = np.abs(hp)
    t = float(max(thr, 0.0))
    kind_l = str(kind).lower()
    if kind_l in ("garrote", "nng", "nn"):
        return np.where(a > t, hp * (1.0 - (t * t) / (a * a + 1e-12)), 0.0).astype(
            np.float32
        )
    if kind_l in ("firm", "semisoft"):
        t2 = float(max(t * float(firm_ratio), t + 1e-6))
        out = np.zeros_like(hp, dtype=np.float32)
        hi = a > t2
        mid = (a > t) & (~hi)
        out[hi] = hp[hi]
        out[mid] = (np.sign(hp[mid]) * t2 * (a[mid] - t) / (t2 - t)).astype(np.float32)
        return out
    return (np.sign(hp) * np.maximum(a - t, 0.0)).astype(np.float32)


def _block_js_hp(hp: np.ndarray, thr: float, block: int = 2) -> np.ndarray:
    """Cai BlockJS / James–Stein on L×L patches of highpass."""
    L = max(1, int(block))
    h, w = hp.shape
    out = hp.copy()
    t2 = float(thr) ** 2 * (L * L)
    for y in range(0, h - L + 1, L):
        for x in range(0, w - L + 1, L):
            patch = hp[y : y + L, x : x + L]
            s = float(np.sum(patch * patch))
            if s > t2:
                out[y : y + L, x : x + L] = patch * (1.0 - t2 / (s + 1e-12))
            else:
                out[y : y + L, x : x + L] = 0.0
    return out.astype(np.float32)


def _neigh_js_hp(hp: np.ndarray, thr: float, win: int = 3) -> np.ndarray:
    """Chen NeighShrink / overlapping James–Stein via local energy window."""
    L = max(3, int(win) | 1)
    e = cv2.boxFilter(hp * hp, ddepth=-1, ksize=(L, L), normalize=False)
    t2 = float(thr) ** 2 * float(L * L)
    scale = np.maximum(1.0 - t2 / (e + 1e-12), 0.0)
    return (hp * scale).astype(np.float32)


def _soft_shrink_map(hp: np.ndarray, thr_map: np.ndarray) -> np.ndarray:
    a = np.abs(hp)
    t = np.maximum(thr_map.astype(np.float32), 0.0)
    return (np.sign(hp) * np.maximum(a - t, 0.0)).astype(np.float32)


def _neighlevel_hp(
    hp: np.ndarray, low: np.ndarray, thr: float, *, alpha: float = 0.5
) -> np.ndarray:
    """Cho NeighLevel: parent low-pass modulates local threshold."""
    parent = np.abs(low).astype(np.float32)
    med = float(np.median(parent)) + 1e-6
    p = parent / med
    thr_map = float(thr) * (1.0 + float(alpha) * np.clip(p - 1.0, 0.0, None))
    return _soft_shrink_map(hp, thr_map)


def _bayesshrink_hp(hp: np.ndarray, noise: float, *, win: int = 5) -> np.ndarray:
    """Chang BayesShrink: T = σ_w² / σ_x with local variance estimates."""
    L = max(3, int(win) | 1)
    sw2 = float(max(noise, 1e-8)) ** 2
    var_y = cv2.boxFilter(hp * hp, ddepth=-1, ksize=(L, L), normalize=True)
    var_x = np.maximum(var_y - sw2, 0.0)
    sigma_x = np.sqrt(var_x + 1e-12)
    thr_map = sw2 / (sigma_x + 1e-12)
    return _soft_shrink_map(hp, thr_map)


def _bineigh_hp(
    hp: np.ndarray,
    low: np.ndarray,
    thr: float,
    *,
    neigh_win: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """MAP_NBShrink lite: BiShrink then NeighLevel on residual."""
    hp1 = _bishrink_hp(hp, thr)
    resid = hp - hp1
    hp2 = _neighlevel_hp(resid, low, thr * 0.75, alpha=alpha)
    return (hp1 + hp2).astype(np.float32)


def _bishrink_hp(hp: np.ndarray, thr: float) -> np.ndarray:
    """Sendur–Selesnick bivariate shrink; parent = 2× downsample of |hp|."""
    a = np.abs(hp).astype(np.float32)
    h, w = a.shape
    parent = cv2.resize(a, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
    parent = cv2.resize(parent, (w, h), interpolation=cv2.INTER_LINEAR)
    r = np.sqrt(hp * hp + parent * parent + 1e-12)
    t = float(max(thr, 0.0)) * np.sqrt(3.0)
    scale = np.maximum(1.0 - t / r, 0.0)
    return (hp * scale).astype(np.float32)


def anscombe_flat_shrink(
    img: np.ndarray,
    *,
    strength: float = 0.25,
    k_mad: float = 0.5,
    sigma: float = 2.4,
    flat_pct: float = 50.0,
    harden: float = 40.0,
    read_sigma: float = 0.0,
    use_eui: bool = False,
    em_bias: float = 0.0,
    shrink_kind: str = "soft",
    firm_ratio: float = 2.0,
    block_size: int = 0,
    neigh_win: int = 0,
    bishrink: bool = False,
    shrink_mode: str = "",
    neighlevel_alpha: float = 0.5,
    bayes_win: int = 5,
) -> np.ndarray:
    """GALOSH/RPG-VST lite: soft/garrote/firm/block-JS/Neigh/Bi/NeighLevel/Bayes HP on flats."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    if strength <= 0:
        return x
    c = 3.0 / 8.0
    bias = float(max(em_bias, 0.0))
    z = np.sqrt(np.clip(x, 0.0, None) + c + float(read_sigma) ** 2 + bias)
    low = cv2.GaussianBlur(z, (0, 0), sigmaX=float(sigma))
    hp = z - low
    mad = flat_noise_proxy(x, mode="mad", flat_pct=flat_pct)
    thr = float(k_mad) * max(mad, 1e-6)
    mode = str(shrink_mode).lower()
    if mode == "neighlevel":
        hp_s = _neighlevel_hp(hp, low, thr, alpha=float(neighlevel_alpha))
    elif mode == "bayes":
        hp_s = _bayesshrink_hp(hp, mad, win=int(bayes_win))
    elif mode == "bineigh":
        hp_s = _bineigh_hp(
            hp, low, thr, neigh_win=max(int(neigh_win), 3), alpha=float(neighlevel_alpha)
        )
    elif mode == "bishrink" or bool(bishrink):
        hp_s = _bishrink_hp(hp, thr)
    elif int(neigh_win) >= 3:
        hp_s = _neigh_js_hp(hp, thr, win=int(neigh_win))
    elif int(block_size) >= 2:
        hp_s = _block_js_hp(hp, thr, block=int(block_size))
    else:
        hp_s = _hp_shrink(hp, thr, kind=shrink_kind, firm_ratio=firm_ratio)
    z2 = low + hp_s
    if use_eui:
        y = gat_eui_inverse(z2, read_sigma=read_sigma)
        if bias > 0:
            y = np.clip(y - bias, 0.0, None).astype(np.float32)
    else:
        y = np.clip(z2 * z2 - c - bias, 0.0, None).astype(np.float32)
    _, flat = _edge_flat_maps(x, temperature=8.0, harden=harden)
    if flat_pct > 0:
        sob = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            x, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = ((flat > 0.5) & (sob < np.percentile(sob, flat_pct))).astype(np.float32)
    s = float(np.clip(strength, 0.0, 1.0))
    return (x * (1.0 - s * flat) + y * (s * flat)).astype(np.float32)


def deploy_p98(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    ans_strength: float = 0.25,
    ans_k_mad: float = 0.5,
    ans_sigma: float = 2.4,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P102/P98: P74 MAD noise gate + fixed Anscombe flat soft-threshold."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    return anscombe_flat_shrink(
        out,
        strength=ans_strength,
        k_mad=ans_k_mad,
        sigma=ans_sigma,
        flat_pct=ans_flat_pct,
        harden=ans_harden,
    )


def deploy_p103(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    ans_s_lo: float = 0.1,
    ans_s_hi: float = 0.3,
    ans_noise_lo: float = 0.002,
    ans_noise_hi: float = 0.012,
    ans_k_mad: float = 0.75,
    ans_sigma: float = 2.8,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P103: MAD-gated Anscombe strength (DES ≈ 0.9506)."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide, mode="mad", flat_pct=30.0)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    return anscombe_flat_shrink(
        out,
        strength=s,
        k_mad=ans_k_mad,
        sigma=ans_sigma,
        flat_pct=ans_flat_pct,
        harden=ans_harden,
    )


def cycle_spin_anscombe(
    img: np.ndarray,
    *,
    max_shift: int = 1,
    **ans_kw,
) -> np.ndarray:
    """Coifman–Donoho cycle-spin of Anscombe shrink (shifts 0..max_shift)."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    ms = max(0, int(max_shift))
    acc = np.zeros_like(x)
    n = 0
    for dy in range(ms + 1):
        for dx in range(ms + 1):
            shifted = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
            out = anscombe_flat_shrink(shifted, **ans_kw)
            acc += np.roll(np.roll(out, -dy, axis=0), -dx, axis=1)
            n += 1
    return (acc / max(n, 1)).astype(np.float32)


def interscale_haar_shrink(
    img: np.ndarray,
    *,
    strength: float = 0.2,
    k_mad: float = 1.0,
    flat_pct: float = 50.0,
    harden: float = 40.0,
) -> np.ndarray:
    """Luisier interscale lite: parent Haar band estimates child threshold."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    if strength <= 0:
        return x
    h0, w0 = x.shape
    ph, pw = h0 % 2, w0 % 2
    sh = np.pad(x, ((0, ph), (0, pw)), mode="reflect") if (ph or pw) else x
    a, h, v, d = _haar_dwt2(sh)
    a2, h2, v2, d2 = _haar_dwt2(a)
    vals = np.concatenate([h2.ravel(), v2.ravel(), d2.ravel()])
    med = float(np.median(vals))
    mad_p = float(np.median(np.abs(vals - med)) * 1.4826)
    thr = float(k_mad) * max(mad_p, 1e-6)
    h = np.sign(h) * np.maximum(np.abs(h) - thr, 0.0)
    v = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
    d = np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)
    rec = _haar_idwt2(a, h, v, d, sh.shape)
    if ph or pw:
        rec = rec[:h0, :w0]
    _, flat = _edge_flat_maps(x, temperature=8.0, harden=harden)
    if flat_pct > 0:
        sob = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            x, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = ((flat > 0.5) & (sob < np.percentile(sob, flat_pct))).astype(np.float32)
    s = float(np.clip(strength, 0.0, 1.0))
    return (x * (1.0 - s * flat) + rec * (s * flat)).astype(np.float32)


def deploy_p103_cycspin(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    max_shift: int = 1,
    ans_s_lo: float = 0.1,
    ans_s_hi: float = 0.3,
    ans_noise_lo: float = 0.002,
    ans_noise_hi: float = 0.012,
    ans_k_mad: float = 0.75,
    ans_sigma: float = 2.8,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P110: cycle-spin Anscombe after P74 MAD gate."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    n = flat_noise_proxy(guide, mode="mad", flat_pct=30.0)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    return cycle_spin_anscombe(
        out,
        max_shift=max_shift,
        strength=s,
        k_mad=ans_k_mad,
        sigma=ans_sigma,
        flat_pct=ans_flat_pct,
        harden=ans_harden,
    )


def deploy_p103_cne(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    residual_scale: float = 0.5,
    ans_s_lo: float = 0.12,
    ans_s_hi: float = 0.4,
    ans_noise_lo: float = 0.0008,
    ans_noise_hi: float = 0.006,
    ans_k_mad: float = 0.9,
    ans_sigma: float = 2.4,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P111: YOND CNE — MAD of (P74 − sota) residual gates Anscombe."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    resid = np.abs(out - sota).astype(np.float32)
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(residual_scale)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    return anscombe_flat_shrink(
        out,
        strength=s,
        k_mad=ans_k_mad,
        sigma=ans_sigma,
        flat_pct=ans_flat_pct,
        harden=ans_harden,
    )


def deploy_p114(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    max_shift: int = 1,
    residual_scale: float = 0.4,
    ans_s_lo: float = 0.12,
    ans_s_hi: float = 0.4,
    ans_noise_lo: float = 0.0008,
    ans_noise_hi: float = 0.006,
    ans_k_mad: float = 1.0,
    ans_sigma: float = 2.8,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P114: CNE residual-MAD + cycle-spin Anscombe (DES ≈ 0.9528)."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    resid = np.abs(out - sota).astype(np.float32)
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(residual_scale)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    return cycle_spin_anscombe(
        out,
        max_shift=int(max_shift),
        strength=s,
        k_mad=ans_k_mad,
        sigma=ans_sigma,
        flat_pct=ans_flat_pct,
        harden=ans_harden,
    )


def deploy_p120(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    n_iters: int = 2,
    thr_decay: float = 0.5,
    max_shift: int = 1,
    residual_scale: float = 0.4,
    ans_s_lo: float = 0.12,
    ans_s_hi: float = 0.4,
    ans_noise_lo: float = 0.0008,
    ans_noise_hi: float = 0.006,
    ans_k_mad: float = 1.2,
    ans_sigma: float = 2.8,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P120: recursive cycle-spin Anscombe (DES ≈ 0.9556)."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    resid = np.abs(out - sota).astype(np.float32)
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(residual_scale)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    k = float(ans_k_mad)
    x = out
    for _ in range(int(n_iters)):
        x = cycle_spin_anscombe(
            x,
            max_shift=int(max_shift),
            strength=s,
            k_mad=k,
            sigma=ans_sigma,
            flat_pct=ans_flat_pct,
            harden=ans_harden,
        )
        k *= float(thr_decay)
    return x.astype(np.float32)


def deploy_p126(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    read_sigma: float = 0.0,
    n_iters: int = 3,
    thr_decay: float = 0.7,
    max_shift: int = 1,
    residual_scale: float = 0.4,
    ans_s_lo: float = 0.12,
    ans_s_hi: float = 0.4,
    ans_noise_lo: float = 0.0008,
    ans_noise_hi: float = 0.006,
    ans_k_mad: float = 1.0,
    ans_sigma: float = 2.4,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    **noise_kw,
) -> np.ndarray:
    """P126: recursive CNE-spin + Makitalo–Foi EUI inverse (DES ≈ 0.9565)."""
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    resid = np.abs(out - sota).astype(np.float32)
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(residual_scale)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    k = float(ans_k_mad)
    rs = float(read_sigma)
    x = out
    for _ in range(int(n_iters)):
        x = cycle_spin_anscombe(
            x,
            max_shift=int(max_shift),
            strength=s,
            k_mad=k,
            sigma=ans_sigma,
            flat_pct=ans_flat_pct,
            harden=ans_harden,
            read_sigma=rs,
            use_eui=True,
        )
        k *= float(thr_decay)
    return x.astype(np.float32)


def deploy_p127(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    is_strength: float = 0.15,
    is_k_mad: float = 1.0,
    is_flat_pct: float = 50.0,
    **p120_kw,
) -> np.ndarray:
    """P127: P120 + Luisier interscale Haar pass on flats."""
    out = deploy_p120(sota, edge, fps, **p120_kw)
    return interscale_haar_shrink(
        out,
        strength=is_strength,
        k_mad=is_k_mad,
        flat_pct=is_flat_pct,
    )


def deploy_p128(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    sc_scale: float = 1.0,
    **p126_kw,
) -> np.ndarray:
    """P128: self-calibrated read_sigma from flat MAD (SCVST arXiv'24 lite)."""
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    mad = flat_noise_proxy(guide, mode="mad", flat_pct=30.0)
    rs = float(np.clip(mad * float(sc_scale), 0.005, 0.08))
    return deploy_p126(sota, edge, fps, read_sigma=rs, **p126_kw)


def deploy_p129(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    is_strength: float = 0.12,
    is_k_mad: float = 1.0,
    read_sigma: float = 0.02,
    **p120_kw,
) -> np.ndarray:
    """P129: EUI recursive spin + interscale (GAT pipeline full lite)."""
    out = deploy_p126(
        sota,
        edge,
        fps,
        read_sigma=read_sigma,
        **{k: v for k, v in p120_kw.items() if k != "read_sigma"},
    )
    return interscale_haar_shrink(
        out, strength=is_strength, k_mad=is_k_mad, flat_pct=50.0
    )


def _stabilized_flat_mad(
    img: np.ndarray, *, sigma: float = 2.4, flat_pct: float = 50.0
) -> float:
    """RPG-VST-lite σz: MAD of Anscombe-domain HP on flats."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    c = 3.0 / 8.0
    z = np.sqrt(np.clip(x, 0.0, None) + c)
    low = cv2.GaussianBlur(z, (0, 0), sigmaX=float(sigma))
    hp = np.abs(z - low)
    sob = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        x, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    flat = sob < float(np.percentile(sob, float(flat_pct)))
    vals = hp[flat] if np.any(flat) else hp.ravel()
    med = float(np.median(vals))
    return float(np.median(np.abs(vals - med)) * 1.4826)


def _recursive_cne_spin_core(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    use_eui: bool = True,
    read_sigma: float = 0.0,
    em_bias: float = 0.0,
    n_iters: int = 3,
    thr_decay: float = 0.7,
    max_shift: int = 1,
    residual_scale: float = 0.4,
    ans_s_lo: float = 0.12,
    ans_s_hi: float = 0.4,
    ans_noise_lo: float = 0.0008,
    ans_noise_hi: float = 0.006,
    ans_k_mad: float = 1.0,
    ans_sigma: float = 2.4,
    ans_flat_pct: float = 50.0,
    ans_harden: float = 40.0,
    shrink_kind: str = "soft",
    firm_ratio: float = 2.0,
    block_size: int = 0,
    neigh_win: int = 0,
    bishrink: bool = False,
    shrink_mode: str = "",
    neighlevel_alpha: float = 0.5,
    bayes_win: int = 5,
    **noise_kw,
) -> np.ndarray:
    out = deploy_p68_noise(sota, edge, fps, **noise_kw)
    resid = np.abs(out - sota).astype(np.float32)
    n = flat_noise_proxy(resid, mode="mad", flat_pct=30.0) * float(residual_scale)
    t = (n - float(ans_noise_lo)) / max(float(ans_noise_hi) - float(ans_noise_lo), 1e-6)
    t = float(np.clip(t, 0.0, 1.0))
    s = float(ans_s_lo) + t * (float(ans_s_hi) - float(ans_s_lo))
    k = float(ans_k_mad)
    x = out
    for _ in range(int(n_iters)):
        x = cycle_spin_anscombe(
            x,
            max_shift=int(max_shift),
            strength=s,
            k_mad=k,
            sigma=ans_sigma,
            flat_pct=ans_flat_pct,
            harden=ans_harden,
            read_sigma=float(read_sigma),
            use_eui=bool(use_eui),
            em_bias=float(em_bias),
            shrink_kind=str(shrink_kind),
            firm_ratio=float(firm_ratio),
            block_size=int(block_size),
            neigh_win=int(neigh_win),
            bishrink=bool(bishrink),
            shrink_mode=str(shrink_mode),
            neighlevel_alpha=float(neighlevel_alpha),
            bayes_win=int(bayes_win),
        )
        k *= float(thr_decay)
    return x.astype(np.float32)


def deploy_p132(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    target_sz: float = 0.02,
    **p126_kw,
) -> np.ndarray:
    """P132: RPG-VST σz gate between algebraic vs EUI recursive spin."""
    eui = _recursive_cne_spin_core(sota, edge, fps, use_eui=True, **p126_kw)
    alg = _recursive_cne_spin_core(sota, edge, fps, use_eui=False, **p126_kw)
    sig = float(p126_kw.get("ans_sigma", 2.4))
    se = _stabilized_flat_mad(eui, sigma=sig)
    sa = _stabilized_flat_mad(alg, sigma=sig)
    tgt = float(target_sz)
    return eui if abs(se - tgt) <= abs(sa - tgt) else alg


def deploy_p133(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    em_scale: float = 0.5,
    **p126_kw,
) -> np.ndarray:
    """P133: YOND EM-VST bias offset on EUI recursive spin."""
    out0 = deploy_p68_noise(sota, edge, fps)
    resid = np.abs(out0 - sota).astype(np.float32)
    bias = float(em_scale) * flat_noise_proxy(resid, mode="mad", flat_pct=30.0)
    return _recursive_cne_spin_core(
        sota, edge, fps, use_eui=True, em_bias=bias, **p126_kw
    )


def deploy_p134(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **p126_kw,
) -> np.ndarray:
    """P134: SURE-lite pick of k_mad on flats after recursive EUI (DES ≈ 0.9572)."""
    p126_kw = dict(p126_kw)
    p126_kw.pop("ans_k_mad", None)
    p126_kw.setdefault("n_iters", n_iters)
    p126_kw.setdefault("thr_decay", thr_decay)
    p126_kw.setdefault("residual_scale", residual_scale)
    p126_kw.setdefault("ans_sigma", ans_sigma)
    best = None
    best_score = 1e30
    base = deploy_p68_noise(sota, edge, fps)
    mad0 = flat_noise_proxy(base, mode="mad", flat_pct=50.0)
    for k in k_list:
        cand = _recursive_cne_spin_core(
            sota, edge, fps, use_eui=True, ans_k_mad=float(k), **p126_kw
        )
        resid = np.abs(cand - base)
        _, flat = _edge_flat_maps(base, temperature=8.0, harden=40.0)
        sob = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            base, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        mask = (flat > 0.5) & (sob < np.percentile(sob, 50.0))
        vals = resid[mask] if np.any(mask) else resid.ravel()
        risk = float(np.mean(vals * vals))
        thr = float(k) * max(mad0, 1e-6)
        keep = float(np.mean(vals > thr)) if vals.size else 0.0
        score = risk + 2.0 * (mad0**2) * keep
        if score < best_score:
            best_score = score
            best = cand
    return best.astype(np.float32) if best is not None else deploy_p126(sota, edge, fps)


def deploy_p135(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    dual_strength: float = 0.35,
    dual_k: float = 1.0,
    dual_sigma: float = 2.4,
    **p126_kw,
) -> np.ndarray:
    """P135: DualDn-lite — intensity EUI + residual-domain Anscombe fuse."""
    img = _recursive_cne_spin_core(sota, edge, fps, use_eui=True, **p126_kw)
    resid = (img - sota).astype(np.float32)
    offset = float(np.median(resid))
    r_pos = np.clip(resid - offset + 0.5, 0.0, None)
    r_dn = anscombe_flat_shrink(
        r_pos,
        strength=float(dual_strength),
        k_mad=float(dual_k),
        sigma=float(dual_sigma),
        use_eui=True,
    )
    resid2 = (r_dn - 0.5 + offset).astype(np.float32)
    _, flat = _edge_flat_maps(img, temperature=8.0, harden=40.0)
    out = img * (1.0 - flat) + (sota + resid2) * flat
    return out.astype(np.float32)


def _sure_pick_k(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...],
    risk_mode: str = "sure",
    **core_kw,
) -> np.ndarray:
    """Pick k_mad by SURE or PURE-lite on flats."""
    core_kw = dict(core_kw)
    core_kw.pop("ans_k_mad", None)
    best = None
    best_score = 1e30
    base = deploy_p68_noise(sota, edge, fps)
    mad0 = flat_noise_proxy(base, mode="mad", flat_pct=50.0)
    _, flat = _edge_flat_maps(base, temperature=8.0, harden=40.0)
    sob = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        base, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    mask = (flat > 0.5) & (sob < np.percentile(sob, 50.0))
    for k in k_list:
        cand = _recursive_cne_spin_core(
            sota, edge, fps, use_eui=True, ans_k_mad=float(k), **core_kw
        )
        resid = np.abs(cand - base)
        vals = resid[mask] if np.any(mask) else resid.ravel()
        thr = float(k) * max(mad0, 1e-6)
        keep = float(np.mean(vals > thr)) if vals.size else 0.0
        if str(risk_mode).lower() == "pure":
            w = np.clip(base[mask] if np.any(mask) else base.ravel(), 1e-4, None)
            r2 = vals * vals
            risk = float(np.mean(r2 / w))
            score = risk + 2.0 * (mad0**2) * keep
        else:
            risk = float(np.mean(vals * vals))
            score = risk + 2.0 * (mad0**2) * keep
        if score < best_score:
            best_score = score
            best = cand
    return best.astype(np.float32) if best is not None else deploy_p134(sota, edge, fps)


def _sure_flat_risk(base: np.ndarray, cand: np.ndarray, mad0: float, thr: float) -> float:
    _, flat = _edge_flat_maps(base, temperature=8.0, harden=40.0)
    sob = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
        base, cv2.CV_32F, 0, 1, ksize=3
    ) ** 2
    mask = (flat > 0.5) & (sob < np.percentile(sob, 50.0))
    resid = np.abs(cand - base)
    vals = resid[mask] if np.any(mask) else resid.ravel()
    keep = float(np.mean(vals > thr)) if vals.size else 0.0
    risk = float(np.mean(vals * vals))
    return risk + 2.0 * (mad0**2) * keep


def _sure_pick_mode(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    mode_list: tuple[str, ...],
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    **core_kw,
) -> np.ndarray:
    """Pick shrink mode by SURE-lite on flats (each mode uses SURE-k internally)."""
    best = None
    best_score = 1e30
    base = deploy_p68_noise(sota, edge, fps)
    mad0 = flat_noise_proxy(base, mode="mad", flat_pct=50.0)
    for mode in mode_list:
        kw = dict(core_kw)
        kw.pop("shrink_mode", None)
        kw.pop("bishrink", None)
        if str(mode).lower() == "bishrink":
            kw["bishrink"] = True
        else:
            kw["shrink_mode"] = str(mode)
        cand = _sure_pick_k(sota, edge, fps, k_list=k_list, use_eui=True, **kw)
        thr = float(k_list[len(k_list) // 2]) * max(mad0, 1e-6)
        score = _sure_flat_risk(base, cand, mad0, thr)
        if score < best_score:
            best_score = score
            best = cand
    return best.astype(np.float32) if best is not None else deploy_p150(sota, edge, fps)


def deploy_p156(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    neighlevel_alpha: float = 0.5,
    n_iters: int = 5,
    thr_decay: float = 0.7,
    residual_scale: float = 0.3,
    ans_sigma: float = 1.8,
    **_kw,
) -> np.ndarray:
    """P156: Cho NeighLevel parent-scaled threshold + SURE-k."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        shrink_mode="neighlevel",
        neighlevel_alpha=float(neighlevel_alpha),
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p157(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    bayes_win: int = 5,
    n_iters: int = 5,
    thr_decay: float = 0.7,
    residual_scale: float = 0.3,
    ans_sigma: float = 1.8,
    **_kw,
) -> np.ndarray:
    """P157: Chang BayesShrink adaptive threshold + SURE-k."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        shrink_mode="bayes",
        bayes_win=int(bayes_win),
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p158(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    neighlevel_alpha: float = 0.5,
    neigh_win: int = 3,
    n_iters: int = 5,
    thr_decay: float = 0.7,
    residual_scale: float = 0.3,
    ans_sigma: float = 1.8,
    **_kw,
) -> np.ndarray:
    """P158: BiShrink + NeighLevel residual cascade (MAP_NBShrink lite) + SURE-k."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        shrink_mode="bineigh",
        neighlevel_alpha=float(neighlevel_alpha),
        neigh_win=int(neigh_win),
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p159(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    mode_list: tuple[str, ...] = ("bishrink", "neighlevel", "bayes", "bineigh"),
    n_iters: int = 5,
    thr_decay: float = 0.7,
    residual_scale: float = 0.3,
    ans_sigma: float = 1.8,
    **_kw,
) -> np.ndarray:
    """P159: SURE pick among BiShrink / NeighLevel / Bayes / BiNeigh."""
    return _sure_pick_mode(
        sota,
        edge,
        fps,
        mode_list=mode_list,
        k_list=k_list,
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p140(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P140: non-negative garrote + SURE-k (Gao/Breiman WaveShrink)."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        shrink_kind="garrote",
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p141(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    firm_ratio: float = 2.0,
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P141: firm/semisoft shrinkage + SURE-k (Gao firm WaveShrink)."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        shrink_kind="firm",
        firm_ratio=firm_ratio,
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p142(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    block_size: int = 2,
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P142: Cai BlockJS / James–Stein patches + SURE-k."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        block_size=int(block_size),
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p143(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P143: PURE-lite intensity-weighted risk for k pick (Luisier PURE)."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="pure",
        shrink_kind="soft",
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p148(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    neigh_win: int = 3,
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P148: overlapping NeighShrink James–Stein (Chen NeighSure-lite)."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        neigh_win=int(neigh_win),
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p149(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    neigh_win: int = 5,
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P149: wider NeighShrink window (Chen 5×5 NeighSure)."""
    return deploy_p148(
        sota,
        edge,
        fps,
        k_list=k_list,
        neigh_win=int(neigh_win),
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p150(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    k_list: tuple[float, ...] = (0.8, 1.0, 1.2),
    n_iters: int = 5,
    thr_decay: float = 0.7,
    residual_scale: float = 0.3,
    ans_sigma: float = 1.8,
    **_kw,
) -> np.ndarray:
    """P150: Sendur–Selesnick bivariate shrink + SURE-k (DES ≈ 0.9589)."""
    return _sure_pick_k(
        sota,
        edge,
        fps,
        k_list=k_list,
        risk_mode="sure",
        bishrink=True,
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
    )


def deploy_p151(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    *,
    mix: float = 0.5,
    k_mad: float = 1.0,
    neigh_win: int = 3,
    n_iters: int = 4,
    thr_decay: float = 0.7,
    residual_scale: float = 0.35,
    ans_sigma: float = 2.0,
    **_kw,
) -> np.ndarray:
    """P151: SURE-LET mix of tiled BlockJS and overlapping NeighShrink."""
    kw = dict(
        n_iters=n_iters,
        thr_decay=thr_decay,
        residual_scale=residual_scale,
        ans_sigma=ans_sigma,
        ans_k_mad=float(k_mad),
        use_eui=True,
    )
    a = _recursive_cne_spin_core(sota, edge, fps, block_size=2, **kw)
    b = _recursive_cne_spin_core(sota, edge, fps, neigh_win=int(neigh_win), **kw)
    w = float(np.clip(mix, 0.0, 1.0))
    return (w * a + (1.0 - w) * b).astype(np.float32)


def anscombe_flat_shrink_map(
    img: np.ndarray,
    strength_map: np.ndarray,
    *,
    k_mad: float = 0.75,
    sigma: float = 2.8,
    flat_pct: float = 50.0,
    harden: float = 40.0,
) -> np.ndarray:
    """Anscombe shrink with spatially varying strength (YOND SNR-map lite)."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    sm = np.clip(strength_map.astype(np.float32), 0.0, 1.0)
    c = 3.0 / 8.0
    z = np.sqrt(np.clip(x, 0.0, None) + c)
    low = cv2.GaussianBlur(z, (0, 0), sigmaX=float(sigma))
    hp = z - low
    mad = flat_noise_proxy(x, mode="mad", flat_pct=flat_pct)
    thr = float(k_mad) * max(mad, 1e-6)
    hp_s = np.sign(hp) * np.maximum(np.abs(hp) - thr, 0.0)
    z2 = low + hp_s
    y = np.clip(z2 * z2 - c, 0.0, None).astype(np.float32)
    _, flat = _edge_flat_maps(x, temperature=8.0, harden=harden)
    if flat_pct > 0:
        sob = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            x, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = ((flat > 0.5) & (sob < np.percentile(sob, flat_pct))).astype(np.float32)
    mix = sm * flat
    return (x * (1.0 - mix) + y * mix).astype(np.float32)


def _haar_dwt2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = (x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2]) * 0.5
    h = (x[0::2, 0::2] + x[0::2, 1::2] - x[1::2, 0::2] - x[1::2, 1::2]) * 0.5
    v = (x[0::2, 0::2] - x[0::2, 1::2] + x[1::2, 0::2] - x[1::2, 1::2]) * 0.5
    d = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) * 0.5
    return a, h, v, d


def _haar_idwt2(
    a: np.ndarray, h: np.ndarray, v: np.ndarray, d: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    hh, ww = a.shape
    out = np.zeros((hh * 2, ww * 2), dtype=np.float32)
    out[0::2, 0::2] = (a + h + v + d) * 0.5
    out[0::2, 1::2] = (a + h - v - d) * 0.5
    out[1::2, 0::2] = (a - h + v - d) * 0.5
    out[1::2, 1::2] = (a - h - v + d) * 0.5
    return out[: shape[0], : shape[1]]


def haar_cycle_spin_shrink(
    img: np.ndarray,
    *,
    strength: float,
    k_mad: float,
    max_shift: int = 1,
    flat_pct: float = 50.0,
    harden: float = 40.0,
) -> np.ndarray:
    """Haar DWT soft-threshold with 2D cycle-spin on flats (Parhi/Coifman)."""
    x = np.ascontiguousarray(img, dtype=np.float32)
    if strength <= 0:
        return x
    mad = flat_noise_proxy(x, mode="mad", flat_pct=flat_pct)
    thr = float(k_mad) * max(mad, 1e-6)
    _, flat = _edge_flat_maps(x, temperature=8.0, harden=harden)
    if flat_pct > 0:
        sob = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(
            x, cv2.CV_32F, 0, 1, ksize=3
        ) ** 2
        flat = ((flat > 0.5) & (sob < np.percentile(sob, flat_pct))).astype(np.float32)
    ms = max(0, int(max_shift))
    acc = np.zeros_like(x)
    n = 0
    h0, w0 = x.shape
    ph, pw = h0 % 2, w0 % 2
    for dy in range(ms + 1):
        for dx in range(ms + 1):
            sh = np.roll(np.roll(x, dy, 0), dx, 1)
            if ph or pw:
                sh = np.pad(sh, ((0, ph), (0, pw)), mode="reflect")
            a, hh, vv, dd = _haar_dwt2(sh)
            hh = np.sign(hh) * np.maximum(np.abs(hh) - thr, 0.0)
            vv = np.sign(vv) * np.maximum(np.abs(vv) - thr, 0.0)
            dd = np.sign(dd) * np.maximum(np.abs(dd) - thr, 0.0)
            rec = _haar_idwt2(a, hh, vv, dd, sh.shape)
            if ph or pw:
                rec = rec[:h0, :w0]
            acc += np.roll(np.roll(rec, -dy, 0), -dx, 1)
            n += 1
    rec = acc / max(n, 1)
    s = float(np.clip(strength, 0.0, 1.0))
    return (x * (1.0 - s * flat) + rec * (s * flat)).astype(np.float32)


def deploy_p64_exposure(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    exposure_ms: float,
    *,
    t_dark: float = 3.0,
    t_bright: float = 15.0,
    bilat_d: float = 0.95,
    bilat_m: float = 0.88,
    bilat_b: float = 0.80,
    u_d: float = 0.22,
    u_m: float = 0.15,
    u_b: float = 0.14,
    low_fps: float = 1.5,
    mid_fps: float = 5.0,
    high_bilat_scale: float = 1.0,
    high_u_scale: float = 1.1,
    unsharp_sigma: float = 1.4,
) -> np.ndarray:
    """P67/P64: exposure_ms-conditioned bilat/unsharp + fps gate/scales."""
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    exp = float(exposure_ms)
    if exp <= float(t_dark):
        bilat, u = float(bilat_d), float(u_d)
    elif exp >= float(t_bright):
        bilat, u = float(bilat_b), float(u_b)
    else:
        bilat, u = float(bilat_m), float(u_m)
    if float(fps) > float(mid_fps):
        bilat *= float(high_bilat_scale)
        u *= float(high_u_scale)

    if float(fps) <= float(low_fps):
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0, edge_weight=1.0
        )
        if bilat > 0:
            out = flat_bilateral_boost(out, guide=out, flat_strength=bilat, harden=40.0)
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=unsharp_sigma, harden=16.0)
    return out.astype(np.float32)


def deploy_p7_base(
    sota: np.ndarray,
    edge: np.ndarray,
    fps: float,
    unsharp_amount: float = 0.15,
    unsharp_sigma: float = 1.4,
    unsharp_amount_low: float | None = 0.20,
    unsharp_amount_mid: float | None = 0.16,
    unsharp_amount_high: float | None = 0.12,
    low_bilat: float = 0.9,
    mid_bilat: float = 0.85,
    low_fps: float = 1.5,
    mid_fps: float = 10.0,
    exposure_ms: float | None = None,
) -> np.ndarray:
    """Current deploy path (P64 exposure schedule when exposure_ms given; else P61 fps)."""
    if exposure_ms is not None:
        return deploy_p64_exposure(sota, edge, fps, float(exposure_ms))
    guide = (0.5 * sota + 0.5 * edge).astype(np.float32)
    if float(fps) <= low_fps:
        out = sota.copy()
        out = flat_bilateral_boost(out, guide=out, flat_strength=low_bilat, harden=40.0)
        u = float(unsharp_amount_low if unsharp_amount_low is not None else unsharp_amount)
    else:
        out, _ = blend_sota_edgekd(
            sota, edge, guide_dn=guide, temperature=8.0, harden=16.0, edge_weight=1.0
        )
        if float(fps) <= mid_fps:
            out = flat_bilateral_boost(
                out, guide=out, flat_strength=mid_bilat, harden=40.0
            )
            u = float(
                unsharp_amount_mid if unsharp_amount_mid is not None else unsharp_amount
            )
        else:
            u = float(
                unsharp_amount_high
                if unsharp_amount_high is not None
                else unsharp_amount
            )
    if u > 0:
        out = edge_unsharp(out, amount=u, sigma=unsharp_sigma, harden=16.0)
    return out.astype(np.float32)
