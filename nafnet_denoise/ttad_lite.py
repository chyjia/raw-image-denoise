"""TTAD pixel-bank lite (ICCV'25): flat HF pull via patch-similarity bank (fast)."""

from __future__ import annotations

import cv2
import numpy as np

from .multiband_fuse import _edge_flat_maps


def ttad_flat_hf_pull(
    base: np.ndarray,
    guide: np.ndarray,
    *,
    strength: float = 0.35,
    patch: int = 5,
    search: int = 7,
    topk: int = 8,
    sigma_hp: float = 1.2,
    scale: float = 0.25,
    stride: int = 2,
    harden: float = 40.0,
) -> np.ndarray:
    """On flat regions, denoise HP via top-k similar-patch centers (coarse grid)."""
    b = np.ascontiguousarray(base, dtype=np.float32)
    g = np.ascontiguousarray(guide, dtype=np.float32)
    if strength <= 0:
        return b
    _, flat = _edge_flat_maps(g, temperature=8.0, harden=harden)
    low = cv2.GaussianBlur(b, (0, 0), sigmaX=float(sigma_hp))
    hp = b - low

    sc = float(np.clip(scale, 0.2, 1.0))
    sh = (max(int(b.shape[0] * sc), 8), max(int(b.shape[1] * sc), 8))
    gs = cv2.resize(g, sh[::-1], interpolation=cv2.INTER_AREA)
    hps = cv2.resize(hp, sh[::-1], interpolation=cv2.INTER_AREA)
    flats = cv2.resize(flat, sh[::-1], interpolation=cv2.INTER_AREA)

    p = max(int(patch) | 1, 3)
    r = max(int(search) // 2, 1)
    pad = p // 2 + r
    gs_p = cv2.copyMakeBorder(gs, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    hp_p = cv2.copyMakeBorder(hps, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    h, w = hps.shape
    corr = np.zeros_like(hps, dtype=np.float32)
    st = max(int(stride), 1)
    k = max(int(topk), 1)

    for y in range(0, h, st):
        y0 = y + pad
        for x in range(0, w, st):
            if flats[y, x] < 0.5:
                continue
            ref = gs_p[y0 - p // 2 : y0 + p // 2 + 1, x + pad - p // 2 : x + pad + p // 2 + 1]
            if ref.shape[0] != p or ref.shape[1] != p:
                continue
            ref_v = ref.astype(np.float32).ravel()
            ref_v = ref_v - float(np.mean(ref_v))
            ref_n = float(np.linalg.norm(ref_v)) + 1e-6
            ref_v = ref_v / ref_n
            sims, vals = [], []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    py, px = y0 + dy, x + pad + dx
                    cand = gs_p[py - p // 2 : py + p // 2 + 1, px - p // 2 : px + p // 2 + 1]
                    if cand.shape[0] != p or cand.shape[1] != p:
                        continue
                    cv = cand.astype(np.float32).ravel()
                    cv = cv - float(np.mean(cv))
                    cn = float(np.linalg.norm(cv)) + 1e-6
                    sim = float(np.dot(ref_v, cv / cn))
                    if sim <= 0.08:
                        continue
                    sims.append(sim)
                    vals.append(float(hp_p[py, px]))
            if not sims:
                continue
            order = np.argsort(sims)[::-1][:k]
            ws = np.array([sims[i] for i in order], dtype=np.float32)
            ws = ws / (float(np.sum(ws)) + 1e-6)
            vs = np.array([vals[i] for i in order], dtype=np.float32)
            val = float(np.sum(ws * vs))
            y1, x1 = min(y + st, h), min(x + st, w)
            corr[y:y1, x:x1] = val

    if st > 1:
        corr = cv2.resize(corr, (w, h), interpolation=cv2.INTER_LINEAR)
    out_hp_s = cv2.resize(corr, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
    hp_full = hp
    s = float(np.clip(strength, 0.0, 1.0))
    merged_hp = hp_full * (1.0 - s * flat) + out_hp_s * (s * flat)
    return (low + merged_hp).astype(np.float32)
