"""True weight-space TTT-MIM (ECCV'24 Mansour et al.) for NAFNet flat arm.

Adapts a model copy with masked reconstruction on FE, then denoises once.
Caches adapted outputs under cache_dir for reuse across recipes.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import frames_to_model_input
from .infer import denoise_from_dn_frames, load_model, model_uses_sigma
from .des_cycle_common import SOTA_CKPT


def _mask_patches(
    h: int, w: int, patch: int, ratio: float, device: torch.device, rng: torch.Generator
) -> torch.Tensor:
    """Boolean mask True = masked (to reconstruct), shape 1x1xHxW."""
    ph = max(int(patch), 1)
    gh = (h + ph - 1) // ph
    gw = (w + ph - 1) // ph
    n = gh * gw
    k = max(1, int(round(n * float(ratio))))
    idx = torch.randperm(n, generator=rng)[:k]
    grid = torch.zeros(gh * gw, dtype=torch.bool)
    grid[idx] = True
    grid = grid.view(gh, gw)
    mask = grid.repeat_interleave(ph, 0).repeat_interleave(ph, 1)[:h, :w]
    return mask.view(1, 1, h, w).to(device)


@torch.enable_grad()
def adapt_mim(
    model: torch.nn.Module,
    fe: np.ndarray,
    exposure_ms: float,
    device: torch.device,
    *,
    niters: int = 8,
    lr: float = 1e-5,
    mask_ratio: float = 0.1,
    patch: int = 8,
    crop: int = 256,
    seed: int = 0,
) -> torch.nn.Module:
    """Few GD steps of MIM on random crops; returns adapted model (train→eval)."""
    m = copy.deepcopy(model).to(device)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=float(lr))
    fe32 = np.ascontiguousarray(fe, dtype=np.float32)
    h, w = fe32.shape
    cs = min(int(crop), h, w)
    rng = np.random.default_rng(int(seed))
    n_in = 4
    intro = getattr(m, "intro", None)
    if intro is not None and hasattr(intro, "in_channels"):
        extras = 2 if model_uses_sigma(m, n_in) else 1
        n_in = max(1, int(intro.in_channels) - extras)

    for step in range(int(niters)):
        y0 = int(rng.integers(0, h - cs + 1))
        x0 = int(rng.integers(0, w - cs + 1))
        crop_np = fe32[y0 : y0 + cs, x0 : x0 + cs]
        frames = [crop_np.copy() for _ in range(n_in)]
        use_sigma = model_uses_sigma(m, n_in)
        x_np = frames_to_model_input(frames, exposure_ms, use_sigma=use_sigma)
        x = torch.from_numpy(x_np[None]).float().to(device)
        _, _, ch, cw = x.shape
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + step * 997)
        mask = _mask_patches(ch, cw, patch, mask_ratio, device, g)
        fill = float(np.median(x_np[:n_in]))
        x_m = x.clone()
        x_m[:, :n_in] = torch.where(mask, fill, x[:, :n_in])
        pred = m(x_m)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if pred.shape[-2:] != (cs, cs):
            pred = F.interpolate(pred, size=(cs, cs), mode="bilinear", align_corners=False)
        tgt = x[:, :1]
        loss = F.mse_loss(pred * mask.float(), tgt * mask.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def denoise_adapted(
    model: torch.nn.Module,
    fe: np.ndarray,
    exposure_ms: float,
    device: torch.device,
    tile: int = 256,
) -> np.ndarray:
    was = bool(getattr(model, "wiener_front_end", False))
    model.wiener_front_end = False
    try:
        intro = getattr(model, "intro", None)
        n_in = 4
        if intro is not None and hasattr(intro, "in_channels"):
            extras = 2 if model_uses_sigma(model, 4) else 1
            n_in = max(1, int(intro.in_channels) - extras)
        frames = [fe.copy() for _ in range(n_in)]
        return denoise_from_dn_frames(
            model, frames, exposure_ms, device, tile=tile
        ).astype(np.float32)
    finally:
        model.wiener_front_end = was


_MODEL = None
_DEVICE = None


def get_base_model(device: torch.device | None = None):
    global _MODEL, _DEVICE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _MODEL is None or _DEVICE != device:
        _DEVICE = device
        _MODEL, _, _ = load_model(SOTA_CKPT, width=None, device=device)
        _MODEL.wiener_front_end = False
        _MODEL.eval()
    return _MODEL, _DEVICE


def ttt_sota_cached(
    cache_dir: Path,
    stem: str,
    *,
    niters: int = 8,
    lr: float = 1e-5,
    mask_ratio: float = 0.1,
    patch: int = 8,
    crop: int = 256,
    seed: int = 0,
    tile: int = 256,
) -> np.ndarray:
    """Return TTT-adapted flat-arm output; disk-cache by hyperparams."""
    tag = f"ttt_n{niters}_lr{lr:g}_r{mask_ratio:g}_p{patch}_c{crop}_s{seed}"
    out_path = cache_dir / f"{stem}_sota_{tag}.npy"
    if out_path.exists():
        return np.load(out_path)
    fe = np.load(cache_dir / f"{stem}_fe.npy").astype(np.float32)
    meta = __import__("json").loads((cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
    exp = float(meta.get("exposure_ms", 10.0))
    base, device = get_base_model()
    adapted = adapt_mim(
        base,
        fe,
        exp,
        device,
        niters=niters,
        lr=lr,
        mask_ratio=mask_ratio,
        patch=patch,
        crop=crop,
        seed=seed,
    )
    out = denoise_adapted(adapted, fe, exp, device, tile=tile)
    np.save(out_path, out.astype(np.float32))
    del adapted
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
