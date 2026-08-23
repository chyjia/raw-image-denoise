"""P13 geometric TTA: pad-to-square rotations (D4) for non-square Mono10 frames."""

from __future__ import annotations

import numpy as np

from .p10_tta import apply_aug as apply_flip
from .p10_tta import undo_aug as undo_flip


def pad_to_square(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    img = np.ascontiguousarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    side = max(h, w)
    if h == side and w == side:
        return img, h, w
    out = np.zeros((side, side), dtype=np.float32)
    out[:h, :w] = img
    return out, h, w


def crop_from_square(image: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.ascontiguousarray(image[:h, :w], dtype=np.float32)


def apply_geom(image: np.ndarray, aug: str) -> np.ndarray:
    """Flip and/or 90° rotations. Rotations pad to square first (caller crops after undo)."""
    if aug in ("id", "ud", "lr", "udlr"):
        return apply_flip(image, aug)
    if aug.startswith("r"):
        # r90 / r180 / r270 / r90_lr / ...
        parts = aug.split("_", 1)
        k = {"r90": 1, "r180": 2, "r270": 3}.get(parts[0])
        if k is None:
            raise ValueError(f"unknown aug {aug}")
        padded, h, w = pad_to_square(image)
        rot = np.ascontiguousarray(np.rot90(padded, k), dtype=np.float32)
        if len(parts) == 2:
            rot = apply_flip(rot, parts[1])
        # stash crop meta in a side channel via attribute on ndarray? return only image;
        # undo must re-pad using original shape — store shape on a wrapper.
        out = rot
        out = out.copy()
        out.flags.writeable = True
        # encode (h,w) in unused way: return tuple from caller instead
        return out
    raise ValueError(f"unknown aug {aug}")


def geom_forward(image: np.ndarray, aug: str) -> tuple[np.ndarray, int, int]:
    """Apply geom aug; returns (transformed, orig_h, orig_w) for crop-after-undo."""
    h0, w0 = image.shape[:2]
    if aug in ("id", "ud", "lr", "udlr"):
        return apply_flip(image, aug), h0, w0
    parts = aug.split("_", 1)
    k = {"r90": 1, "r180": 2, "r270": 3}.get(parts[0])
    if k is None:
        raise ValueError(f"unknown aug {aug}")
    padded, h, w = pad_to_square(image)
    rot = np.ascontiguousarray(np.rot90(padded, k), dtype=np.float32)
    if len(parts) == 2:
        rot = apply_flip(rot, parts[1])
    return rot, h, w


def geom_inverse(image: np.ndarray, aug: str, h: int, w: int) -> np.ndarray:
    if aug in ("id", "ud", "lr", "udlr"):
        return undo_flip(image, aug)
    parts = aug.split("_", 1)
    k = {"r90": 1, "r180": 2, "r270": 3}.get(parts[0])
    if k is None:
        raise ValueError(f"unknown aug {aug}")
    img = image
    if len(parts) == 2:
        img = undo_flip(img, parts[1])
    # inverse rot90(k) = rot90(4-k)
    inv = np.ascontiguousarray(np.rot90(img, (4 - k) % 4), dtype=np.float32)
    return crop_from_square(inv, h, w)
