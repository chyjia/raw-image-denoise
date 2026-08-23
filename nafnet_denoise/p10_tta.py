"""P10 geometric TTA helpers (NTIRE self-ensemble on FE→spatial, not DN-post)."""

from __future__ import annotations

from typing import Callable

import numpy as np

AUGS = ("id", "ud", "lr", "udlr")


def apply_aug(image: np.ndarray, aug: str) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    if aug == "id":
        return img
    if aug == "ud":
        return np.flipud(img).copy()
    if aug == "lr":
        return np.fliplr(img).copy()
    if aug == "udlr":
        return np.flipud(np.fliplr(img)).copy()
    raise ValueError(f"unknown aug {aug}")


def undo_aug(image: np.ndarray, aug: str) -> np.ndarray:
    # Flip transforms are involutions
    return apply_aug(image, aug)


def average_augs(
    images: dict[str, np.ndarray],
    augs: tuple[str, ...],
    weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    selected = [images[a] for a in augs if a in images]
    if not selected:
        raise ValueError("no augs selected")
    stack = np.stack(selected, axis=0).astype(np.float32)
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(weights[: len(selected)], dtype=np.float32)
    w = w / max(float(w.sum()), 1e-6)
    return (stack * w[:, None, None]).sum(axis=0).astype(np.float32)
