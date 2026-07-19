"""Global translation alignment helpers for Mono10 burst frames."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_translation(
    reference: np.ndarray,
    frame: np.ndarray,
    downsample: int = 4,
    max_shift: float = 16.0,
) -> tuple[float, float, float]:
    """Estimate the translation that warps ``frame`` into ``reference`` coordinates."""
    if downsample > 1:
        size = (reference.shape[1] // downsample, reference.shape[0] // downsample)
        ref_small = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        frame_small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    else:
        ref_small = reference
        frame_small = frame

    ref_small = ref_small.astype(np.float32)
    frame_small = frame_small.astype(np.float32)
    ref_small -= float(ref_small.mean())
    frame_small -= float(frame_small.mean())
    window = cv2.createHanningWindow(
        (ref_small.shape[1], ref_small.shape[0]),
        cv2.CV_32F,
    )
    (phase_x, phase_y), response = cv2.phaseCorrelate(
        ref_small,
        frame_small,
        window,
    )
    # phaseCorrelate(reference, frame) reports the reference->frame shift.
    # The inverse shift is the warp needed to put frame in reference coordinates.
    tx = float(np.clip(-phase_x * downsample, -max_shift, max_shift))
    ty = float(np.clip(-phase_y * downsample, -max_shift, max_shift))
    return tx, ty, float(response)


def warp_translation(
    image: np.ndarray,
    tx: float,
    ty: float,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )


def aligned_crop(
    image: np.ndarray,
    x0: int,
    y0: int,
    size: int,
    tx: float,
    ty: float,
) -> np.ndarray:
    """Sample a target-coordinate crop from a translated source frame."""
    center = (
        x0 + (size - 1) * 0.5 - tx,
        y0 + (size - 1) * 0.5 - ty,
    )
    return cv2.getRectSubPix(
        image.astype(np.float32, copy=False),
        (size, size),
        center,
    )
