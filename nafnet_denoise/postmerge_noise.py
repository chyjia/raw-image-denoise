"""Post-Wiener residual noise model for σ conditioning and synthetic residuals.

After temporal+spatial Wiener, DualHead no longer sees single-frame PTC noise.
This module fits / applies a residual σ model measured on Wiener-merged flats
(Unprocessing-style train/test match on the signal the network actually sees).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import (
    DEFAULT_BLACK_LEVEL_DN,
    DEFAULT_DARK_VARIANCE_PER_S,
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    signal_variance_dn2,
)


@dataclass
class PostMergeNoiseCalib:
    """Affine residual model: σ_res = alpha * σ_ptc + beta (+ optional /√N)."""

    version: int = 1
    alpha: float = 1.0
    beta_dn: float = 0.0
    sigma_floor_dn: float = 0.05
    divide_sqrt_n: bool = True
    n_merge: int = 16
    spatial_wiener: bool = True
    wiener_tile: int = 32
    wiener_overlap: int = 16
    wiener_c_factor: float = 8.0
    wiener_spatial_c_factor: float = 1.0
    n_samples: int = 0
    rmse_dn: float = 0.0
    median_ratio: float = 1.0
    notes: str = ""

    def predict_sigma_dn(
        self,
        image_dn: np.ndarray,
        exposure_ms: float | None = None,
        black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
        dark_variance_per_s: float = 0.0,
        slope: float = PTC_SLOPE,
        intercept: float = PTC_INTERCEPT,
    ) -> np.ndarray:
        ptc = np.sqrt(
            signal_variance_dn2(
                image_dn,
                exposure_ms,
                slope,
                intercept,
                black_level_dn,
                dark_variance_per_s,
            )
        )
        if self.divide_sqrt_n:
            ptc = ptc / np.sqrt(max(float(self.n_merge), 1.0))
        sigma = self.alpha * ptc + float(self.beta_dn)
        return np.maximum(sigma, float(self.sigma_floor_dn)).astype(np.float32)

    def predict_sigma_dn_scalar(
        self,
        mean_dn: float,
        exposure_ms: float | None = None,
        black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
        dark_variance_per_s: float = 0.0,
    ) -> float:
        arr = self.predict_sigma_dn(
            np.asarray(mean_dn, dtype=np.float32),
            exposure_ms=exposure_ms,
            black_level_dn=black_level_dn,
            dark_variance_per_s=dark_variance_per_s,
        )
        return float(np.asarray(arr))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PostMergeNoiseCalib":
        fields = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**fields)


def save_postmerge_calib(path: str | Path, calib: PostMergeNoiseCalib) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")


def load_postmerge_calib(path: str | Path | None) -> PostMergeNoiseCalib | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing post-merge noise calib: {path}")
    return PostMergeNoiseCalib.from_dict(json.loads(path.read_text(encoding="utf-8")))


def highpass_mad_sigma(image: np.ndarray, blur_sigma: float = 1.2) -> float:
    """DES-compatible highpass MAD / 0.6745 on a DN patch."""
    image = np.asarray(image, dtype=np.float32)
    lowpass = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    residual = image - lowpass
    return float(np.median(np.abs(residual - np.median(residual))) / 0.6745)


def highpass_mad_sigma_masked(
    image: np.ndarray,
    mask: np.ndarray,
    blur_sigma: float = 1.2,
) -> float:
    """Highpass MAD on 2D image, statistics restricted to ``mask`` (keep topology)."""
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 32:
        return highpass_mad_sigma(image, blur_sigma=blur_sigma)
    lowpass = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    residual = (image - lowpass)[mask]
    return float(np.median(np.abs(residual - np.median(residual))) / 0.6745)


def flat_mask(image: np.ndarray, percentile: float = 40.0) -> np.ndarray:
    """Soft flat mask from Sobel magnitude (True = flatter)."""
    image = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    threshold = float(np.percentile(mag, percentile))
    return mag <= threshold


def normalize_sigma_dn_channel(
    sigma_dn: np.ndarray | float,
    image_dn: np.ndarray,
    exposure_ms: float | None = None,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """Map DN-domain σ to the same [0,1]-ish channel range as PTC / residual maps."""
    sigma_max = np.sqrt(
        float(
            signal_variance_dn2(
                np.asarray(RAW_MAX),
                exposure_ms,
                slope,
                intercept,
                black_level_dn,
                dark_variance_per_s,
            )
        )
    )
    scale = max(sigma_max, 1e-6)
    if np.isscalar(sigma_dn) or getattr(sigma_dn, "ndim", 0) == 0:
        return np.full(
            np.asarray(image_dn).shape,
            float(sigma_dn) / scale,
            dtype=np.float32,
        )
    return (np.asarray(sigma_dn, dtype=np.float32) / scale).astype(np.float32)


def residual_noise_sigma_map(
    image_dn: np.ndarray,
    calib: PostMergeNoiseCalib,
    exposure_ms: float | None = None,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
) -> np.ndarray:
    """Normalized residual σ map (same denom as PTC map → comparable channel range)."""
    sigma = calib.predict_sigma_dn(
        image_dn,
        exposure_ms=exposure_ms,
        black_level_dn=black_level_dn,
        dark_variance_per_s=dark_variance_per_s,
        slope=slope,
        intercept=intercept,
    )
    return normalize_sigma_dn_channel(
        sigma,
        image_dn,
        exposure_ms=exposure_ms,
        black_level_dn=black_level_dn,
        dark_variance_per_s=dark_variance_per_s,
        slope=slope,
        intercept=intercept,
    )


def measured_fe_sigma_dn(
    image_dn: np.ndarray,
    flat_percentile: float = 40.0,
) -> float:
    """FFDNet-style measured residual σ on flats after Wiener FE (DN)."""
    image = np.asarray(image_dn, dtype=np.float32)
    mask = flat_mask(image, percentile=flat_percentile)
    if float(mask.mean()) < 0.05:
        mask = np.ones_like(image, dtype=bool)
    return highpass_mad_sigma_masked(image, mask)


def measured_fe_sigma_map(
    image_dn: np.ndarray,
    exposure_ms: float | None = None,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
    slope: float = PTC_SLOPE,
    intercept: float = PTC_INTERCEPT,
    flat_percentile: float = 40.0,
) -> np.ndarray:
    """Constant spatial map from measured post-FE flat highpass MAD (normalized)."""
    sigma = measured_fe_sigma_dn(image_dn, flat_percentile=flat_percentile)
    return normalize_sigma_dn_channel(
        sigma,
        image_dn,
        exposure_ms=exposure_ms,
        black_level_dn=black_level_dn,
        dark_variance_per_s=dark_variance_per_s,
        slope=slope,
        intercept=intercept,
    )


def scale_residual_to_calib(
    merged_dn: np.ndarray,
    clean_dn: np.ndarray,
    calib: PostMergeNoiseCalib,
    exposure_ms: float | None = None,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
) -> np.ndarray:
    """Rescale synthetic Wiener residual so flat highpass σ matches calib."""
    merged = np.asarray(merged_dn, dtype=np.float32)
    clean = np.asarray(clean_dn, dtype=np.float32)
    mask = flat_mask(clean)
    if float(mask.mean()) < 0.05:
        mask = np.ones_like(clean, dtype=bool)
    actual = highpass_mad_sigma_masked(merged, mask)
    target = calib.predict_sigma_dn_scalar(
        float(clean[mask].mean()),
        exposure_ms=exposure_ms,
        black_level_dn=black_level_dn,
        dark_variance_per_s=dark_variance_per_s,
    )
    scale = float(np.clip(target / max(actual, 1e-4), scale_min, scale_max))
    out = clean + (merged - clean) * scale
    return np.clip(out, 0.0, RAW_MAX).astype(np.float32)


def fit_affine_sigma(
    ptc_sigma: np.ndarray,
    measured_sigma: np.ndarray,
) -> tuple[float, float, float]:
    """Least-squares σ_meas ≈ alpha * σ_ptc_feature + beta."""
    x = np.asarray(ptc_sigma, dtype=np.float64).reshape(-1)
    y = np.asarray(measured_sigma, dtype=np.float64).reshape(-1)
    if x.size < 3:
        ratio = float(np.median(y / np.maximum(x, 1e-6))) if x.size else 1.0
        return ratio, 0.0, 0.0
    a = np.stack([x, np.ones_like(x)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    pred = alpha * x + beta
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return alpha, beta, rmse


# Default dark variance used for real train bursts when building σ maps.
DEFAULT_POSTMERGE_DARK_VAR = DEFAULT_DARK_VARIANCE_PER_S
