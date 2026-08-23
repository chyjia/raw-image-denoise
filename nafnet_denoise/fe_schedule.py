"""fps / residual-σ gated edge-aware spatial Wiener schedules (P0-a).

Presets come from the ea_mild_gamma sweep win (mean DES 0.9051) and the
baseline spatial pass that better preserved f10 edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .postmerge_noise import measured_fe_sigma_dn
from .wiener_merge import adaptive_spatial_wiener, spatial_wiener_tiles


@dataclass(frozen=True)
class SpatialFeParams:
    name: str
    adaptive: bool
    flat_c_mult: float = 1.0
    edge_c_mult: float = 1.0
    dark_boost: float = 0.0
    mask_harden: float = 0.0
    freq_gamma: float = 0.0
    spatial_c_factor: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# Frozen DualHead SOTA FE from compare_edge_aware_wiener.
EA_MILD_GAMMA = SpatialFeParams(
    name="ea_mild_gamma",
    adaptive=True,
    flat_c_mult=1.75,
    edge_c_mult=0.45,
    dark_boost=0.35,
    mask_harden=8.0,
    freq_gamma=0.15,
    spatial_c_factor=1.0,
)

# Non-adaptive baseline spatial (best for high-fps / low residual σ).
BASELINE_SPATIAL = SpatialFeParams(
    name="baseline_spatial",
    adaptive=False,
    spatial_c_factor=1.0,
)

# Soft mid for ambiguous clips.
EA_MID_SOFT = SpatialFeParams(
    name="ea_mid_soft",
    adaptive=True,
    flat_c_mult=1.45,
    edge_c_mult=0.70,
    dark_boost=0.20,
    mask_harden=6.0,
    freq_gamma=0.06,
    spatial_c_factor=1.0,
)


def select_spatial_fe_params(
    fps: float | None = None,
    fe_sigma_dn: float | None = None,
    low_fps: float = 1.0,
    high_fps: float = 8.0,
    high_sigma: float = 1.20,
    low_sigma: float = 0.55,
) -> SpatialFeParams:
    """Pick FE spatial params from fps (primary) and post-temporal flat σ.

    Post-temporal σ is typically ~0.5–1.6 DN before spatial Wiener, so σ
    thresholds are calibrated to that scale. fps is the main switch:
    low-fps → ea_mild_gamma, high-fps → baseline (protect f10 edges).
    """
    fps_v = None if fps is None else float(fps)
    sig_v = None if fe_sigma_dn is None else float(fe_sigma_dn)

    # Primary: fps metadata (stable across holdout naming).
    if fps_v is not None:
        if fps_v <= low_fps:
            return EA_MILD_GAMMA
        if fps_v >= high_fps:
            return BASELINE_SPATIAL

    # Mid-fps or missing fps: use post-temporal residual σ.
    if sig_v is not None:
        if sig_v >= high_sigma:
            return EA_MILD_GAMMA
        if sig_v <= low_sigma:
            return BASELINE_SPATIAL
    return EA_MID_SOFT


def apply_spatial_fe(
    temporal_merged: np.ndarray,
    params: SpatialFeParams,
    n_frames_averaged: int = 16,
    tile_size: int = 32,
    overlap: int = 16,
) -> np.ndarray:
    """Apply scheduled spatial Wiener on an already temporally merged DN image."""
    if params.adaptive:
        return adaptive_spatial_wiener(
            temporal_merged,
            n_frames_averaged=n_frames_averaged,
            tile_size=tile_size,
            overlap=overlap,
            spatial_c_factor=params.spatial_c_factor,
            flat_c_mult=params.flat_c_mult,
            dark_boost=params.dark_boost,
            edge_c_mult=params.edge_c_mult,
            mask_harden=params.mask_harden,
            freq_gamma=params.freq_gamma,
        )
    return spatial_wiener_tiles(
        temporal_merged,
        n_frames_averaged=n_frames_averaged,
        tile_size=tile_size,
        overlap=overlap,
        c_factor=params.spatial_c_factor,
        freq_gamma=0.0,
    )


def gated_spatial_from_temporal(
    temporal_merged: np.ndarray,
    fps: float | None = None,
    n_frames_averaged: int = 16,
    tile_size: int = 32,
    overlap: int = 16,
) -> tuple[np.ndarray, SpatialFeParams, float]:
    """Measure post-temporal FE σ, select schedule, apply spatial Wiener."""
    sigma = measured_fe_sigma_dn(temporal_merged)
    params = select_spatial_fe_params(fps=fps, fe_sigma_dn=sigma)
    out = apply_spatial_fe(
        temporal_merged,
        params,
        n_frames_averaged=n_frames_averaged,
        tile_size=tile_size,
        overlap=overlap,
    )
    return out, params, sigma
