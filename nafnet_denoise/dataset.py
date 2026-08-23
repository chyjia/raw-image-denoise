"""PyTorch dataset: PixelShift200 clean patches + PTC-calibrated synthetic noise.

Each sample synthesizes ``input_frames`` independent noisy observations of the
same clean patch, brightness-aligns them, and feeds the VST stack plus an
exposure-conditioning channel to the network. The target is the clean center
frame in the normalized VST domain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    DEFAULT_BLACK_LEVEL_DN,
    DEFAULT_DARK_VARIANCE_PER_S,
    DEFAULT_INPUT_FRAMES,
    PTC_INTERCEPT,
    PTC_SLOPE,
    TARGET_MEAN_DN_MAX,
    TARGET_MEAN_DN_MIN,
    center_frame_index,
    frames_to_model_input,
    poisson_gaussian_noise,
    raw_to_model_input,
)


class PixelShiftPatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int = 128,
        patches_per_epoch: int = 4000,
        exposure_ms_range: tuple[float, float] = (50.0, 200.0),
        input_frames: int = DEFAULT_INPUT_FRAMES,
        noise_jitter: float = 0.15,
        black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
        dark_variance_per_s_range: tuple[float, float] = (
            0.0,
            DEFAULT_DARK_VARIANCE_PER_S * 2.0,
        ),
        black_drift_sigma: float = 1.0,
        fixed_pattern_sigma: float = 0.5,
        row_noise_sigma: float = 0.3,
        column_noise_sigma: float = 0.2,
        cache_images_in_ram: bool = False,
        seed: int = 0,
        use_sigma: bool = False,
        wiener_front_end: bool = False,
        wiener_merge_frames: int | None = None,
        wiener_tile: int = 32,
        wiener_overlap: int = 16,
        wiener_c_factor: float = 8.0,
        wiener_spatial: bool = False,
        wiener_spatial_c_factor: float | None = None,
        dark_target_mean_max: float | None = None,
        postmerge_calib=None,
        scale_synth_residual: bool = True,
    ):
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.exposure_ms_range = exposure_ms_range
        self.input_frames = input_frames
        self.center_index = center_frame_index(input_frames)
        self.noise_jitter = noise_jitter
        self.black_level_dn = black_level_dn
        self.dark_variance_per_s_range = dark_variance_per_s_range
        self.black_drift_sigma = black_drift_sigma
        self.fixed_pattern_sigma = fixed_pattern_sigma
        self.row_noise_sigma = row_noise_sigma
        self.column_noise_sigma = column_noise_sigma
        self.cache_images_in_ram = cache_images_in_ram
        self.use_sigma = use_sigma
        self.wiener_front_end = bool(wiener_front_end)
        self.wiener_merge_frames = int(
            wiener_merge_frames if wiener_merge_frames is not None else input_frames
        )
        if self.wiener_front_end and self.wiener_merge_frames < input_frames:
            raise ValueError("wiener_merge_frames must be >= input_frames")
        self.wiener_tile = int(wiener_tile)
        self.wiener_overlap = int(wiener_overlap)
        self.wiener_c_factor = float(wiener_c_factor)
        self.wiener_spatial = bool(wiener_spatial)
        self.wiener_spatial_c_factor = (
            None if wiener_spatial_c_factor is None else float(wiener_spatial_c_factor)
        )
        self.dark_target_mean_max = dark_target_mean_max
        self.postmerge_calib = postmerge_calib
        self.scale_synth_residual = bool(scale_synth_residual)
        self.rng = np.random.default_rng(seed)

        self.entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")

        self._images: dict[int, np.ndarray] = {}
        if cache_images_in_ram:
            for index, entry in enumerate(self.entries):
                self._images[index] = np.load(entry["mono"]).astype(np.float32)

    def __len__(self) -> int:
        return self.patches_per_epoch

    def _get_image(self, index: int) -> np.ndarray:
        if index in self._images:
            return self._images[index]
        return np.load(self.entries[index]["mono"], mmap_mode="r").astype(np.float32)

    def __getitem__(self, _: int):
        image_index = int(self.rng.integers(len(self.entries)))
        image = self._get_image(image_index)
        height, width = image.shape
        size = self.patch_size

        y0 = int(self.rng.integers(0, height - size + 1))
        x0 = int(self.rng.integers(0, width - size + 1))
        patch = image[y0 : y0 + size, x0 : x0 + size].copy()

        patch_min = float(patch.min())
        patch_max = float(patch.max())
        if patch_max > patch_min:
            patch = (patch - patch_min) / (patch_max - patch_min)

        mean_hi = TARGET_MEAN_DN_MAX
        if self.dark_target_mean_max is not None:
            mean_hi = min(mean_hi, float(self.dark_target_mean_max))
        target_mean = float(
            self.rng.uniform(
                max(TARGET_MEAN_DN_MIN, self.black_level_dn + 20.0),
                max(mean_hi, self.black_level_dn + 40.0),
            )
        )
        current_mean = float(patch.mean())
        clean = np.clip(
            self.black_level_dn
            + patch
            * ((target_mean - self.black_level_dn) / max(current_mean, 1e-6)),
            self.black_level_dn,
            1023.0,
        ).astype(
            np.float32,
        )

        # One perturbed camera noise model is shared by the whole burst. This
        # broadens the calibrated PTC distribution without creating an
        # unrealistic per-frame camera response change.
        jitter_low = 1.0 - self.noise_jitter
        jitter_high = 1.0 + self.noise_jitter
        slope = PTC_SLOPE * float(self.rng.uniform(jitter_low, jitter_high))
        intercept = PTC_INTERCEPT * float(self.rng.uniform(jitter_low, jitter_high))
        exposure_ms = float(self.rng.uniform(*self.exposure_ms_range))
        dark_variance_per_s = float(
            self.rng.uniform(*self.dark_variance_per_s_range)
        )
        fixed_pattern = self.rng.normal(
            0.0,
            self.fixed_pattern_sigma,
            size=clean.shape,
        ).astype(np.float32)
        n_synth = (
            self.wiener_merge_frames if self.wiener_front_end else self.input_frames
        )
        noisy_frames = []
        for _ in range(n_synth):
            noisy = poisson_gaussian_noise(
                clean,
                self.rng,
                slope,
                intercept,
                exposure_ms,
                self.black_level_dn,
                dark_variance_per_s,
            )
            noisy += fixed_pattern
            noisy += float(self.rng.normal(0.0, self.black_drift_sigma))
            noisy += self.rng.normal(
                0.0,
                self.row_noise_sigma,
                size=(size, 1),
            ).astype(np.float32)
            noisy += self.rng.normal(
                0.0,
                self.column_noise_sigma,
                size=(1, size),
            ).astype(np.float32)
            noisy_frames.append(
                np.rint(np.clip(noisy, 0.0, 1023.0)).astype(np.float32)
            )

        k = int(self.rng.integers(4))
        rotated_noisy = [np.rot90(frame, k) for frame in noisy_frames]
        clean_rot = np.rot90(clean, k)
        if self.rng.random() < 0.5:
            rotated_noisy = [np.fliplr(frame) for frame in rotated_noisy]
            clean_rot = np.fliplr(clean_rot)

        if self.wiener_front_end:
            from .wiener_merge import merge_burst_dn

            ref = center_frame_index(len(rotated_noisy))
            merged, _meta = merge_burst_dn(
                rotated_noisy,
                reference_index=ref,
                align=False,
                tile_size=self.wiener_tile,
                overlap=self.wiener_overlap,
                c_factor=self.wiener_c_factor,
                spatial_wiener=self.wiener_spatial,
                spatial_c_factor=self.wiener_spatial_c_factor,
            )
            if self.postmerge_calib is not None and self.scale_synth_residual:
                from .postmerge_noise import scale_residual_to_calib

                merged = scale_residual_to_calib(
                    merged,
                    clean_rot,
                    self.postmerge_calib,
                    exposure_ms=exposure_ms,
                    black_level_dn=self.black_level_dn,
                    dark_variance_per_s=dark_variance_per_s,
                )
            rotated_noisy = [merged.copy() for _ in range(self.input_frames)]

        model_input = frames_to_model_input(
            rotated_noisy,
            exposure_ms,
            reference_index=self.center_index,
            dark_variance_per_s=dark_variance_per_s,
            use_sigma=self.use_sigma,
            postmerge_calib=(
                self.postmerge_calib if self.wiener_front_end else None
            ),
        )
        clean_vst = raw_to_model_input(
            clean_rot,
            exposure_ms=exposure_ms,
            dark_variance_per_s=dark_variance_per_s,
        )

        input_tensor = torch.from_numpy(model_input)
        target_tensor = torch.from_numpy(np.ascontiguousarray(clean_vst[None, :, :]))
        return input_tensor.float(), target_tensor.float()
