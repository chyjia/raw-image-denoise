"""Synthetic and real 16-frame datasets for BurstNAFNet."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .alignment import soft_gate_burst
from .common import (
    DEFAULT_BLACK_LEVEL_DN,
    DEFAULT_DARK_VARIANCE_PER_S,
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    center_frame_index,
    decode_mono10,
    exposure_intercept,
    noise_sigma_map,
    normalize_exposure,
    poisson_gaussian_noise,
    raw_to_model_input,
    vst_forward,
    vst_inverse,
)


def apply_d4(arrays: list[np.ndarray], mode: int) -> list[np.ndarray]:
    rotation = mode % 4
    reflected = mode >= 4
    outputs = [np.rot90(array, rotation, axes=(-2, -1)) for array in arrays]
    if reflected:
        outputs = [np.flip(array, axis=-1) for array in outputs]
    return [np.ascontiguousarray(array) for array in outputs]


def apply_real_flips(
    arrays: list[np.ndarray],
    horizontal: bool,
    vertical: bool,
) -> list[np.ndarray]:
    outputs = arrays
    if horizontal:
        outputs = [np.flip(array, axis=-1) for array in outputs]
    if vertical:
        outputs = [np.flip(array, axis=-2) for array in outputs]
    return [np.ascontiguousarray(array) for array in outputs]


def make_burst_tensor(
    frames_dn: list[np.ndarray],
    exposure_ms: float,
    slope: float,
    intercept: float,
    black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
    dark_variance_per_s: float = 0.0,
    postmerge_calib=None,
    measured_fe_sigma: bool = False,
) -> np.ndarray:
    exposure = normalize_exposure(exposure_ms)
    channels = []
    for frame in frames_dn:
        vst = raw_to_model_input(
            frame,
            exposure_ms=exposure_ms,
            dark_variance_per_s=dark_variance_per_s,
        )
        if measured_fe_sigma:
            from .postmerge_noise import measured_fe_sigma_map

            sigma = measured_fe_sigma_map(
                frame,
                exposure_ms=exposure_ms,
                black_level_dn=black_level_dn,
                dark_variance_per_s=dark_variance_per_s,
                slope=slope,
                intercept=intercept,
            )
        elif postmerge_calib is not None:
            from .postmerge_noise import residual_noise_sigma_map

            sigma = residual_noise_sigma_map(
                frame,
                postmerge_calib,
                exposure_ms=exposure_ms,
                black_level_dn=black_level_dn,
                dark_variance_per_s=dark_variance_per_s,
                slope=slope,
                intercept=intercept,
            )
        else:
            sigma = noise_sigma_map(
                frame,
                slope,
                intercept,
                exposure_ms=exposure_ms,
                black_level_dn=black_level_dn,
                dark_variance_per_s=dark_variance_per_s,
            )
        exposure_channel = np.full_like(vst, exposure, dtype=np.float32)
        channels.append(np.stack((vst, sigma, exposure_channel), axis=0))
    return np.ascontiguousarray(np.stack(channels, axis=0))


def _motion_warp(
    image: np.ndarray,
    shift_x: float,
    shift_y: float,
    angle_degrees: float,
) -> np.ndarray:
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) * 0.5, (height - 1) * 0.5),
        angle_degrees,
        1.0,
    )
    matrix[:, 2] += (shift_x, shift_y)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


class SyntheticBurstDataset(Dataset):
    """PixelShift clean patches with motion, D4, and calibrated noisy bursts."""

    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int = 256,
        input_frames: int = 16,
        samples_per_epoch: int = 4000,
        noise_jitter: float = 0.15,
        max_shift: float = 2.0,
        max_rotation: float = 0.25,
        motion_strength: float = 1.0,
        exposure_ms_range: tuple[float, float] = (10.0, 400.0),
        black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
        black_drift_sigma: float = 0.0,
        fixed_pattern_sigma: float = 0.0,
        row_noise_sigma: float = 0.0,
        column_noise_sigma: float = 0.0,
        dark_variance_per_s_range: tuple[float, float] = (
            0.0,
            DEFAULT_DARK_VARIANCE_PER_S * 2.0,
        ),
        seed: int = 0,
    ):
        self.entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.patch_size = patch_size
        self.input_frames = input_frames
        self.reference_index = center_frame_index(input_frames)
        self.samples_per_epoch = samples_per_epoch
        self.noise_jitter = noise_jitter
        self.max_shift = max_shift
        self.max_rotation = max_rotation
        self.motion_strength = motion_strength
        if exposure_ms_range[0] <= 0 or exposure_ms_range[1] < exposure_ms_range[0]:
            raise ValueError("exposure_ms_range must be positive and ordered")
        self.exposure_ms_range = exposure_ms_range
        if not 0.0 <= black_level_dn < RAW_MAX:
            raise ValueError("black_level_dn must be in the Mono10 range")
        self.black_level_dn = black_level_dn
        self.black_drift_sigma = max(0.0, black_drift_sigma)
        self.fixed_pattern_sigma = max(0.0, fixed_pattern_sigma)
        self.row_noise_sigma = max(0.0, row_noise_sigma)
        self.column_noise_sigma = max(0.0, column_noise_sigma)
        if dark_variance_per_s_range[0] < 0 or (
            dark_variance_per_s_range[1] < dark_variance_per_s_range[0]
        ):
            raise ValueError("dark_variance_per_s_range must be non-negative and ordered")
        self.dark_variance_per_s_range = dark_variance_per_s_range
        self.rng = np.random.default_rng(seed)
        self._images: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _image(self, index: int) -> np.ndarray:
        if index not in self._images:
            self._images[index] = np.load(
                self.entries[index]["mono"],
                mmap_mode="r",
            )
        return self._images[index]

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        image = self._image(int(self.rng.integers(len(self.entries))))
        size = self.patch_size
        margin = 8
        y0 = int(self.rng.integers(margin, image.shape[0] - size - margin + 1))
        x0 = int(self.rng.integers(margin, image.shape[1] - size - margin + 1))
        clean = np.asarray(image[y0 : y0 + size, x0 : x0 + size], dtype=np.float32).copy()
        low, high = np.percentile(clean, (0.5, 99.5))
        clean = np.clip((clean - low) / max(high - low, 1.0), 0.0, 1.0)
        minimum_mean = max(100.0, self.black_level_dn + 20.0)
        target_mean = float(self.rng.uniform(minimum_mean, 950.0))
        signal_mean = max(float(clean.mean()), 1e-6)
        clean = self.black_level_dn + clean * (
            (target_mean - self.black_level_dn) / signal_mean
        )
        clean = np.clip(clean, self.black_level_dn, RAW_MAX).astype(np.float32)

        jitter_low = 1.0 - self.noise_jitter
        jitter_high = 1.0 + self.noise_jitter
        slope = PTC_SLOPE * float(self.rng.uniform(jitter_low, jitter_high))
        intercept = PTC_INTERCEPT * float(self.rng.uniform(jitter_low, jitter_high))
        exposure_ms = float(self.rng.uniform(*self.exposure_ms_range))
        dark_variance_per_s = float(
            self.rng.uniform(*self.dark_variance_per_s_range)
        )

        # Correlated random-walk motion, expressed relative to the reference.
        shifts = np.cumsum(
            self.rng.normal(0.0, 0.45, size=(self.input_frames, 2)),
            axis=0,
        )
        angles = np.cumsum(
            self.rng.normal(0.0, 0.05, size=self.input_frames),
            axis=0,
        )
        shifts -= shifts[self.reference_index]
        angles -= angles[self.reference_index]
        shifts = np.clip(shifts, -self.max_shift, self.max_shift) * self.motion_strength
        angles = np.clip(angles, -self.max_rotation, self.max_rotation) * self.motion_strength

        noisy_frames = []
        fixed_pattern = self.rng.normal(
            0.0,
            self.fixed_pattern_sigma,
            size=clean.shape,
        ).astype(np.float32)
        for frame_index in range(self.input_frames):
            noisy = poisson_gaussian_noise(
                clean,
                self.rng,
                slope,
                intercept,
                exposure_ms=exposure_ms,
                black_level_dn=self.black_level_dn,
                dark_variance_per_s=dark_variance_per_s,
            )
            frame_offset = float(self.rng.normal(0.0, self.black_drift_sigma))
            row_noise = self.rng.normal(
                0.0,
                self.row_noise_sigma,
                size=(size, 1),
            ).astype(np.float32)
            column_noise = self.rng.normal(
                0.0,
                self.column_noise_sigma,
                size=(1, size),
            ).astype(np.float32)
            noisy = np.clip(
                noisy + fixed_pattern + frame_offset + row_noise + column_noise,
                0.0,
                RAW_MAX,
            )
            noisy = np.rint(noisy).astype(np.float32)
            noisy = _motion_warp(
                noisy,
                float(shifts[frame_index, 0]),
                float(shifts[frame_index, 1]),
                float(angles[frame_index]),
            )
            # Sparse local occlusion teaches attention to reject inconsistent frames.
            if frame_index != self.reference_index and self.rng.random() < 0.08:
                box = int(self.rng.integers(8, max(9, size // 5)))
                oy = int(self.rng.integers(0, size - box + 1))
                ox = int(self.rng.integers(0, size - box + 1))
                noisy[oy : oy + box, ox : ox + box] = np.median(noisy)
            noisy_frames.append(noisy)

        mode = int(self.rng.integers(8))
        transformed = apply_d4([*noisy_frames, clean], mode)
        noisy_frames = transformed[:-1]
        clean = transformed[-1]
        burst = make_burst_tensor(
            noisy_frames,
            exposure_ms,
            slope,
            intercept,
            black_level_dn=self.black_level_dn,
            dark_variance_per_s=dark_variance_per_s,
        )
        target = raw_to_model_input(
            clean,
            exposure_ms=exposure_ms,
            dark_variance_per_s=dark_variance_per_s,
        )[None]
        mask = np.ones_like(target, dtype=np.float32)
        return {
            "burst": torch.from_numpy(burst).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "teacher": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "mask": torch.from_numpy(mask).float(),
            "teacher_valid": torch.tensor(0.0, dtype=torch.float32),
        }


class RealBurstDataset(Dataset):
    """Aligned crops from real Mono10 videos and cached temporal references."""

    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int = 256,
        input_frames: int = 16,
        samples_per_epoch: int = 4000,
        center_mode: str = "random",
        seed: int = 0,
        target_frames: int = 32,
        align_soft_gate: bool = False,
        align_threshold: float = 0.03,
        align_temperature: float = 0.015,
        align_drop_prob: float = 0.0,
        wiener_front_end: bool = False,
        wiener_merge_frames: int | None = None,
        wiener_tile: int = 32,
        wiener_overlap: int = 16,
        wiener_c_factor: float = 8.0,
        wiener_spatial: bool = False,
        wiener_spatial_c_factor: float | None = None,
        wiener_spatial_adaptive: bool = False,
        wiener_spatial_flat_c_mult: float = 2.5,
        wiener_spatial_dark_boost: float = 0.5,
        wiener_spatial_edge_c_mult: float = 1.0,
        wiener_spatial_mask_harden: float = 0.0,
        wiener_spatial_freq_gamma: float = 0.0,
        wiener_fe_schedule: str | None = None,
        dark_boost: float = 0.0,
        postmerge_calib=None,
        measured_fe_sigma: bool = False,
        wiener_bm3d_teacher: bool = False,
        wiener_bm3d_sigma: float = 0.35,
        wiener_bm3d_prob: float = 1.0,
        scene_loss_power: float = 0.0,
        scene_loss_ref_fps: float = 10.0,
        scene_loss_max: float = 4.0,
        hard_scene_substr: tuple[str, ...] | list[str] = (),
        hard_scene_sample_mult: float = 1.0,
        hard_scene_loss_mult: float = 1.0,
        hard_wiener_bm3d_sigma: float | None = None,
        hard_wiener_bm3d_sigma_map: dict[str, float] | None = None,
        edge_focus_substr: tuple[str, ...] | list[str] = (),
    ):
        self.entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.patch_size = patch_size
        self.input_frames = input_frames
        self.target_frames = max(int(target_frames), 4)
        self.reference_index = center_frame_index(input_frames)
        self.samples_per_epoch = samples_per_epoch
        if center_mode not in {"anchor", "random"}:
            raise ValueError("center_mode must be 'anchor' or 'random'")
        self.center_mode = center_mode
        self.align_soft_gate = bool(align_soft_gate)
        self.align_threshold = float(align_threshold)
        self.align_temperature = float(align_temperature)
        self.align_drop_prob = float(np.clip(align_drop_prob, 0.0, 1.0))
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
        self.wiener_spatial_adaptive = bool(wiener_spatial_adaptive)
        self.wiener_spatial_flat_c_mult = float(max(wiener_spatial_flat_c_mult, 1.0))
        self.wiener_spatial_dark_boost = float(max(wiener_spatial_dark_boost, 0.0))
        self.wiener_spatial_edge_c_mult = float(max(wiener_spatial_edge_c_mult, 1e-3))
        self.wiener_spatial_mask_harden = float(max(wiener_spatial_mask_harden, 0.0))
        self.wiener_spatial_freq_gamma = float(max(wiener_spatial_freq_gamma, 0.0))
        self.wiener_fe_schedule = (
            None
            if wiener_fe_schedule in (None, "", "none", "fixed")
            else str(wiener_fe_schedule)
        )
        self.dark_boost = float(max(0.0, dark_boost))
        self.postmerge_calib = postmerge_calib
        self.measured_fe_sigma = bool(measured_fe_sigma)
        self.wiener_bm3d_teacher = bool(wiener_bm3d_teacher)
        self.wiener_bm3d_sigma = float(max(wiener_bm3d_sigma, 1e-3))
        self.wiener_bm3d_prob = float(np.clip(wiener_bm3d_prob, 0.0, 1.0))
        self.scene_loss_power = float(max(0.0, scene_loss_power))
        self.scene_loss_ref_fps = float(max(scene_loss_ref_fps, 0.25))
        self.scene_loss_max = float(max(scene_loss_max, 1.0))
        self.hard_scene_substr = tuple(
            part.strip() for part in hard_scene_substr if str(part).strip()
        )
        self.hard_scene_sample_mult = float(max(hard_scene_sample_mult, 1.0))
        self.hard_scene_loss_mult = float(max(hard_scene_loss_mult, 1.0))
        self.edge_focus_substr = tuple(
            part.strip() for part in edge_focus_substr if str(part).strip()
        )
        self.hard_wiener_bm3d_sigma = (
            None
            if hard_wiener_bm3d_sigma is None
            else float(max(hard_wiener_bm3d_sigma, 1e-3))
        )
        self.hard_wiener_bm3d_sigma_map = {
            str(key): float(max(value, 1e-3))
            for key, value in (hard_wiener_bm3d_sigma_map or {}).items()
        }
        self.gather_frames = (
            self.wiener_merge_frames if self.wiener_front_end else self.input_frames
        )
        self.gather_reference_index = center_frame_index(self.gather_frames)
        self.rng = np.random.default_rng(seed)
        self._raw: dict[int, np.memmap] = {}
        self._arrays: dict[tuple[int, str], np.ndarray] = {}
        # Prefer low-fps / long-exposure clips when dark_boost > 0 (DES gap scenes).
        weights = np.ones(len(self.entries), dtype=np.float64)
        if self.dark_boost > 0.0:
            for i, entry in enumerate(self.entries):
                fps = max(float(entry.get("fps", 1.0)), 0.25)
                weights[i] = fps ** (-self.dark_boost)
        if self.hard_scene_substr and self.hard_scene_sample_mult > 1.0:
            for i, entry in enumerate(self.entries):
                if self._is_hard_entry(entry):
                    weights[i] *= self.hard_scene_sample_mult
        self._entry_probs = weights / weights.sum()

    def _entry_name(self, entry: dict) -> str:
        return str(entry.get("name") or Path(entry["path"]).name)

    def _is_hard_entry(self, entry: dict) -> bool:
        if not self.hard_scene_substr:
            return False
        name = self._entry_name(entry)
        path = str(entry.get("path", ""))
        return any(token in name or token in path for token in self.hard_scene_substr)

    def _is_edge_focus_entry(self, entry: dict) -> bool:
        if not self.edge_focus_substr:
            return False
        name = self._entry_name(entry)
        path = str(entry.get("path", ""))
        return any(token in name or token in path for token in self.edge_focus_substr)

    def _hard_token(self, entry: dict) -> str | None:
        name = self._entry_name(entry)
        path = str(entry.get("path", ""))
        for token in self.hard_scene_substr:
            if token in name or token in path:
                return token
        for token in self.hard_wiener_bm3d_sigma_map:
            if token in name or token in path:
                return token
        return None

    def _resolve_wiener_bm3d_sigma(self, entry: dict, is_hard: bool) -> float:
        token = self._hard_token(entry)
        if token is not None and token in self.hard_wiener_bm3d_sigma_map:
            return self.hard_wiener_bm3d_sigma_map[token]
        if is_hard and self.hard_wiener_bm3d_sigma is not None:
            return self.hard_wiener_bm3d_sigma
        return self.wiener_bm3d_sigma

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _array(self, index: int, key: str) -> np.ndarray:
        cache_key = (index, key)
        if cache_key not in self._arrays:
            self._arrays[cache_key] = np.load(
                self.entries[index][key],
                mmap_mode="r",
            )
        return self._arrays[cache_key]

    def _frames(self, index: int) -> np.memmap:
        if index not in self._raw:
            entry = self.entries[index]
            bytes_per_frame = entry["width"] * entry["height"] * 2
            frame_count = Path(entry["path"]).stat().st_size // bytes_per_frame
            self._raw[index] = np.memmap(
                entry["path"],
                mode="r",
                dtype="<u2",
                shape=(frame_count, entry["height"], entry["width"]),
            )
        return self._raw[index]

    @staticmethod
    def _aligned_raw_crop(
        raw_frame: np.ndarray,
        x0: int,
        y0: int,
        size: int,
        tx: float,
        ty: float,
    ) -> np.ndarray:
        source_center_x = x0 + (size - 1) * 0.5 - tx
        source_center_y = y0 + (size - 1) * 0.5 - ty
        source_x0 = int(math.floor(source_center_x - (size - 1) * 0.5)) - 2
        source_y0 = int(math.floor(source_center_y - (size - 1) * 0.5)) - 2
        region_size = size + 4
        region = raw_frame[
            source_y0 : source_y0 + region_size,
            source_x0 : source_x0 + region_size,
        ]
        if np.issubdtype(region.dtype, np.integer):
            region = decode_mono10(region)
        else:
            region = region.astype(np.float32, copy=False)
        local_center = (
            source_center_x - source_x0,
            source_center_y - source_y0,
        )
        return cv2.getRectSubPix(region, (size, size), local_center)

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        entry_index = int(self.rng.choice(len(self.entries), p=self._entry_probs))
        entry = self.entries[entry_index]
        frames = self._frames(entry_index)
        shifts = self._array(entry_index, "shifts")
        offsets = self._array(entry_index, "offsets")
        mask = self._array(entry_index, "mask")

        if self.center_mode == "anchor":
            center = int(entry["anchor_index"])
        else:
            min_center = self.gather_reference_index
            max_center = entry["frame_count"] - (
                self.gather_frames - self.gather_reference_index
            )
            center = int(self.rng.integers(min_center, max_center + 1))
        indices = [
            center - self.gather_reference_index + offset
            for offset in range(self.gather_frames)
        ]
        margin = int(np.ceil(np.abs(shifts).max())) + 4
        size = self.patch_size
        y0 = int(self.rng.integers(margin, entry["height"] - size - margin + 1))
        x0 = int(self.rng.integers(margin, entry["width"] - size - margin + 1))

        noisy_frames = []
        for frame_index in indices:
            # Cached shifts/offsets are relative to the sequence anchor. Convert
            # them to the current center-frame coordinate system so that train
            # and full-frame inference use exactly the same representation.
            tx = float(shifts[frame_index, 0] - shifts[center, 0])
            ty = float(shifts[frame_index, 1] - shifts[center, 1])
            brightness_offset = float(offsets[frame_index] - offsets[center])
            patch = self._aligned_raw_crop(
                frames[frame_index],
                x0,
                y0,
                size,
                tx,
                ty,
            )
            patch = np.clip(patch - brightness_offset, 0.0, RAW_MAX)
            noisy_frames.append(patch.astype(np.float32))

        if self.align_soft_gate:
            noisy_frames, _responses = soft_gate_burst(
                noisy_frames,
                self.gather_reference_index,
                threshold=self.align_threshold,
                temperature=self.align_temperature,
            )
            if self.align_drop_prob > 0.0:
                reference = noisy_frames[self.gather_reference_index]
                for index in range(len(noisy_frames)):
                    if index == self.gather_reference_index:
                        continue
                    if float(self.rng.random()) < self.align_drop_prob:
                        noisy_frames[index] = reference.copy()

        if self.wiener_front_end:
            from .wiener_merge import merge_burst_dn

            if self.wiener_fe_schedule in ("fps_sigma", "gated"):
                temporal, _meta = merge_burst_dn(
                    noisy_frames,
                    reference_index=self.gather_reference_index,
                    align=False,
                    tile_size=self.wiener_tile,
                    overlap=self.wiener_overlap,
                    c_factor=self.wiener_c_factor,
                    spatial_wiener=False,
                )
                from .fe_schedule import gated_spatial_from_temporal

                fps = float(entry.get("fps", 10.0))
                merged, _params, _sig = gated_spatial_from_temporal(
                    temporal,
                    fps=fps,
                    n_frames_averaged=len(noisy_frames),
                    tile_size=self.wiener_tile,
                    overlap=self.wiener_overlap,
                )
            else:
                merged, _meta = merge_burst_dn(
                    noisy_frames,
                    reference_index=self.gather_reference_index,
                    align=False,
                    tile_size=self.wiener_tile,
                    overlap=self.wiener_overlap,
                    c_factor=self.wiener_c_factor,
                    spatial_wiener=self.wiener_spatial,
                    spatial_c_factor=self.wiener_spatial_c_factor,
                    spatial_adaptive=self.wiener_spatial_adaptive,
                    spatial_flat_c_mult=self.wiener_spatial_flat_c_mult,
                    spatial_dark_boost=self.wiener_spatial_dark_boost,
                    spatial_edge_c_mult=self.wiener_spatial_edge_c_mult,
                    spatial_mask_harden=self.wiener_spatial_mask_harden,
                    spatial_freq_gamma=self.wiener_spatial_freq_gamma,
                )
            noisy_frames = [merged.copy() for _ in range(self.input_frames)]
            wiener_merged = merged
        else:
            wiener_merged = None
            if len(noisy_frames) != self.input_frames:
                # Should not happen without wiener front-end.
                raise RuntimeError(
                    f"Expected {self.input_frames} frames, got {len(noisy_frames)}"
                )
        input_set = set(indices)
        target_indices = sorted(
            (index for index in range(entry["frame_count"]) if index not in input_set),
            key=lambda index: abs(index - center),
        )[: self.target_frames]
        target_frames = []
        for frame_index in target_indices:
            tx = float(shifts[frame_index, 0] - shifts[center, 0])
            ty = float(shifts[frame_index, 1] - shifts[center, 1])
            brightness_offset = float(offsets[frame_index] - offsets[center])
            patch = self._aligned_raw_crop(
                frames[frame_index],
                x0,
                y0,
                size,
                tx,
                ty,
            )
            patch = np.clip(patch - brightness_offset, 0.0, RAW_MAX)
            target_frames.append(patch.astype(np.float32))
        target_stack = np.stack(target_frames, axis=0)
        target_stack.sort(axis=0)
        trim = max(1, int(len(target_frames) * 0.10))
        trimmed = target_stack[trim:-trim]
        target_dn = trimmed.mean(axis=0).astype(np.float32)
        temporal_variance = trimmed.var(axis=0)
        expected_variance = PTC_SLOPE * target_dn + PTC_INTERCEPT
        online_mask = temporal_variance <= 2.5 * np.maximum(expected_variance, 1.0)
        cached_mask = self._aligned_raw_crop(
            mask,
            x0,
            y0,
            size,
            float(-shifts[center, 0]),
            float(-shifts[center, 1]),
        )
        mask_patch = cached_mask * online_mask.astype(np.float32)
        exposure_ms = float(entry["exposure_ms"])
        teacher_valid = entry.get("teacher") is not None
        is_hard = self._is_hard_entry(entry)
        is_edge_focus = self._is_edge_focus_entry(entry)
        use_wiener_bm3d = (
            self.wiener_bm3d_teacher
            and wiener_merged is not None
            and float(self.rng.random()) < self.wiener_bm3d_prob
        )
        if use_wiener_bm3d:
            try:
                import bm3d
            except ImportError as error:
                raise RuntimeError(
                    "Wiener BM3D teacher requires: pip install bm3d"
                ) from error

            intercept = exposure_intercept(
                exposure_ms,
                dark_variance_per_s=DEFAULT_DARK_VARIANCE_PER_S,
            )
            transformed = vst_forward(wiener_merged, intercept=intercept)
            sigma_psd = self._resolve_wiener_bm3d_sigma(entry, is_hard)
            denoised = bm3d.bm3d(
                transformed.astype(np.float64),
                sigma_psd=sigma_psd,
            ).astype(np.float32)
            teacher_dn = np.clip(
                vst_inverse(denoised, intercept=intercept),
                0.0,
                RAW_MAX,
            ).astype(np.float32)
            teacher_valid = True
        elif teacher_valid:
            teacher_array = self._array(entry_index, "teacher")
            # The cached teacher is in anchor coordinates/brightness. Reproject
            # it into the sampled center frame before distillation.
            teacher_dn = self._aligned_raw_crop(
                teacher_array,
                x0,
                y0,
                size,
                float(-shifts[center, 0]),
                float(-shifts[center, 1]),
            )
            teacher_dn = np.clip(
                teacher_dn + float(offsets[center]),
                0.0,
                RAW_MAX,
            ).astype(np.float32)
        else:
            teacher_dn = target_dn

        arrays = apply_real_flips(
            [*noisy_frames, target_dn, teacher_dn, mask_patch],
            horizontal=bool(self.rng.integers(2)),
            vertical=bool(self.rng.integers(2)),
        )
        noisy_frames = arrays[: self.input_frames]
        target_dn, teacher_dn, mask_patch = arrays[self.input_frames :]
        burst = make_burst_tensor(
            noisy_frames,
            exposure_ms,
            PTC_SLOPE,
            PTC_INTERCEPT,
            black_level_dn=DEFAULT_BLACK_LEVEL_DN,
            dark_variance_per_s=DEFAULT_DARK_VARIANCE_PER_S,
            postmerge_calib=(
                self.postmerge_calib if self.wiener_front_end else None
            ),
            measured_fe_sigma=(
                self.measured_fe_sigma if self.wiener_front_end else False
            ),
        )
        target = raw_to_model_input(
            target_dn,
            exposure_ms=exposure_ms,
            dark_variance_per_s=DEFAULT_DARK_VARIANCE_PER_S,
        )[None]
        teacher = raw_to_model_input(
            teacher_dn,
            exposure_ms=exposure_ms,
            dark_variance_per_s=DEFAULT_DARK_VARIANCE_PER_S,
        )[None]
        fps = max(float(entry.get("fps", self.scene_loss_ref_fps)), 0.25)
        if self.scene_loss_power > 0.0:
            loss_weight = float(
                np.clip(
                    (self.scene_loss_ref_fps / fps) ** self.scene_loss_power,
                    1.0,
                    self.scene_loss_max,
                )
            )
        else:
            loss_weight = 1.0
        if is_hard and self.hard_scene_loss_mult > 1.0:
            loss_weight *= self.hard_scene_loss_mult
        return {
            "burst": torch.from_numpy(burst).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "teacher": torch.from_numpy(np.ascontiguousarray(teacher)).float(),
            "mask": torch.from_numpy(mask_patch[None].copy()).float(),
            "teacher_valid": torch.tensor(float(teacher_valid), dtype=torch.float32),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
            "is_hard": torch.tensor(1.0 if is_hard else 0.0, dtype=torch.float32),
            "is_edge_focus": torch.tensor(
                1.0 if is_edge_focus else 0.0, dtype=torch.float32
            ),
        }


class VideoBurstSyntheticDataset(Dataset):
    """Consecutive public-video frames with Mono10 PTC/black-level noise synthesis."""

    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int = 256,
        input_frames: int = 16,
        samples_per_epoch: int = 4000,
        noise_jitter: float = 0.15,
        exposure_ms_range: tuple[float, float] = (10.0, 2000.0),
        black_level_dn: float = DEFAULT_BLACK_LEVEL_DN,
        black_drift_sigma: float = 1.0,
        fixed_pattern_sigma: float = 0.5,
        row_noise_sigma: float = 0.3,
        column_noise_sigma: float = 0.2,
        dark_variance_per_s_range: tuple[float, float] = (
            0.0,
            DEFAULT_DARK_VARIANCE_PER_S * 2.0,
        ),
        motion_blur_strength: float = 1.0,
        seed: int = 0,
    ):
        self.entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.patch_size = patch_size
        self.input_frames = input_frames
        self.reference_index = center_frame_index(input_frames)
        self.samples_per_epoch = samples_per_epoch
        self.noise_jitter = noise_jitter
        self.exposure_ms_range = exposure_ms_range
        self.black_level_dn = black_level_dn
        self.black_drift_sigma = max(0.0, black_drift_sigma)
        self.fixed_pattern_sigma = max(0.0, fixed_pattern_sigma)
        self.row_noise_sigma = max(0.0, row_noise_sigma)
        self.column_noise_sigma = max(0.0, column_noise_sigma)
        self.dark_variance_per_s_range = dark_variance_per_s_range
        self.motion_blur_strength = max(0.0, motion_blur_strength)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_gray(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read frame {path}")
        return image.astype(np.float32)

    def _exposure_blur(self, image: np.ndarray, exposure_ms: float) -> np.ndarray:
        # Longer exposures get stronger motion blur; keep it mild for short exposures.
        kernel = int(
            round(
                self.motion_blur_strength
                * max(1.0, min(7.0, exposure_ms / 400.0))
            )
        )
        if kernel <= 1:
            return image
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        entry = self.entries[int(self.rng.integers(len(self.entries)))]
        frames = entry["frames"]
        if len(frames) < self.input_frames:
            raise ValueError(f"{entry['name']}: not enough frames")
        start = int(self.rng.integers(0, len(frames) - self.input_frames + 1))
        selected = frames[start : start + self.input_frames]

        clean_frames = [self._load_gray(path) for path in selected]
        height, width = clean_frames[0].shape
        size = self.patch_size
        if height < size + 8 or width < size + 8:
            raise ValueError(f"{entry['name']}: frame smaller than patch")
        y0 = int(self.rng.integers(4, height - size - 3))
        x0 = int(self.rng.integers(4, width - size - 3))
        clean_frames = [frame[y0 : y0 + size, x0 : x0 + size] for frame in clean_frames]

        reference = clean_frames[self.reference_index]
        low, high = np.percentile(reference, (1.0, 99.0))
        scale = max(high - low, 1.0)
        clean_frames = [
            np.clip((frame - low) / scale, 0.0, 1.0) for frame in clean_frames
        ]
        minimum_mean = max(100.0, self.black_level_dn + 20.0)
        target_mean = float(self.rng.uniform(minimum_mean, 950.0))
        signal_mean = max(float(np.mean(clean_frames[self.reference_index])), 1e-6)
        gain = (target_mean - self.black_level_dn) / signal_mean
        clean_frames = [
            np.clip(self.black_level_dn + frame * gain, self.black_level_dn, RAW_MAX).astype(
                np.float32
            )
            for frame in clean_frames
        ]

        jitter_low = 1.0 - self.noise_jitter
        jitter_high = 1.0 + self.noise_jitter
        slope = PTC_SLOPE * float(self.rng.uniform(jitter_low, jitter_high))
        intercept = PTC_INTERCEPT * float(self.rng.uniform(jitter_low, jitter_high))
        exposure_ms = float(self.rng.uniform(*self.exposure_ms_range))
        dark_variance_per_s = float(
            self.rng.uniform(*self.dark_variance_per_s_range)
        )
        flicker = float(self.rng.normal(0.0, 0.01))

        fixed_pattern = self.rng.normal(
            0.0,
            self.fixed_pattern_sigma,
            size=(size, size),
        ).astype(np.float32)
        noisy_frames = []
        for frame_index, clean in enumerate(clean_frames):
            clean = np.clip(clean * (1.0 + flicker), 0.0, RAW_MAX)
            clean = self._exposure_blur(clean, exposure_ms)
            noisy = poisson_gaussian_noise(
                clean,
                self.rng,
                slope,
                intercept,
                exposure_ms=exposure_ms,
                black_level_dn=self.black_level_dn,
                dark_variance_per_s=dark_variance_per_s,
            )
            frame_offset = float(self.rng.normal(0.0, self.black_drift_sigma))
            row_noise = self.rng.normal(
                0.0,
                self.row_noise_sigma,
                size=(size, 1),
            ).astype(np.float32)
            column_noise = self.rng.normal(
                0.0,
                self.column_noise_sigma,
                size=(1, size),
            ).astype(np.float32)
            noisy = np.clip(
                noisy + fixed_pattern + frame_offset + row_noise + column_noise,
                0.0,
                RAW_MAX,
            )
            noisy_frames.append(np.rint(noisy).astype(np.float32))

        target = clean_frames[self.reference_index]
        mode = int(self.rng.integers(8))
        transformed = apply_d4([*noisy_frames, target], mode)
        noisy_frames = transformed[:-1]
        target = transformed[-1]
        burst = make_burst_tensor(
            noisy_frames,
            exposure_ms,
            slope,
            intercept,
            black_level_dn=self.black_level_dn,
            dark_variance_per_s=dark_variance_per_s,
        )
        target_vst = raw_to_model_input(
            target,
            exposure_ms=exposure_ms,
            dark_variance_per_s=dark_variance_per_s,
        )[None]
        mask = np.ones_like(target_vst, dtype=np.float32)
        return {
            "burst": torch.from_numpy(burst).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target_vst)).float(),
            "teacher": torch.from_numpy(np.ascontiguousarray(target_vst)).float(),
            "mask": torch.from_numpy(mask).float(),
            "teacher_valid": torch.tensor(0.0, dtype=torch.float32),
        }
