"""Synthetic and real 16-frame datasets for BurstNAFNet."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    PTC_INTERCEPT,
    PTC_SLOPE,
    RAW_MAX,
    center_frame_index,
    decode_mono10,
    noise_sigma_map,
    normalize_exposure,
    poisson_gaussian_noise,
    raw_to_model_input,
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
) -> np.ndarray:
    exposure = normalize_exposure(exposure_ms)
    channels = []
    for frame in frames_dn:
        vst = raw_to_model_input(frame)
        sigma = noise_sigma_map(frame, slope, intercept)
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
        black_level_dn: float = 0.0,
        black_drift_sigma: float = 0.0,
        fixed_pattern_sigma: float = 0.0,
        row_noise_sigma: float = 0.0,
        column_noise_sigma: float = 0.0,
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
            noisy = poisson_gaussian_noise(clean, self.rng, slope, intercept)
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
        )
        target = raw_to_model_input(clean)[None]
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
    ):
        self.entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.patch_size = patch_size
        self.input_frames = input_frames
        self.reference_index = center_frame_index(input_frames)
        self.samples_per_epoch = samples_per_epoch
        if center_mode not in {"anchor", "random"}:
            raise ValueError("center_mode must be 'anchor' or 'random'")
        self.center_mode = center_mode
        self.rng = np.random.default_rng(seed)
        self._raw: dict[int, np.memmap] = {}
        self._arrays: dict[tuple[int, str], np.ndarray] = {}

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
        entry_index = int(self.rng.integers(len(self.entries)))
        entry = self.entries[entry_index]
        frames = self._frames(entry_index)
        shifts = self._array(entry_index, "shifts")
        offsets = self._array(entry_index, "offsets")
        mask = self._array(entry_index, "mask")

        if self.center_mode == "anchor":
            center = int(entry["anchor_index"])
        else:
            min_center = self.reference_index
            max_center = entry["frame_count"] - (self.input_frames - self.reference_index)
            center = int(self.rng.integers(min_center, max_center + 1))
        indices = [
            center - self.reference_index + offset
            for offset in range(self.input_frames)
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

        # Build a leave-burst-out target from the 32 nearest other frames.
        input_set = set(indices)
        target_indices = sorted(
            (index for index in range(entry["frame_count"]) if index not in input_set),
            key=lambda index: abs(index - center),
        )[:32]
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
        teacher_valid = entry.get("teacher") is not None
        if teacher_valid:
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
            float(entry["exposure_ms"]),
            PTC_SLOPE,
            PTC_INTERCEPT,
        )
        target = raw_to_model_input(target_dn)[None]
        teacher = raw_to_model_input(teacher_dn)[None]
        return {
            "burst": torch.from_numpy(burst).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "teacher": torch.from_numpy(np.ascontiguousarray(teacher)).float(),
            "mask": torch.from_numpy(mask_patch[None].copy()).float(),
            "teacher_valid": torch.tensor(float(teacher_valid), dtype=torch.float32),
        }
