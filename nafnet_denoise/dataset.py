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
        cache_images_in_ram: bool = False,
        seed: int = 0,
    ):
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.exposure_ms_range = exposure_ms_range
        self.input_frames = input_frames
        self.center_index = center_frame_index(input_frames)
        self.noise_jitter = noise_jitter
        self.cache_images_in_ram = cache_images_in_ram
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

        target_mean = float(self.rng.uniform(TARGET_MEAN_DN_MIN, TARGET_MEAN_DN_MAX))
        current_mean = float(patch.mean())
        clean = np.clip(patch * (target_mean / max(current_mean, 1e-6)), 0.0, 1023.0).astype(
            np.float32
        )

        # One perturbed camera noise model is shared by the whole burst. This
        # broadens the calibrated PTC distribution without creating an
        # unrealistic per-frame camera response change.
        jitter_low = 1.0 - self.noise_jitter
        jitter_high = 1.0 + self.noise_jitter
        slope = PTC_SLOPE * float(self.rng.uniform(jitter_low, jitter_high))
        intercept = PTC_INTERCEPT * float(self.rng.uniform(jitter_low, jitter_high))
        noisy_frames = [
            poisson_gaussian_noise(clean, self.rng, slope, intercept)
            for _ in range(self.input_frames)
        ]
        exposure_ms = float(self.rng.uniform(*self.exposure_ms_range))

        k = int(self.rng.integers(4))
        rotated_noisy = [np.rot90(frame, k) for frame in noisy_frames]
        clean_rot = np.rot90(clean, k)
        if self.rng.random() < 0.5:
            rotated_noisy = [np.fliplr(frame) for frame in rotated_noisy]
            clean_rot = np.fliplr(clean_rot)

        model_input = frames_to_model_input(
            rotated_noisy,
            exposure_ms,
            reference_index=self.center_index,
        )
        clean_vst = raw_to_model_input(clean_rot)

        input_tensor = torch.from_numpy(model_input)
        target_tensor = torch.from_numpy(np.ascontiguousarray(clean_vst[None, :, :]))
        return input_tensor.float(), target_tensor.float()
