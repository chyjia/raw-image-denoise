"""VST-domain supervised patch dataset for denoising.

Input  : noisy frame (normalized VST), exposure condition, optional frame-brightness
         condition (handles light flicker across frames).
Target : clean temporal reference aligned to the current frame brightness, in VST.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import common


class VstDenoiseDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        patch_size: int = 128,
        samples_per_epoch: int = 4000,
        augment: bool = True,
        seed: int = 0,
        flat_roi_fraction: float = 0.0,
        brightness_condition: bool = True,
        brightness_offset: bool = True,
        exposure_priors_path: Path | None = None,
        align_target_brightness: bool = True,
        exposure_boost_ms: float = 50.0,
        exposure_boost_sigma: float = 15.0,
        fps20_boost: float = 3.0,
        exposure_boost: float = 3.0,
        flicker_boost: float = 1.5,
    ) -> None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.entries = manifest["entries"]
        if not self.entries:
            raise ValueError("Manifest has no entries.")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.flat_roi_fraction = float(np.clip(flat_roi_fraction, 0.0, 1.0))
        self.brightness_condition = brightness_condition
        self.brightness_offset = brightness_offset
        self.align_target_brightness = align_target_brightness
        priors_path = exposure_priors_path or Path(manifest_path).parent / "exposure_brightness_priors.json"
        self.exposure_priors = common.load_exposure_brightness_priors(priors_path)
        if brightness_condition and brightness_offset and not self.exposure_priors:
            raise ValueError(
                f"Missing exposure brightness priors at {priors_path}. "
                "Run build_references.py first."
            )

        self.clean_dn_cache: list[np.ndarray] = []
        self.meta: list[common.RawMeta] = []
        self.raw_paths: list[Path] = []
        sample_weights = []
        for entry in self.entries:
            clean_dn = np.load(entry["ref_path"])
            self.clean_dn_cache.append(clean_dn.astype(np.float32))
            self.meta.append(
                common.RawMeta(
                    width=entry["width"],
                    height=entry["height"],
                    fps=entry["fps"],
                    exposure_ms=entry["exposure_ms"],
                )
            )
            self.raw_paths.append(Path(entry["raw_path"]))

            weight = 1.0
            if abs(entry["exposure_ms"] - exposure_boost_ms) <= exposure_boost_sigma:
                weight *= exposure_boost
            if abs(entry["fps"] - 20.0) < 0.5:
                weight *= fps20_boost
            flicker_std = float(entry.get("frame_roi_mean_std", 0.0))
            if flicker_std > 2.0:
                weight *= flicker_boost
            sample_weights.append(weight)

        weights = np.array(sample_weights, dtype=np.float64)
        self.sample_probs = weights / weights.sum()
        self._raw_handles: dict[int, np.memmap] = {}

    @property
    def inp_channels(self) -> int:
        return 3 if self.brightness_condition else 2

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _raw(self, file_index: int) -> np.memmap:
        handle = self._raw_handles.get(file_index)
        if handle is None:
            handle = common.open_raw(self.raw_paths[file_index], self.meta[file_index])
            self._raw_handles[file_index] = handle
        return handle

    def __getitem__(self, _: int):
        file_index = int(self.rng.choice(len(self.entries), p=self.sample_probs))
        raw = self._raw(file_index)
        frame_count, height, width = raw.shape
        frame_index = int(self.rng.integers(frame_count))

        patch = self.patch_size
        if self.flat_roi_fraction > 0 and self.rng.random() < self.flat_roi_fraction:
            cy0, cx0, roi_h, roi_w = common.central_roi_bounds(height, width, 0.4)
            y0 = int(self.rng.integers(cy0, max(cy0 + 1, cy0 + roi_h - patch + 1)))
            x0 = int(self.rng.integers(cx0, max(cx0 + 1, cx0 + roi_w - patch + 1)))
        else:
            y0 = int(self.rng.integers(0, height - patch + 1))
            x0 = int(self.rng.integers(0, width - patch + 1))

        frame_dn = common.decode_frame(raw[frame_index])
        noisy_dn = frame_dn[y0 : y0 + patch, x0 : x0 + patch]
        noisy = common.raw_to_model_input(noisy_dn).astype(np.float32)

        clean_dn = self.clean_dn_cache[file_index]
        if self.align_target_brightness:
            clean_dn = common.align_reference_to_frame_brightness(clean_dn, frame_dn)
        clean_dn_patch = clean_dn[y0 : y0 + patch, x0 : x0 + patch]
        clean = common.raw_to_model_input(clean_dn_patch).astype(np.float32)

        if self.augment:
            k = int(self.rng.integers(4))
            noisy = np.rot90(noisy, k).copy()
            clean = np.rot90(clean, k).copy()
            if self.rng.random() < 0.5:
                noisy = np.fliplr(noisy).copy()
                clean = np.fliplr(clean).copy()

        exposure = self.entries[file_index]["exposure_condition"]
        exposure_channel = np.full((patch, patch), exposure, dtype=np.float32)
        channels = [noisy, exposure_channel]
        if self.brightness_condition:
            roi_mean = common.central_roi_mean(frame_dn)
            if self.brightness_offset:
                brightness = common.brightness_offset_to_condition(
                    roi_mean, self.entries[file_index]["exposure_ms"], self.exposure_priors
                )
            else:
                brightness = common.brightness_to_condition(roi_mean)
            brightness_channel = np.full((patch, patch), brightness, dtype=np.float32)
            channels.append(brightness_channel)

        input_tensor = torch.from_numpy(np.stack(channels, axis=0))
        target_tensor = torch.from_numpy(clean[None, :, :].copy())
        return input_tensor, target_tensor
