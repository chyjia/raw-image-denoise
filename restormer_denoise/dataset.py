"""PyTorch dataset that samples VST-domain noisy/clean patch pairs.

Each sample is a single noisy frame (input) paired with the cached clean
temporal mean (target). Both are brightness-aligned, VST-transformed, and
normalized. A constant exposure channel conditions the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    decode_mono10,
    memmap_frames,
    normalize_exposure,
    raw_to_model_input,
)


class VstPatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int = 128,
        patches_per_epoch: int = 4000,
        cache_clean_in_ram: bool = True,
        seed: int = 0,
    ):
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.cache_clean_in_ram = cache_clean_in_ram
        self.rng = np.random.default_rng(seed)

        self.entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not self.entries:
            raise ValueError(f"Empty manifest: {manifest_path}")

        self._frames: dict[int, np.memmap] = {}
        self._clean: dict[int, np.ndarray] = {}
        self._offsets: dict[int, np.ndarray] = {}
        if cache_clean_in_ram:
            for index, entry in enumerate(self.entries):
                self._clean[index] = np.load(entry["clean"]).astype(np.float32)
                self._offsets[index] = np.load(entry["offsets"]).astype(np.float32)

    def __len__(self) -> int:
        return self.patches_per_epoch

    def _get_frames(self, index: int) -> np.memmap:
        frames = self._frames.get(index)
        if frames is None:
            entry = self.entries[index]
            frames = memmap_frames(Path(entry["path"]), entry["width"], entry["height"])
            self._frames[index] = frames
        return frames

    def _get_clean(self, index: int) -> np.ndarray:
        if index in self._clean:
            return self._clean[index]
        return np.load(self.entries[index]["clean"]).astype(np.float32)

    def _get_offsets(self, index: int) -> np.ndarray:
        if index in self._offsets:
            return self._offsets[index]
        return np.load(self.entries[index]["offsets"]).astype(np.float32)

    def __getitem__(self, _: int):
        video_index = int(self.rng.integers(len(self.entries)))
        entry = self.entries[video_index]
        frames = self._get_frames(video_index)
        clean = self._get_clean(video_index)
        offsets = self._get_offsets(video_index)

        frame_count, height, width = frames.shape
        size = self.patch_size
        y0 = int(self.rng.integers(0, height - size + 1))
        x0 = int(self.rng.integers(0, width - size + 1))
        frame_index = int(self.rng.integers(frame_count))

        noisy_patch = decode_mono10(
            frames[frame_index, y0 : y0 + size, x0 : x0 + size]
        )
        noisy_patch = noisy_patch - float(offsets[frame_index])
        clean_patch = clean[y0 : y0 + size, x0 : x0 + size]

        noisy_vst = raw_to_model_input(noisy_patch)
        clean_vst = raw_to_model_input(clean_patch)

        # Random 8-fold dihedral augmentation shared by input and target.
        k = int(self.rng.integers(4))
        noisy_vst = np.rot90(noisy_vst, k)
        clean_vst = np.rot90(clean_vst, k)
        if self.rng.random() < 0.5:
            noisy_vst = np.fliplr(noisy_vst)
            clean_vst = np.fliplr(clean_vst)

        exposure = normalize_exposure(entry["exposure_ms"])
        exposure_channel = np.full_like(noisy_vst, exposure, dtype=np.float32)

        input_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack([noisy_vst, exposure_channel], axis=0))
        )
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(clean_vst[None, :, :])
        )
        return input_tensor.float(), target_tensor.float()
