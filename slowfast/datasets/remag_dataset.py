import os
import json
import random
import torch
import torch.utils.data
import numpy as np
from collections import Counter
from slowfast.utils.env import pathmgr
from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets import utils as data_utils

@DATASET_REGISTRY.register()
class REMAGDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self._num_frames = 64
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._frame_dir = cfg.DATA.PATH_TO_DATA_DIR
        self._num_retries = 10
        self._min_agreement_ratio = 0.8
        self._enable_multigrid = cfg.MULTIGRID.ENABLE
        self._enable_test_crops = cfg.TEST.NUM_SPATIAL_CROPS > 1
        self._test_classes = getattr(cfg.DATA, "TEST_CLASS_IDS", [5, 6, 8])
        self._construct_loader()

    def _construct_loader(self):
        self._video_metadata = []

        folders = sorted(os.listdir(self._frame_dir))
        for folder in folders:
            folder_path = os.path.join(self._frame_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            label_path = os.path.join(folder_path, "labels.json")
            if not os.path.isfile(label_path):
                label_path = os.path.join(os.path.dirname(folder_path), "labels.json")
            if not os.path.isfile(label_path):
                continue

            try:
                with open(label_path, "r") as f:
                    labels = json.load(f)
                frame_names = labels["file_names"]
                activity_ids = labels["activity_ids"]
            except Exception:
                continue

            valid_indices = []
            for i in range(0, len(frame_names) - self._num_frames + 1):
                window = activity_ids[i:i + self._num_frames]
                if window.count(-1) > 0:
                    continue
                label_counts = Counter(window)
                most_common_label, count = label_counts.most_common(1)[0]

                # Class-wise split logic
                if self.mode == "train" and most_common_label in self._test_classes:
                    continue
                if self.mode == "test" and most_common_label not in self._test_classes:
                    continue

                if count / self._num_frames >= self._min_agreement_ratio:
                    valid_indices.append((i, most_common_label))

            if valid_indices:
                self._video_metadata.append((folder_path, frame_names, activity_ids, valid_indices))

    def __getitem__(self, index):
        for _ in range(self._num_retries):
            try:
                folder_path, frame_names, _, valid_indices = self._video_metadata[index]
                start_idx, label = random.choice(valid_indices)
                frame_list = frame_names[start_idx:start_idx + self._num_frames]
                frames = [
                    data_utils.retry_load_image(os.path.join(folder_path, fname))
                    for fname in frame_list
                ]
                frames = torch.stack([
                    data_utils.tensor_normalize(
                        torch.from_numpy(frame).permute(2, 0, 1).float(),
                        self.cfg.DATA.MEAN,
                        self.cfg.DATA.STD
                    ) / 255.0 for frame in frames
                ], dim=1)

                if self._enable_multigrid and self.mode == "train":
                    short_cycle_idx = getattr(self.cfg, "SHORT_CYCLE_IDX", None)
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
                    if short_cycle_idx in [0, 1]:
                        crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
                    min_scale = int(round(float(self.cfg.DATA.TRAIN_JITTER_SCALES[0]) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))
                else:
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
                    min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]

                if self._enable_test_crops and self.mode == "test":
                    spatial_sample_index = index % self.cfg.TEST.NUM_SPATIAL_CROPS
                else:
                    spatial_sample_index = -1 if self.mode == "train" else 1

                frames = data_utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=self.cfg.DATA.TRAIN_JITTER_SCALES[1],
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    aspect_ratio=self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE if self.mode == "train" else None,
                    scale=self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE if self.mode == "train" else None,
                    motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode == "train" else False,
                )

                frames = data_utils.pack_pathway_output(self.cfg, frames)
                return frames, label, index, 0, {}
            except Exception:
                index = random.randint(0, len(self._video_metadata) - 1)
                continue
        raise RuntimeError(f"Failed to load data after {self._num_retries} retries")

    def __len__(self):
        return len(self._video_metadata)
