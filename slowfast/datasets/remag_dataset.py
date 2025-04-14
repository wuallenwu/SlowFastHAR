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
        self._enable_test_crops = cfg.TEST.NUM_SPATIAL_CROPS > 1
        self._test_classes = getattr(cfg.DATA, "TEST_CLASS_IDS", [5, 6, 8])
        self._enable_multigrid = getattr(cfg.MULTIGRID, "ENABLE", False)
        self._construct_loader()

    def _construct_loader(self):
        self._video_metadata = []
        total_folders = 0
        skipped_folders = 0
        total_valid_clips = 0
        label_histogram = Counter()

        folders = sorted(os.listdir(self._frame_dir))
        print(f"[INFO] Found {len(folders)} folders in: {self._frame_dir}")

        for folder in folders:
            total_folders += 1
            folder_path = os.path.join(self._frame_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"[WARN] Skipping non-directory: {folder_path}")
                skipped_folders += 1
                continue

            label_path = os.path.join(folder_path, "labels.json")
            if not os.path.isfile(label_path):
                alt_path = os.path.join(os.path.dirname(folder_path), "labels.json")
                if os.path.isfile(alt_path):
                    label_path = alt_path
                else:
                    print(f"[WARN] No labels.json found for {folder_path}, skipping.")
                    skipped_folders += 1
                    continue

            try:
                with open(label_path, "r") as f:
                    labels = json.load(f)
                frame_names = labels["file_names"]
                activity_ids = labels["activity_ids"]
                assert len(frame_names) == len(activity_ids), f"[ERROR] Mismatched frame/activity length in {folder_path}"
            except Exception as e:
                print(f"[ERROR] Failed to load labels.json for {folder_path}: {e}")
                skipped_folders += 1
                continue

            valid_indices = []
            for i in range(0, len(frame_names) - self._num_frames + 1):
                window = activity_ids[i:i + self._num_frames]
                if window.count(-1) > 0:
                    continue
                label_counts = Counter(window)
                most_common_label, count = label_counts.most_common(1)[0]

                # Track label frequency
                label_histogram[most_common_label] += 1

                # Class filtering
                if self.mode == "train" and most_common_label in self._test_classes:
                    continue
                if self.mode == "test" and most_common_label not in self._test_classes:
                    continue

                if count / self._num_frames >= self._min_agreement_ratio:
                    valid_indices.append((i, most_common_label))

            if valid_indices:
                self._video_metadata.append((folder_path, frame_names, activity_ids, valid_indices))
                total_valid_clips += len(valid_indices)
                print(f"[INFO] {folder_path}: {len(valid_indices)} valid clips.")
            else:
                print(f"[INFO] {folder_path}: 0 valid clips.")

        print(f"[SUMMARY] Total folders: {total_folders}")
        print(f"[SUMMARY] Skipped folders: {skipped_folders}")
        print(f"[SUMMARY] Kept folders: {len(self._video_metadata)}")
        print(f"[SUMMARY] Total valid clips: {total_valid_clips}")
        print(f"[SUMMARY] Label histogram (post-filter): {dict(label_histogram)}")

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
            except Exception as e:
                print(f"[WARN] Exception during __getitem__: {e}")
                index = random.randint(0, len(self._video_metadata) - 1)
                continue
        raise RuntimeError(f"Failed to load data after {self._num_retries} retries")

    def __len__(self):
        folder_count = len(self._video_metadata)
        clip_count = sum(len(entry[3]) for entry in self._video_metadata)
        print(f"[DEBUG] Dataset length called. Folders: {folder_count}, Total valid clips: {clip_count}")
        return folder_count
