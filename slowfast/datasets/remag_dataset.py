import os
import glob
import json
import random
import torch
import torch.utils.data
import numpy as np
from collections import Counter
from slowfast.utils.env import pathmgr
from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets import utils as data_utils

from PIL import Image
import time

def load_tensor_from_image(path):
    # try:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    
    # Fix any case where .convert("RGB") didn't result in 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)  

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    #assert the shape is ((224, 224, 3))
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"[ERROR] Loaded tensor has unexpected shape: {tensor.shape}")
    return tensor

@DATASET_REGISTRY.register()
class REMAGDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self._num_frames = 64
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._frame_dir = cfg.DATA.PATH_TO_DATA_DIR
        self._num_retries = 1000
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

        for folder in folders:
            total_folders += 1
            folder_path = os.path.join(self._frame_dir, folder)

            if not os.path.isdir(folder_path):
                print(f"[WARN] {folder_path} is not a directory. Skipping.")
                skipped_folders += 1
                continue

            # 1) try the standard labels.json
            label_path = os.path.join(folder_path, "labels.json")
            if os.path.isfile(label_path):
                pass

            else:
                # 2) fallback: resolve symlink by look for any .json in its parent
                resolved = os.path.realpath(folder_path)
                parent = os.path.dirname(resolved)
                candidates = glob.glob(os.path.join(parent, "*.json"))

                if candidates:
                    chosen = os.path.basename(candidates[0])
                    label_path = candidates[0]
                else:
                    # 3) still nothing, skip
                    skipped_folders += 1
                    print(f"[WARN] No labels.json found in {folder_path} or its parent.")
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

                label_histogram[most_common_label] += 1
                if self.mode == "train" and most_common_label in self._test_classes:
                    continue
                if self.mode == "test" and most_common_label not in self._test_classes:
                    continue

                if count / self._num_frames >= self._min_agreement_ratio:
                    valid_indices.append((i, most_common_label))

            if valid_indices:
                self._video_metadata.append((folder_path, frame_names, activity_ids, valid_indices))
                total_valid_clips += len(valid_indices)

        print(f"[SUMMARY] Total folders: {total_folders}")
        print(f"[SUMMARY] Skipped folders: {skipped_folders}")
        print(f"[SUMMARY] Kept folders: {len(self._video_metadata)}")
        print(f"[SUMMARY] Total valid clips: {total_valid_clips}")
        print(f"[SUMMARY] Label histogram (post-filter): {dict(label_histogram)}")

    def __getitem__(self, index):
        for retry in range(self._num_retries):
            # try:
            folder_path, frame_names, _, valid_indices = self._video_metadata[index]
            start_idx, label = random.choice(valid_indices)
            frame_list = frame_names[start_idx:start_idx + self._num_frames]

            tensors = []
            for fname in frame_list:
                path = os.path.join(folder_path, fname)
                tensor = load_tensor_from_image(path)
                tensors.append(tensor)

            frames = torch.stack(tensors, dim=1)

            # Multigrid cropping logic
            if self._enable_multigrid and self.mode == "train":
                short_cycle_idx = getattr(self.cfg, "SHORT_CYCLE_IDX", None)
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
                if short_cycle_idx in [0, 1]:
                    crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
                min_scale = int(round(float(self.cfg.DATA.TRAIN_JITTER_SCALES[0]) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))
            else:
                crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
                min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]

            spatial_sample_index = (
                index % self.cfg.TEST.NUM_SPATIAL_CROPS if self._enable_test_crops and self.mode == "test"
                else -1 if self.mode == "train" else 1
            )

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

            # ======== DEBUG BLOCK =========
            # print("\n[DEBUG] Returning from __getitem__()")
            # if isinstance(frames, (list, tuple)):
            #     print(f"frames: list of {len(frames)} pathways")
            #     for i, f in enumerate(frames):
            #         print(f"  - Pathway {i} shape: {f.shape}, dtype: {f.dtype}")
            # else:
            #     print(f"frames shape: {frames.shape}, dtype: {frames.dtype}")

            # print(f"label: {label} (type: {type(label)})")
            # print(f"index: {index}")

            return [frames], label, index, 0, {}

    def __len__(self):
        folder_count = len(self._video_metadata)
        clip_count = sum(len(entry[3]) for entry in self._video_metadata)
        print(f"[DEBUG] Dataset length called. Folders: {folder_count}, Total valid clips: {clip_count}")
        return folder_count
