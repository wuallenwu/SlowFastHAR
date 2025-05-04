import os
import re
import glob
import json
import random
import torch
import torch.utils.data
import numpy as np
from collections import Counter
from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets import utils as data_utils

from PIL import Image

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
        self._num_frames = cfg.DATA.NUM_FRAMES
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._frame_dir = cfg.DATA.PATH_TO_DATA_DIR
        self._num_retries = 100
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

            match = re.search(r"Subj\+0*([0-9]+)", folder)
            if not match:
                continue  
            subject_id = int(match.group(1))
            is_test = subject_id in self._test_classes
            if (self.mode == "train" and is_test) or (self.mode == "test" and not is_test):
                continue  
            # print(f"[DEBUG] Mode: {self.mode}, using TEST_CLASS_IDS: {self._test_classes}")
            # print(f"[DEBUG] Loaded {len(self._video_metadata)} folders with valid clips")


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

            # --- Check and skip folders containing confound classes [6-11] ---
            if any(6 <= label <= 11 for label in activity_ids):
                skipped_folders += 1
                continue

            # --- Relabel labels in [11,12,13,14,15] by subtracting 5 ---
            activity_ids = [label - 5 if label in {11, 12, 13, 14, 15} else label for label in activity_ids]

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
        self._flat_index = []
        for folder_idx, (_, _, _, valid_clips) in enumerate(self._video_metadata):
            for clip_idx in range(len(valid_clips)):
                self._flat_index.append((folder_idx, clip_idx))


        print(f"[SUMMARY] Total folders: {total_folders}")
        print(f"[SUMMARY] Skipped folders: {skipped_folders}")
        print(f"[SUMMARY] Kept folders: {len(self._video_metadata)}")
        print(f"[SUMMARY] Total valid clips: {total_valid_clips}")
        print(f"[SUMMARY] Label histogram (post-filter): {dict(label_histogram)}")

    def __getitem__(self, index):
        for retry in range(self._num_retries):
            folder_idx, clip_idx = self._flat_index[index]
            folder_path, frame_names, _, valid_indices = self._video_metadata[folder_idx]

            if not valid_indices:
                index = (index + 1) % len(self)
                continue

            start_idx, label = valid_indices[clip_idx]
            frame_list = frame_names[start_idx : start_idx + self._num_frames * self._sample_rate : self._sample_rate]


            tensors = []
            for fname in frame_list:
                path = os.path.join(folder_path, fname)
                if not os.path.isfile(path):
                    tensors = []
                    break
                try:
                    tensor = load_tensor_from_image(path)
                except Exception as e:
                    tensors = []
                    break
                tensors.append(tensor)

            if len(tensors) != self._num_frames:
                continue

            frames = torch.stack(tensors, dim=1)

            frames = data_utils.pack_pathway_output(self.cfg, frames)

            if frames is None or frames[0].numel() == 0:
                print(f"[WARN] Empty frames for index {index}. Trying next.")
                next_index = (index + 1) % len(self._video_metadata)
                return self.__getitem__(next_index)
            if frames[0].shape[-2:] != (224, 224):
                print(f"[WARN] Bad frame shape: {frames[0].shape[-2:]}, skipping")
                next_index = (index + 1) % len(self._video_metadata)
                return self.__getitem__(next_index)
            # print(f"[INFO] Successfully loaded frames for index {index} on retry {retry + 1}.")
            return [frames], label, index, 0, {}

        # If all retries fail
        # print(f"[WARN] No valid frames for index {index} after {self._num_retries} retries. Trying next index.")
        next_index = (index + 1) % len(self._video_metadata)
        return self.__getitem__(next_index)

    # If all retries fail
        # print(f"[WARN] No valid frames for index {index} after {self._num_retries} retries. Trying next index.")
        next_index = (index + 1) % len(self._video_metadata)
        return self.__getitem__(next_index)


    def __len__(self):
        return len(self._flat_index)
