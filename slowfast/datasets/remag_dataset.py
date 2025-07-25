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
from slowfast.datasets.utils import pack_pathway_output
import logging
import cv2
import einops
import torchvision.transforms.functional as F
from torchvision import transforms
import zipfile

logging.getLogger("PIL").setLevel(logging.WARNING)

from PIL import Image

def load_tensor_from_image(path):
    # try:
    img = Image.open(path).convert("RGB")
    img = img.resize((224,224))
    arr = np.array(img)
    
    # Fix any case where .convert("RGB") didn't result in 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)  
        
    # tensor = einops.rearrange(arr, 'h w c -> c h w').float() / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    # assert the shape is ((224, 224, 3))
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"[ERROR] Loaded tensor has unexpected shape: {tensor.shape}")
    #     print(1/0)
    return tensor

def create_balanced_indices(dataset, cfg):
    label_to_indices = {i: [] for i in range(cfg.MODEL.NUM_CLASSES)}
    for idx, (folder_idx, clip_idx) in enumerate(dataset._flat_index):
        _, _, _, valid_clips = dataset._video_metadata[folder_idx]
        _, label = valid_clips[clip_idx]
        label_to_indices[label].append((folder_idx, clip_idx))

    balance_num = max(len(v) for v in label_to_indices.values())
    balanced = []
    for label in label_to_indices:
        balanced.extend(random.choices(label_to_indices[label], k=balance_num))

    random.shuffle(balanced)
    return balanced

@DATASET_REGISTRY.register()
class REMAGDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self._num_frames = cfg.DATA.NUM_FRAMES
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        if mode == 'train' and cfg.DATA.SYN_GEN:
            self._frame_dir = cfg.DATA.PATH_TO_DATA_DIR_SYN
        else:
            self._frame_dir = cfg.DATA.PATH_TO_DATA_DIR
        if cfg.DATA.REMAG_TYPE == 'ucf':
            if mode == 'train' and not cfg.DATA.SYN_GEN:
                self._frame_dir += '/train'
            if mode in ['val', 'test']:
                self._frame_dir += '/test'
                
        if cfg.DATA.PATH_TO_DATA_DIR_VAL != "" and self.mode == "val":
            self._frame_dir = cfg.DATA.PATH_TO_DATA_DIR_VAL
        self._num_retries = 100
        self._min_agreement_ratio = 0.8
        self._enable_test_crops = cfg.TEST.NUM_SPATIAL_CROPS > 1
        self._train_ids = cfg.DATA.TRAIN_IDS #getattr(cfg.DATA, "TRAIN_IDS", [1, 3, 10, 15, 19, 22, 2, 7, 11, 13])
        self._test_ids = cfg.DATA.TEST_IDS #getattr(cfg.DATA, "TEST_IDS", [5, 6, 8, 21, 20])
        self._enable_multigrid = getattr(cfg.MULTIGRID, "ENABLE", False)
        self._construct_loader()

    def _construct_loader(self):
        self._video_metadata = []
        total_folders = 0
        skipped_folders = 0
        total_valid_clips = 0
        label_histogram = Counter()
        
        self.real_mean_std = []            

        folders = sorted(os.listdir(self._frame_dir))
        

        for folder in folders:
            total_folders += 1
            folder_path = os.path.join(self._frame_dir, folder)
            frames = os.listdir(folder_path)
            frames = [frame for frame in frames if '.jpg' in frame or '.png' in frame]
            if len(frames) == 0:
                skipped_folders += 1
                continue
            sample_img1 = Image.open(folder_path + '/' + frames[0]).convert("RGB")
            sample_img2 = Image.open(folder_path + '/' + frames[10]).convert("RGB")
            arr = np.abs(np.array(sample_img1) - np.array(sample_img2))
            if np.sum(arr) < 20:
                skipped_folders += 1
                continue
            
            if frames[0][-3:] == 'jpg':
                mean_rgb = []
                std_rgb = []
                for c in range(3):  # R, G, B
                    img_np = np.array(sample_img1) / 255.0
                    real_mean, real_std = img_np[:, :, c].mean(), img_np[:, :, c].std()
                    mean_rgb.append(real_mean)
                    std_rgb.append(real_std)
                self.real_mean_std.append([mean_rgb, std_rgb])
            
            if self.cfg.DATA.REMAG_TYPE == '':
                match = re.search(r"Subj\+0*([0-9]+)", folder)
                if not match:
                    continue  
                if folder[7] == 'B' and self.mode in ['train', 'val']:
                    continue
            
                subject_id = int(match.group(1))
                is_test = subject_id in self._test_ids
                is_train = subject_id in self._train_ids
            
                if self.mode == "train" and is_test:
                    continue
                if self.mode in ["test", "val"] and is_train:
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
            if any(label > 11 for label in activity_ids):
                # skipped_folders += 1
                continue

            # --- Relabel labels in [11,12,13,14,15] by subtracting 5 ---
            # if self.mode == 'test':
            #     activity_ids = [label - 1 if label in {12, 13, 14, 15, 16} else label for label in activity_ids]

            valid_indices = []
            for i in range(0, len(frame_names) - self._num_frames + 1, self._num_frames):
                window = activity_ids[i:i + self._num_frames]
                if window.count(-1) > 0:
                    continue
                label_counts = Counter(window)
                most_common_label, count = label_counts.most_common(1)[0]

                label_histogram[most_common_label] += 1

                if count / self._num_frames >= self._min_agreement_ratio:
                    valid_indices.append((i, most_common_label))

            if valid_indices:
                self._video_metadata.append((folder_path, frame_names, activity_ids, valid_indices))
                total_valid_clips += len(valid_indices)
        self._flat_index = []
        all_labels = {i:[] for i in range(self.cfg.MODEL.NUM_CLASSES)}
        for folder_idx, (_, _, _, valid_clips) in enumerate(self._video_metadata):
            for clip_idx, (_, label) in enumerate(valid_clips):
                self._flat_index.append((folder_idx, clip_idx))
                all_labels[label].append((folder_idx, clip_idx))
        balanced_count = {}
        if self.mode in ['train', 'val']:
            self._flat_index = []
            balance_num = max([len(all_labels[label]) for label in all_labels.keys()])
            # print([len(all_labels[label]) for label in all_labels.keys()])
            # print(len(all_labels[label]))
            for label in all_labels.keys():
                # print(all_labels[label])
                # print(all_labels[label], balance_num)
                new_labels = random.choices(all_labels[label], k=balance_num)
                all_labels[label] = new_labels
                balanced_count[label] = len(new_labels)
                self._flat_index.extend(new_labels)
        else:
            balanced_count = dict(label_histogram)
        self.folder_indices = list(set([i[0] for i in self._flat_index]))
        self.num_videos = self.__len__()
        print(f"[MODE] {self.mode}")
        print(f"[SUMMARY] Total folders: {total_folders}")
        print(f"[SUMMARY] Skipped folders: {skipped_folders}")
        print(f"[SUMMARY] Kept folders: {len(self._video_metadata)}")
        print(f"[SUMMARY] Total valid clips: {total_valid_clips}")
        print(f"[SUMMARY] Label histogram (post-filter): {dict(label_histogram)}")
        print(f"[SUMMARY] Balanced: {balanced_count}")

    def __getitem__(self, index):
        for retry in range(self._num_retries):
            if self.mode == 'val': #(self.mode == 'train' and not self.cfg.DATA.SYN_GEN) or
                # print(index)
                # while len(clip_idxs) == 0:
                #     # folder_idx = index
                folder_idx = self.folder_indices[index]
                clip_idxs = [e for e in self._flat_index if e[0] == folder_idx]
                clip_idx = random.choice(clip_idxs)[1]
            else:
                folder_idx, clip_idx = self._flat_index[index]
            
            factor_mean, factor_std = self.real_mean_std[index % len(self.real_mean_std)]
            
            folder_path, frame_names, _, valid_indices = self._video_metadata[folder_idx]

            if not valid_indices:
                index = (index + 1) % len(self)
                # print('line', 213)
                continue
                # return None

            start_idx, label = valid_indices[clip_idx]
            frame_list = frame_names[start_idx : start_idx + self._num_frames * self._sample_rate : self._sample_rate]
            
            # brightness_factor_increase = torch.rand(1)
            tensors = []
            for fname in frame_list:
                # if self.cfg.DATA.SYN_GEN and self.mode == 'train':
                if not os.path.exists(os.path.join(folder_path, fname)):
                    fname = fname[1:-3] + 'png'
                path = os.path.join(folder_path, fname)
                if not os.path.isfile(path):
                    tensors = []
                    break
                try:
                    tensor = load_tensor_from_image(path)
                except Exception as e:
                    tensors = []
                    break
                if 'png' in fname and self.mode == 'train' and self.cfg.SOLVER.MAX_EPOCH >= 15:
                    for c in range(3):
                        fake_mean, fake_std = torch.mean(tensor[c]), torch.std(tensor[c])
                        tensor[c] = ((tensor[c] - fake_mean) / fake_std) * factor_std[c] + factor_mean[c]
                    tensor = F.adjust_brightness(tensor, 0.95)
                    gb = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 2.0))
                    tensor = gb(tensor)
                tensors.append(tensor)

            if len(tensors) != self._num_frames:
                index = (index + 1) % len(self)
                continue

            frames = torch.stack(tensors, dim=1)
            # frames = np.stack(tensors, axis=1)            

            # frames = resize_and_random_crop_single_video(frames)

            if frames is None or frames[0].size == 0:
                print(f"[WARN] Empty frames for index {index}. Trying next.")
                next_index = (index + 1) % len(self)
                return self.__getitem__(next_index)
            if frames[0].shape[-2:] != (224, 224):
                print(f"[WARN] Bad frame shape: {frames[0].shape[-2:]}, skipping")
                next_index = (index + 1) % len(self)
                return self.__getitem__(next_index)
            # print(f"[INFO] Successfully loaded frames for index {index} on retry {retry + 1}.")
            frames = pack_pathway_output(self.cfg, frames)
            if self.mode in ['train', 'val']:
                frames = [frames]
            return frames, label, index, 0, {}

        # If all retries fail
        # print(f"[WARN] No valid frames for index {index} after {self._num_retries} retries. Trying next index.")
        next_index = (index + 1) % len(self)
        return self.__getitem__(next_index)

        # # If all retries fail
        # # print(f"[WARN] No valid frames for index {index} after {self._num_retries} retries. Trying next index.")
        # next_index = (index + 1) % len(self._video_metadata)
        # return self.__getitem__(next_index)


    def __len__(self):
        if self.mode == 'val': #(self.mode == 'train' and not self.cfg.DATA.SYN_GEN) or
            return len(self.folder_indices)
        else:
            return len(self._flat_index)


# def resize_and_random_crop_single_video(video, resize_shape=(256, 256), crop_shape=(224, 224)):
#     """
#     Resize and randomly crop a single video (NumPy array) with shape (F, C, H, W).

#     Args:
#         video: np.ndarray of shape (F, C, H, W)
#         resize_shape: tuple (new_H, new_W) for resizing
#         crop_shape: tuple (crop_H, crop_W) for cropping

#     Returns:
#         Cropped video of shape (F, C, crop_H, crop_W)
#     """
#     F, C, H, W = video.shape
#     new_H, new_W = resize_shape
#     crop_H, crop_W = crop_shape

#     resized = np.zeros((F, C, new_H, new_W), dtype=video.dtype)
#     for f in range(F):
#         for c in range(C):
#             resized[f, c] = cv2.resize(video[f, c], (new_W, new_H), interpolation=cv2.INTER_LINEAR)

#     top = np.random.randint(0, new_H - crop_H + 1)
#     left = np.random.randint(0, new_W - crop_W + 1)

#     cropped = resized[:, :, top:top + crop_H, left:left + crop_W]
#     return cropped
