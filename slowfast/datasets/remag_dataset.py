import os
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor


class ActionWithLabelFilteringDataset(Dataset):
    def __init__(self, root_dir, json_anno_path, clip_len=64, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform or Compose([Resize((224, 224)), ToTensor()])
        self.annotations = self._load_annotations(json_anno_path)

        # Each item: (folder, list_of_frame_names)
        self.samples = self._build_samples()

    def _load_annotations(self, json_path):
        # 🔹 Expecting format: { "video_001": { "00001.jpg": "walk", "00002.jpg": "walk", ... }, ... }
        with open(json_path, "r") as f:
            return json.load(f)

    def _build_samples(self):
        valid_samples = []

        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            if folder not in self.annotations:
                continue

            frame_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
            labels = self.annotations[folder]

            for i in range(len(frame_list) - self.clip_len + 1):
                clip_frames = frame_list[i:i + self.clip_len]
                clip_labels = [labels.get(f, None) for f in clip_frames]

                if None in clip_labels:
                    continue

                # 🔹 Check if at least 80% have the same label
                label_counts = {lbl: clip_labels.count(lbl) for lbl in set(clip_labels)}
                dominant_label, max_count = max(label_counts.items(), key=lambda x: x[1])

                if max_count / self.clip_len >= 0.8:
                    valid_samples.append((folder_path, clip_frames, dominant_label))

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, clip_frames, label = self.samples[idx]
        clip = []

        for fname in clip_frames:
            img_path = os.path.join(folder_path, fname)
            img = read_image(img_path).float() / 255.0
            img = self.transform(img)
            clip.append(img)

        clip_tensor = torch.stack(clip, dim=1)  # [C, T, H, W]
        return clip_tensor, label
