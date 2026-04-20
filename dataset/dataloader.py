"""
DataLoader for Task-Aware Semantic Grasp Quality Prediction.
Handles synthetic dataset with procedural image generation (no real images needed).
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter


# ─────────────────────────────────────────────────────────────────────────────
# Procedural Image Generator
# Generates synthetic object images so the pipeline works without real photos
# ─────────────────────────────────────────────────────────────────────────────

OBJECT_COLORS = {
    "mug":      {"body": (180, 100,  60), "handle": (140,  70,  40)},
    "knife":    {"blade": (200, 200, 210), "handle": (120,  80,  50)},
    "hammer":   {"head": (160, 160, 160),  "handle": (100,  60,  30)},
    "glass":    {"body": ( 70, 130, 200),  "handle": (200, 200, 200)},  # ← add
    "pan":      {"body": (180, 180, 180),  "handle": ( 80,  60,  40)},  # ← add
}


def generate_synthetic_image(obj_name: str, regions: dict,
                              img_size: int = 224,
                              augment: bool = False) -> Image.Image:
    """
    Creates a plausible procedural RGB image for an object
    by painting each region in a distinct color with noise.
    """
    img = Image.new("RGB", (img_size, img_size), (240, 235, 220))
    draw = ImageDraw.Draw(img)

    colors = OBJECT_COLORS.get(obj_name, {})

    for region_name, bbox in regions.items():
        x1, y1, x2, y2 = bbox
        base_color = colors.get(region_name, (150, 150, 150))
        if augment:
            # Color jitter
            noise = [random.randint(-20, 20) for _ in range(3)]
            color = tuple(int(np.clip(c + n, 0, 255))
                          for c, n in zip(base_color, noise))
        else:
            color = base_color
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(50, 50, 50), width=2)

    if augment:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.8)))

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GraspDataset(Dataset):
    """
    Loads grasp records from JSON.
    Returns: image tensor, grasp_point tensor, task_id, label, stability_score
    """

    def __init__(self, json_path: str, objects_cfg: dict,
                 img_size: int = 224, split: str = "train"):

        self.img_size    = img_size
        self.split       = split
        self.objects_cfg = objects_cfg

        with open(json_path) as f:
            self.records = json.load(f)

        self.is_train = (split == "train")

        # ImageNet normalization — backbone is pretrained on ImageNet
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )

        self.to_tensor = T.ToTensor()

        # Train augmentations (spatial only — no color jitter, preserves region semantics)
        self.spatial_aug = T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(degrees=5),
        ]) if self.is_train else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        obj_name  = rec["object_name"]
        grasp_x   = rec["grasp_x"]
        grasp_y   = rec["grasp_y"]
        task_id   = rec["task_id"]
        label     = rec["label"]
        stab      = rec["stability_score"]

        # ── Image ──
        regions = self.objects_cfg[obj_name]["regions"]
        img = generate_synthetic_image(
            obj_name, regions, self.img_size,
            augment=self.is_train
        )
        img_tensor = self.to_tensor(img)            # [3, 224, 224], [0,1]
        img_tensor = self.normalize(img_tensor)     # ImageNet normalized

        # ── Grasp point encoding ──
        # Normalize (x,y) to [-1, 1] — more stable than [0,1] for linear layers
        gx = (grasp_x / self.img_size) * 2.0 - 1.0
        gy = (grasp_y / self.img_size) * 2.0 - 1.0
        grasp_point = torch.tensor([gx, gy], dtype=torch.float32)

        return {
            "image":       img_tensor,
            "grasp_point": grasp_point,
            "task_id":     torch.tensor(task_id, dtype=torch.long),
            "label":       torch.tensor(label,   dtype=torch.float32),
            "stability":   torch.tensor(stab,    dtype=torch.float32),
            # For logging
            "object_name": obj_name,
            "region":      rec["region"],
            "task_name":   rec["task_name"],
        }


def build_dataloaders(cfg: dict, data_dir: str = "data/") -> dict:
    """Returns train/val/test DataLoaders."""

    dataset_cfg = cfg["dataset"]
    train_cfg   = cfg["training"]
    objects_cfg = dataset_cfg["objects"]
    img_size    = dataset_cfg["image_size"]

    loaders = {}
    for split in ["train", "val", "test"]:
        json_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"{json_path} not found. Run dataset/generate_dataset.py first."
            )

        ds = GraspDataset(json_path, objects_cfg, img_size, split)

        loaders[split] = DataLoader(
            ds,
            batch_size  = train_cfg["batch_size"],
            shuffle     = (split == "train"),
            num_workers = train_cfg["num_workers"],
            pin_memory  = train_cfg.get("pin_memory", True),
            drop_last   = (split == "train"),
        )

    print(f"DataLoaders built:")
    for k, v in loaders.items():
        print(f"  {k}: {len(v.dataset)} samples, {len(v)} batches")

    return loaders
