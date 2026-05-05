# part2-decoding/src/utils.py

"""
Shared utilities for all decoder training scripts.

Provides: WatermarkDataset, build_model, make_transform, compute_metrics.
Imported by 01_train_id_decoder.py, 02_train_bit_decoder.py, 03_train_vit_decoder.py.
"""


# === IMPORTS ===

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms


# === DATASET ===

class WatermarkDataset(Dataset):
    """
    Loads watermarked images and their bit labels from metadata + splits.

    Args:
        metadata_path: path to metadata.json (JSON array of 2560 records)
        images_dir: parent directory containing train/, val/, test/ subdirs
        splits: dict loaded from splits.json; used to resolve each index to its subdir
        transform: torchvision transform applied to each image

    Returns per item:
        image  (C, H, W) float tensor
        bits   (8,) float tensor with values in {0, 1}
        id_int int watermark ID
    """

    def __init__(
        self,
        metadata_path: Path,
        images_dir: Path,
        splits: dict[str, list[int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        with open(metadata_path) as f:
            self.metadata: list[dict[str, Any]] = json.load(f)
        self.images_dir = images_dir
        self.transform = transform
        # map global index → split subdir name
        self._idx_to_split: dict[int, str] = {}
        for split_name, indices in splits.items():
            for idx in indices:
                self._idx_to_split[idx] = split_name

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.metadata[idx]
        split = self._idx_to_split[idx]
        img_path = self.images_dir / split / entry["file"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        bits = torch.tensor(entry["bits"], dtype=torch.float32)
        return {"image": image, "bits": bits, "id_int": entry["id_int"]}


# === MODEL FACTORY ===

def build_model(architecture: str, num_outputs: int, pretrained: bool) -> nn.Module:
    """
    Return a classification head on top of a pretrained backbone.

    Args:
        architecture: "resnet50" or "efficientnet_b0"
        num_outputs: number of binary outputs (8 for bit decoding)
        pretrained: whether to load ImageNet weights

    The backbone's original head is replaced with nn.Identity(); a single
    Linear layer is appended as the classifier.
    """
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"unknown architecture: {architecture!r}. Choose resnet50 or efficientnet_b0.")

    class _Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(in_features, num_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x))

    return _Decoder()


# === TRANSFORMS ===

def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# === METRICS ===

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """
    Args:
        preds:   (N, 8) binary float tensor
        targets: (N, 8) binary float tensor

    Returns dict with per_bit_accuracy (list), mean_bit_accuracy, exact_match_rate.
    """
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    per_bit = [(p[:, i] == t[:, i]).mean() for i in range(p.shape[1])]
    exact_match = (p == t).all(axis=1).mean()
    return {
        "per_bit_accuracy": [float(x) for x in per_bit],
        "mean_bit_accuracy": float(np.mean(per_bit)),
        "exact_match_rate": float(exact_match),
    }
