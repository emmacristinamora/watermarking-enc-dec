"""8 independent ResNet-50 binary classifiers, one per watermark bit.

Each bit is decoded by its own ResNet-50 backbone with a single-logit head.
This contrasts with the shared-backbone baseline in src/models/resnet50.py.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def _build_resnet50_binary(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = models.resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, 1)
    return backbone


class SeparateBitClassifier(nn.Module):
    """Wraps 8 independent ResNet-50 binary classifiers.

    forward(x) returns a (batch, 8) tensor of probabilities (sigmoid applied).
    Use ``forward_logits`` if you need raw logits (e.g. for BCEWithLogitsLoss).
    """

    NUM_BITS = 8

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.bit_models = nn.ModuleList(
            [_build_resnet50_binary(pretrained=pretrained) for _ in range(self.NUM_BITS)]
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = [m(x) for m in self.bit_models]
        return torch.cat(logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def forward_bit(self, x: torch.Tensor, bit_idx: int) -> torch.Tensor:
        return self.bit_models[bit_idx](x).squeeze(-1)

    def save_all(self, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, m in enumerate(self.bit_models):
            path = os.path.join(output_dir, f"bit_{i}_best.pth")
            torch.save({"bit_index": i, "model_state_dict": m.state_dict()}, path)
            paths.append(path)
        return paths

    def load_all(self, input_dir: str, map_location=None) -> None:
        for i, m in enumerate(self.bit_models):
            path = os.path.join(input_dir, f"bit_{i}_best.pth")
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing per-bit checkpoint: {path}")
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            m.load_state_dict(state)


if __name__ == "__main__":
    model = SeparateBitClassifier(pretrained=False)
    x = torch.randn(2, 3, 1024, 1024)
    probs = model(x)
    logits = model.forward_logits(x)
    print(f"probs shape: {probs.shape}, range [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"logits shape: {logits.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.1f}M (8 ResNet-50 backbones)")
