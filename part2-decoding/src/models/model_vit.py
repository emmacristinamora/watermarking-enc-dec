"""ViT-B/16 watermark decoder with 8 independent binary heads.

Same forward-pass interface as ``src/models/resnet50.ResNet50Classifier``: returns
raw logits of shape (batch, 8). Apply sigmoid externally for probabilities, or
pass logits directly to ``BCEWithLogitsLoss``.
"""
import torch
import torch.nn as nn
from torchvision import models


class ViTWatermarkDecoder(nn.Module):
    NUM_BITS = 8
    INPUT_SIZE = 224

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None
        self.backbone = models.vit_b_16(weights=weights)

        hidden_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

        self.bit_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.NUM_BITS)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = [head(features) for head in self.bit_heads]
        return torch.cat(logits, dim=1)


if __name__ == "__main__":
    model = ViTWatermarkDecoder(pretrained=False)
    x = torch.randn(2, 3, ViTWatermarkDecoder.INPUT_SIZE, ViTWatermarkDecoder.INPUT_SIZE)
    out = model(x)
    print(f"output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.1f}M")
