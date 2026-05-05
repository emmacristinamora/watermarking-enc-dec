import torch.nn as nn
from torchvision import models

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_outputs=8, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
        
        self.backbone = models.efficientnet_b0(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Linear(num_features, num_outputs)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
