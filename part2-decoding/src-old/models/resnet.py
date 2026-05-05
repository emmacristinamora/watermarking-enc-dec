import torch.nn as nn
from torchvision import models

class ResNet50Classifier(nn.Module):
    def __init__(self, num_outputs=8, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Linear(num_features, num_outputs)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
