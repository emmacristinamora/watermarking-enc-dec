from .resnet import ResNet50Classifier
from .efficientnet import EfficientNetB0Classifier

def get_model(architecture, num_outputs=8, pretrained=True):
    if architecture == 'resnet50':
        return ResNet50Classifier(num_outputs=num_outputs, pretrained=pretrained)
    elif architecture == 'efficientnet_b0':
        return EfficientNetB0Classifier(num_outputs=num_outputs, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

