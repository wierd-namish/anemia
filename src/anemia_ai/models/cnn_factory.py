"""
Convolutional Neural Network Backbone Factory.

Supports:
- EfficientNet-B0
- DenseNet121
- MobileNetV3-Large
- ResNet50
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def create_cnn_backbone(
    architecture: str = "efficientnet_b0",
    pretrained: bool = False,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Instantiates a deep CNN backbone configured for binary classification.

    Args:
        architecture: Architecture name ('efficientnet_b0', 'densenet121', 'mobilenet_v3_large', 'resnet50').
        pretrained: Whether to initialize with ImageNet default weights.
        dropout_rate: Dropout probability in the classification head.

    Returns:
        nn.Module outputting single logit.
    """
    arch = architecture.lower()

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )
        return backbone

    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )
        return backbone

    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, 1)
        return backbone

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )
        return backbone

    else:
        raise ValueError(f"Unsupported CNN architecture: {architecture}")
