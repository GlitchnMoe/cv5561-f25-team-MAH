"""
Common CNN models for face analysis:
- AgeNet: age regression
- GenderNet: gender classification
- ExprNet: facial expression classification

Backbones: MobileNetV3-Small (default) or ResNet-18 (optional).
Exports: TorchScript
"""
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import resnet18, ResNet18_Weights



class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        return self.pool(x).flatten(1)


def _build_backbone(name: str = "mnetv3_small", pretrained: bool = True, width_mult: float = 1.0) -> nn.Module:
    """
    Returns a convolutional feature extractor without classification head.
    """
    name = name.lower()
    if name == "mnetv3_small":
        assert mobilenet_v3_small is not None, "torchvision is required for MobileNetV3"
        weights = None
        if pretrained and MobileNet_V3_Small_Weights is not None:
            try:
                weights = MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None
        m = mobilenet_v3_small(weights=weights, width_mult=width_mult)
        features = m.features
        return features
    elif name == "resnet18":
        assert resnet18 is not None, "torchvision is required for ResNet18"
        weights = None
        if pretrained and ResNet18_Weights is not None:
            try:
                weights = ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        m = resnet18(weights=weights)
        # keep conv trunk only (remove avgpool+fc)
        features = nn.Sequential(*(list(m.children())[:-2]))
        return features
    else:
        raise ValueError("Unsupported backbone")


def _infer_channels(features: nn.Module, size: int = 224) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, 3, size, size)
        c = features(dummy).shape[1]
    return c


class AgeNet(nn.Module):
    """Age regression network (returns age scalar per sample)."""
    def __init__(self, backbone: str = "mnetv3_small", pretrained: bool = True, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        self.features = _build_backbone(backbone, pretrained=pretrained, width_mult=width_mult)
        c = _infer_channels(self.features)
        self.pool = GlobalAvgPool()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(c, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        v = self.drop(self.pool(f))
        age = self.head(v).squeeze(1)
        return age


class GenderNet(nn.Module):
    """Gender classification network (2 logits)."""
    def __init__(self, backbone: str = "mnetv3_small", pretrained: bool = True, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        self.features = _build_backbone(backbone, pretrained=pretrained, width_mult=width_mult)
        c = _infer_channels(self.features)
        self.pool = GlobalAvgPool()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        v = self.drop(self.pool(f))
        logits = self.fc(v)
        return logits


class ExprNet(nn.Module):
    """Facial expression classifier (K logits)."""
    def __init__(self, num_classes: int = 7, backbone: str = "mnetv3_small", pretrained: bool = True, width_mult: float = 1.0, dropout: float = 0.3):
        super().__init__()
        self.features = _build_backbone(backbone, pretrained=pretrained, width_mult=width_mult)
        c = _infer_channels(self.features)
        self.pool = GlobalAvgPool()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        v = self.drop(self.pool(f))
        logits = self.fc(v)
        return logits
    