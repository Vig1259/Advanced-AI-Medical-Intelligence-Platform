"""
Model architecture for chest X-ray pneumonia classification.

Uses transfer learning on top of a torchvision DenseNet121 backbone
(pretrained on ImageNet), which is a common and well-performing choice
for chest radiograph classification tasks (cf. CheXNet, Rajpurkar et al.).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASS_NAMES)


class ChestXrayClassifier(nn.Module):
    """DenseNet121-based binary classifier for chest X-rays.

    The final classifier layer of torchvision's DenseNet121 is replaced
    with a small head suited to binary classification. `features` block
    is kept intact so Grad-CAM can hook into the last conv layer
    (`features.denseblock4` / `features.norm5`).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        self.features = backbone.features  # conv feature extractor (Grad-CAM target)
        in_features = backbone.classifier.in_features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_maps = self.features(x)          # (B, C, H, W) -- Grad-CAM taps this
        out = torch.relu(feat_maps)
        pooled = self.pool(out).flatten(1)     # (B, C)
        logits = self.classifier(pooled)       # (B, num_classes)
        return logits

    def get_feature_maps_and_logits(self, x: torch.Tensor):
        """Returns (feature_maps, logits). Used directly by Grad-CAM."""
        feat_maps = self.features(x)
        out = torch.relu(feat_maps)
        pooled = self.pool(out).flatten(1)
        logits = self.classifier(pooled)
        return feat_maps, logits


def build_model(pretrained: bool = True) -> ChestXrayClassifier:
    return ChestXrayClassifier(num_classes=NUM_CLASSES, pretrained=pretrained)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and {"model_state_dict": ...} checkpoints
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
