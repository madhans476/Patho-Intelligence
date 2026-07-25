"""Model definition: pretrained backbone + binary classification head."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(backbone_name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """Loads a pretrained backbone and replaces its head for binary classification.

    num_classes=1: we output a single number (a "logit"), not two numbers
    for "tumor"/"no tumor". Combined with BCEWithLogitsLoss (next step),
    this is the standard, numerically stable way to do binary classification
    — one probability directly implies the other (p and 1-p), so a second
    output number would just be redundant.
    """
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=1)
    return model


def freeze_backbone(model: nn.Module, backbone_name: str = "resnet50") -> None:
    """Freezes every parameter except the final classification layer.

    timm names the final layer 'fc' for resnet50 (check model.default_cfg or
    print(model) if you switch backbones later — the head layer name isn't
    always called 'fc').
    """
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreezes every parameter — used for stage 2 fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True