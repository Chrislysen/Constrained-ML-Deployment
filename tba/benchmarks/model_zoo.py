"""Model zoo: pre-trained ImageNet models for deployment benchmarking.

All models come from torchvision — no training needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


MODEL_CONFIGS = {
    "resnet18": {
        "factory": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "input_size": 224,
    },
    "resnet50": {
        "factory": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "input_size": 224,
    },
    "mobilenet_v2": {
        "factory": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2),
        "input_size": 224,
    },
    "efficientnet_b0": {
        "factory": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "input_size": 224,
    },
    "vit_tiny": {
        "factory": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
        "input_size": 224,
    },
}


def load_model(model_name: str) -> nn.Module:
    """Load a pre-trained model by name. Returns model in eval mode."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    model = MODEL_CONFIGS[model_name]["factory"]()
    model.eval()
    return model


def get_input_size(model_name: str) -> int:
    return MODEL_CONFIGS[model_name]["input_size"]


def list_models() -> list[str]:
    return list(MODEL_CONFIGS.keys())
