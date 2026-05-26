"""TSXr model loading utilities.

Loads DenseNet121-based model for chest X-ray classification.
Supports GPU (CUDA) and CPU inference with automatic device detection.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Common chest X-ray findings for multi-label classification
TSXR_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

MODEL_VERSION = "tsxr-v2.1"


class TSXrModel(nn.Module):
    """DenseNet121-based model for chest X-ray classification.

    Multi-label classifier for 14 common chest X-ray findings.
    Architecture based on CheXNet (Rajpurkar et al., 2017).
    """

    def __init__(self, num_classes: int = 14, pretrained_backbone: bool = True):
        """Initialize TSXr model.

        Args:
            num_classes: Number of output classes (default: 14 findings).
            pretrained_backbone: Use ImageNet pretrained weights for backbone.
        """
        super().__init__()

        # Load DenseNet121 backbone
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.backbone = models.densenet121(weights=weights)

        # Replace classifier for multi-label output
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Sigmoid(),  # Multi-label: independent probabilities per class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224) or (batch, 3, 512, 512).

        Returns:
            Tensor of shape (batch, num_classes) with probabilities.
        """
        return self.backbone(x)

    @property
    def features(self) -> nn.Module:
        """Access backbone features for Grad-CAM visualization."""
        return self.backbone.features


def get_device() -> torch.device:
    """Detect and return the best available device.

    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tsxr_model(
    weights_path: Path | str | None = None,
    device: torch.device | None = None,
) -> tuple[TSXrModel, torch.device]:
    """Load TSXr model for inference.

    Args:
        weights_path: Optional path to custom trained weights.
            If None, uses ImageNet pretrained backbone.
        device: Optional device override. If None, auto-detects.

    Returns:
        Tuple of (model, device) ready for inference.
    """
    if device is None:
        device = get_device()

    model = TSXrModel(num_classes=len(TSXR_LABELS), pretrained_backbone=True)

    # Load custom weights if provided
    if weights_path is not None:
        weights_path = Path(weights_path)
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    return model, device
