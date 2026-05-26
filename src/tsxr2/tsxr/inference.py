"""TSXr inference pipeline.

Handles image preprocessing, model inference, and result extraction
for chest X-ray classification.
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import transforms

from tsxr2.tsxr.model_loader import TSXR_LABELS, TSXrModel

# ImageNet normalization (used by pretrained DenseNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_for_model(
    image: NDArray[np.uint8],
    target_size: int = 224,
) -> torch.Tensor:
    """Preprocess a normalized image for model input.

    Applies:
    1. Resize to model input size (224x224 for DenseNet)
    2. Convert to tensor and normalize channels
    3. Apply ImageNet normalization

    Args:
        image: Normalized image array (H, W, 3) uint8.
        target_size: Model input size (default 224 for DenseNet).

    Returns:
        Tensor of shape (1, 3, target_size, target_size) ready for model.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(image)
    # Add batch dimension
    return tensor.unsqueeze(0)


def run_inference(
    model: TSXrModel,
    image: NDArray[np.uint8],
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run inference on a preprocessed image.

    Args:
        model: Loaded TSXr model.
        image: Normalized image array (H, W, 3) uint8 from preprocessing.
        device: Device to run inference on.
        threshold: Probability threshold for positive findings (default 0.5).

    Returns:
        Dict containing:
        - probabilities: List of 14 class probabilities
        - labels: List of 14 class labels
        - findings: List of findings above threshold
        - abnormality_score: Overall abnormality score (max probability)
        - confidence_index: Average probability of detected findings
    """
    # Preprocess image for model
    tensor = preprocess_for_model(image).to(device)

    # Run inference
    with torch.no_grad():
        output = model(tensor)

    # Extract probabilities
    probs = output.squeeze().cpu().numpy().tolist()

    # Find findings above threshold
    findings = []
    for label, prob in zip(TSXR_LABELS, probs):
        if prob >= threshold:
            findings.append({
                "label": label,
                "probability": prob,
            })

    # Sort findings by probability (highest first)
    findings.sort(key=lambda x: x["probability"], reverse=True)

    # Calculate global scores
    abnormality_score = max(probs) if probs else 0.0
    confidence_index = (
        sum(f["probability"] for f in findings) / len(findings)
        if findings
        else 1.0 - abnormality_score  # High confidence in "normal" if no findings
    )

    return {
        "probabilities": probs,
        "labels": TSXR_LABELS,
        "findings": findings,
        "abnormality_score": abnormality_score,
        "confidence_index": confidence_index,
    }
