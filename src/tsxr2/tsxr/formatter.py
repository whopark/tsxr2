"""TSXr output formatting to match PRD schema.

Converts raw inference results to the structured TSXrOutput
schema required by the API and Gemini reasoning layer.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from tsxr2.schemas import TSXrOutput
from tsxr2.schemas.tsxr_output import (
    Finding,
    GlobalScores,
    ImageInfo,
    QualityChecks,
    TSXrMetadata,
)
from tsxr2.tsxr.model_loader import MODEL_VERSION

# Severity thresholds for findings
SEVERITY_THRESHOLDS = {
    "severe": 0.8,
    "moderate": 0.5,
    "mild": 0.0,
}


def determine_severity(probability: float) -> Literal["mild", "moderate", "severe"]:
    """Determine severity level based on probability.

    Args:
        probability: Finding probability (0-1).

    Returns:
        Severity level string.
    """
    if probability >= SEVERITY_THRESHOLDS["severe"]:
        return "severe"
    elif probability >= SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "mild"


def determine_side(label: str) -> Literal["left", "right", "bilateral", "central"]:
    """Determine anatomical side based on finding label.

    For now, returns "bilateral" as we don't have localization.
    In a full implementation, this would use Grad-CAM heatmaps.

    Args:
        label: Finding label.

    Returns:
        Anatomical side string.
    """
    # Cardiomegaly and Hernia are typically central
    if label in ["Cardiomegaly", "Hernia"]:
        return "central"
    # Default to bilateral without localization data
    return "bilateral"


def format_tsxr_output(
    inference_result: dict[str, Any],
    image_dimensions: tuple[int, int],
    view_position: str = "PA",
    quality_rotation: Literal["none", "low", "moderate", "high"] = "low",
    quality_inspiration: Literal["inadequate", "adequate", "hyperinflated"] = "adequate",
) -> TSXrOutput:
    """Format inference results to TSXrOutput schema.

    Converts the raw inference dict to a Pydantic model matching
    the PRD specification for TSXr output.

    Args:
        inference_result: Dict from run_inference() containing:
            - probabilities: List of 14 class probabilities
            - labels: List of 14 class labels
            - findings: List of findings above threshold
            - abnormality_score: Overall abnormality score
            - confidence_index: Model confidence index
        image_dimensions: Image dimensions (height, width).
        view_position: X-ray view type ("PA", "AP", "lateral").
        quality_rotation: Image rotation quality assessment.
        quality_inspiration: Inspiration quality assessment.

    Returns:
        TSXrOutput schema instance.
    """
    # Build metadata
    metadata = TSXrMetadata(
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc),
    )

    # Build image info
    # Validate view_position
    valid_views = ["PA", "AP", "lateral"]
    if view_position not in valid_views:
        view_position = "PA"

    image_info = ImageInfo(
        dimensions=image_dimensions,
        view=view_position,  # type: ignore
    )

    # Build findings list
    findings = []
    for finding_dict in inference_result.get("findings", []):
        label = finding_dict["label"]
        prob = finding_dict["probability"]

        finding = Finding(
            label=label,
            probability=prob,
            severity=determine_severity(prob),
            side=determine_side(label),
            bbox=None,  # No localization without Grad-CAM
        )
        findings.append(finding)

    # Build global scores
    global_scores = GlobalScores(
        abnormality_score=inference_result.get("abnormality_score", 0.0),
        confidence_index=inference_result.get("confidence_index", 0.0),
    )

    # Build quality checks
    quality_checks = QualityChecks(
        rotation=quality_rotation,
        inspiration=quality_inspiration,
    )

    return TSXrOutput(
        metadata=metadata,
        image_info=image_info,
        findings=findings,
        global_scores=global_scores,
        quality_checks=quality_checks,
    )
