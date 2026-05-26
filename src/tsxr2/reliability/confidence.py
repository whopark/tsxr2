"""Confidence assessment for TSXr model predictions.

Provides mechanisms to evaluate prediction confidence and
generate appropriate warnings for clinical review.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConfidenceAssessment:
    """Assessment of model prediction confidence.

    Attributes:
        level: Confidence level (high, medium, low).
        warnings: List of warning messages for clinical review.
        requires_review: Whether findings require additional human review.
    """

    level: Literal["high", "medium", "low"]
    warnings: list[str] = field(default_factory=list)
    requires_review: bool = False


# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.5

# Abnormality score thresholds for borderline detection
BORDERLINE_LOW = 0.4
BORDERLINE_HIGH = 0.6


def assess_confidence(
    confidence_index: float,
    abnormality_score: float,
) -> ConfidenceAssessment:
    """Assess confidence of model predictions.

    Evaluates the confidence index and abnormality score to determine
    overall prediction reliability and generate appropriate warnings.

    Args:
        confidence_index: Model confidence index (0-1).
        abnormality_score: Overall abnormality score (0-1).

    Returns:
        ConfidenceAssessment with level, warnings, and review flag.
    """
    warnings: list[str] = []
    requires_review = False

    # Determine confidence level
    if confidence_index >= HIGH_CONFIDENCE_THRESHOLD:
        level: Literal["high", "medium", "low"] = "high"
    elif confidence_index >= MEDIUM_CONFIDENCE_THRESHOLD:
        level = "medium"
        warnings.append(
            "Moderate confidence - recommend radiologist review for confirmation."
        )
        requires_review = True
    else:
        level = "low"
        warnings.append(
            "Low confidence - results may be unreliable. Exercise caution and "
            "consider repeat imaging or additional clinical correlation."
        )
        requires_review = True

    # Check for borderline abnormality score
    if BORDERLINE_LOW <= abnormality_score <= BORDERLINE_HIGH:
        warnings.append(
            "Borderline abnormality score - findings are equivocal and "
            "require careful clinical interpretation."
        )
        requires_review = True

    return ConfidenceAssessment(
        level=level,
        warnings=warnings,
        requires_review=requires_review,
    )
