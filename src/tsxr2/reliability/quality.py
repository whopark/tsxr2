"""Image quality validation for reliability assessment.

Validates image quality metrics and flags issues that may
affect model prediction accuracy.
"""

from dataclasses import dataclass, field

from tsxr2.schemas.tsxr_output import QualityChecks


@dataclass
class QualityValidation:
    """Result of image quality validation.

    Attributes:
        is_acceptable: Whether image quality meets minimum standards.
        issues: List of quality issues detected.
    """

    is_acceptable: bool
    issues: list[str] = field(default_factory=list)


def validate_image_quality(quality_checks: QualityChecks) -> QualityValidation:
    """Validate image quality for reliable analysis.

    Checks rotation and inspiration quality to determine if the
    image meets minimum standards for accurate model predictions.

    Args:
        quality_checks: QualityChecks from TSXr output.

    Returns:
        QualityValidation with acceptability and issue list.
    """
    issues: list[str] = []

    # Check rotation - high or moderate rotation affects accuracy
    if quality_checks.rotation in ("high", "moderate"):
        issues.append(
            f"Image has {quality_checks.rotation} rotation - may affect "
            "cardiac and mediastinal measurements."
        )

    # Check inspiration quality
    if quality_checks.inspiration == "inadequate":
        issues.append(
            "Inadequate inspiration - lung bases may be obscured, "
            "limiting evaluation of lower lung zones."
        )
    elif quality_checks.inspiration == "hyperinflated":
        issues.append(
            "Hyperinflated lungs - may indicate air trapping or "
            "technical factor; correlate clinically."
        )

    is_acceptable = len(issues) == 0

    return QualityValidation(
        is_acceptable=is_acceptable,
        issues=issues,
    )
