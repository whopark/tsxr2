"""Reliability layer for TSXr2 predictions.

Provides confidence assessment, quality validation, and
fallback mechanisms for robust clinical reporting.
"""

from tsxr2.reliability.confidence import ConfidenceAssessment, assess_confidence
from tsxr2.reliability.fallback import generate_fallback_report
from tsxr2.reliability.quality import QualityValidation, validate_image_quality
from tsxr2.reliability.validation import DicomValidationResult, validate_dicom

__all__ = [
    "ConfidenceAssessment",
    "assess_confidence",
    "DicomValidationResult",
    "generate_fallback_report",
    "QualityValidation",
    "validate_dicom",
    "validate_image_quality",
]
