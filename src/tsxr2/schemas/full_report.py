"""Full report schema combining all analysis outputs.

Provides a comprehensive response model that includes TSXr analysis,
Gemini report, and reliability assessment in a single structure.
"""

from typing import Any

from pydantic import BaseModel, Field

from tsxr2.schemas.gemini_report import GeminiReport
from tsxr2.schemas.tsxr_output import TSXrOutput


class ConfidenceInfo(BaseModel):
    """Confidence assessment information."""

    level: str = Field(..., description="Confidence level (high, medium, low)")
    warnings: list[str] = Field(default_factory=list)
    requires_review: bool = False


class QualityInfo(BaseModel):
    """Image quality validation information."""

    is_acceptable: bool = True
    issues: list[str] = Field(default_factory=list)


class ValidationInfo(BaseModel):
    """DICOM validation information."""

    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReliabilityInfo(BaseModel):
    """Combined reliability assessment."""

    confidence: ConfidenceInfo
    quality: QualityInfo
    used_fallback: bool = Field(
        default=False,
        description="Whether fallback report generation was used due to Gemini failure",
    )


class FullReportResponse(BaseModel):
    """Comprehensive analysis response.

    Combines TSXr model output, Gemini clinical report, and
    reliability assessment into a single response structure.
    """

    tsxr_output: dict[str, Any] = Field(
        ..., description="TSXr model analysis results"
    )
    gemini_report: dict[str, Any] = Field(
        ..., description="Gemini-generated clinical report"
    )
    reliability: ReliabilityInfo = Field(
        ..., description="Reliability and confidence assessment"
    )
    validation: ValidationInfo = Field(
        ..., description="Input validation results"
    )
