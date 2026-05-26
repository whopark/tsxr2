"""TSXr model output schema matching PRD specification."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TSXrMetadata(BaseModel):
    """Metadata about the TSXr model run."""

    model_version: str = Field(..., description="TSXr model version identifier")
    timestamp: datetime = Field(..., description="Timestamp of analysis")


class ImageInfo(BaseModel):
    """Information about the analyzed image."""

    dimensions: tuple[int, int] = Field(..., description="Image dimensions [width, height]")
    view: Literal["PA", "AP", "lateral"] = Field(..., description="X-ray view type")


class Finding(BaseModel):
    """Individual lesion finding from TSXr analysis."""

    label: str = Field(..., description="Classification label (e.g., Pneumonia, Nodule)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Classification probability")
    severity: Literal["mild", "moderate", "severe"] = Field(..., description="Severity level")
    side: Literal["left", "right", "bilateral", "central"] = Field(
        ..., description="Anatomical location"
    )
    bbox: tuple[int, int, int, int] | None = Field(
        default=None, description="Bounding box [x1, y1, x2, y2]"
    )


class GlobalScores(BaseModel):
    """Global analysis scores."""

    abnormality_score: float = Field(..., ge=0.0, le=1.0, description="Overall abnormality score")
    confidence_index: float = Field(..., ge=0.0, le=1.0, description="Model confidence index")


class QualityChecks(BaseModel):
    """Image quality assessment."""

    rotation: Literal["none", "low", "moderate", "high"] = Field(
        ..., description="Rotation quality"
    )
    inspiration: Literal["inadequate", "adequate", "hyperinflated"] = Field(
        ..., description="Inspiration quality"
    )


class TSXrOutput(BaseModel):
    """Complete TSXr model output schema per PRD specification."""

    metadata: TSXrMetadata
    image_info: ImageInfo
    findings: list[Finding] = Field(default_factory=list)
    global_scores: GlobalScores
    quality_checks: QualityChecks
