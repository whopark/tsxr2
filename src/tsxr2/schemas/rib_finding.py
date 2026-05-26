"""Rib fracture detection schemas for systematic rib analysis."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CoordinatePoint(BaseModel):
    """2D coordinate point for annotations."""

    x: int = Field(..., description="X coordinate in pixels")
    y: int = Field(..., description="Y coordinate in pixels")


class ArrowAnnotation(BaseModel):
    """Arrow annotation for fracture visualization."""

    target_point: CoordinatePoint = Field(
        ..., description="Point the arrow points to (fracture location)"
    )
    origin_point: CoordinatePoint = Field(..., description="Arrow start point")
    label: str = Field(..., description="Annotation label (e.g., 'L5 Fracture')")
    color: tuple[int, int, int] = Field(
        default=(255, 0, 0), description="RGB color for the arrow"
    )
    associated_rib: str = Field(..., description="Reference to rib_id (e.g., L5, R3)")


class RibFinding(BaseModel):
    """Individual rib analysis result with fracture detection."""

    rib_id: str = Field(
        ...,
        description="Rib identifier (L1-L12 or R1-R12)",
        pattern=r"^[LR](1[0-2]?|[1-9])$",
    )
    bbox: tuple[int, int, int, int] = Field(
        ..., description="Bounding box [x1, y1, x2, y2] in pixels"
    )
    centroid: CoordinatePoint = Field(..., description="Center point of the rib region")
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of rib detection"
    )
    fracture_status: Literal["intact", "fractured", "suspicious"] = Field(
        ..., description="Fracture classification result"
    )
    fracture_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of fracture classification"
    )
    fracture_type: str | None = Field(
        default=None,
        description="Fracture type if detected (e.g., 'displaced', 'non-displaced', 'healing')",
    )
    annotation: ArrowAnnotation | None = Field(
        default=None, description="Arrow annotation for this finding"
    )


class OtherBoneFinding(BaseModel):
    """Non-rib bone fracture finding (clavicle, scapula, spine)."""

    bone_name: str = Field(
        ..., description="Bone name (e.g., 'clavicle', 'scapula', 'spine')"
    )
    side: Literal[
        "left", "right", "midline",
        "upper_thoracic", "mid_thoracic", "lower_thoracic"  # Spine regions
    ] = Field(
        ..., description="Anatomical side or spine region"
    )
    bbox: tuple[int, int, int, int] = Field(
        ..., description="Bounding box [x1, y1, x2, y2]"
    )
    fracture_status: Literal["intact", "fractured", "suspicious", "osteoporosis"] = Field(
        default="intact", description="Fracture/pathology classification"
    )
    fracture_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Fracture detection confidence"
    )
    annotation: ArrowAnnotation | None = Field(
        default=None, description="Arrow annotation for this finding"
    )


class RibAnalysisMetadata(BaseModel):
    """Metadata for rib analysis run."""

    model_version: str = Field(default="rib-detector-v1.0", description="Model version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(), description="Analysis timestamp"
    )
    scan_duration_ms: float | None = Field(
        default=None, description="Time taken for analysis in milliseconds"
    )


class RibAnalysisOutput(BaseModel):
    """Complete rib fracture analysis output."""

    metadata: RibAnalysisMetadata = Field(
        default_factory=RibAnalysisMetadata, description="Analysis metadata"
    )
    rib_findings: list[RibFinding] = Field(
        default_factory=list, description="All analyzed ribs with their status"
    )
    fractures_detected: list[RibFinding] = Field(
        default_factory=list, description="Ribs with detected fractures only"
    )
    other_bone_findings: list[OtherBoneFinding] = Field(
        default_factory=list, description="Non-rib bone fractures"
    )
    scan_order_report: list[str] = Field(
        default_factory=list,
        description="Systematic scan log: L1→L10, R1→R10, other bones",
    )
    total_fracture_count: int = Field(
        default=0, description="Total number of fractures detected"
    )


class AnnotatedImageOutput(BaseModel):
    """Output containing annotated image data."""

    original_image_base64: str = Field(
        ..., description="Original image as base64 PNG"
    )
    annotated_image_base64: str = Field(
        ..., description="Image with arrow annotations as base64 PNG"
    )
    annotations: list[ArrowAnnotation] = Field(
        default_factory=list, description="List of all annotations"
    )
    image_dimensions: tuple[int, int] = Field(
        ..., description="Image dimensions [width, height]"
    )


class RibAnalysisResponse(BaseModel):
    """Complete response for /analyze-ribs endpoint."""

    rib_analysis: RibAnalysisOutput = Field(..., description="Rib analysis results")
    annotated_image: AnnotatedImageOutput | None = Field(
        default=None, description="Annotated image if requested"
    )
    dicom_modified: bool = Field(
        default=False, description="Whether DICOM metadata was updated"
    )
    dicom_session_id: str | None = Field(
        default=None,
        description="Session ID for downloading modified DICOM via /analyze-ribs/dicom/{session_id}",
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )
