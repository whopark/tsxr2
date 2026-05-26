"""DICOM metadata encoder for AI fracture findings.

Encodes rib fracture detection results into DICOM private tags
and standard comment fields for integration with PACS systems.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from tsxr2.schemas.rib_finding import (
    ArrowAnnotation,
    OtherBoneFinding,
    RibAnalysisOutput,
    RibFinding,
)

# Private DICOM tag definitions for TSXr2 AI findings
# Using private block 0x0099 with creator "TSXr2_AI"
PRIVATE_CREATOR = "TSXr2_AI"
PRIVATE_BLOCK = 0x0099

# Private tag addresses (group 0x0099)
# Tag format: (group, element) where element = (block << 8) | offset
PRIVATE_TAGS = {
    "creator": (0x0099, 0x0010),  # Private Creator ID
    "findings_json": (0x0099, 0x1001),  # JSON-encoded rib findings
    "annotations_json": (0x0099, 0x1002),  # JSON-encoded annotations
    "analysis_timestamp": (0x0099, 0x1003),  # Analysis timestamp
    "model_version": (0x0099, 0x1004),  # Model version string
    "fracture_count": (0x0099, 0x1005),  # Total fracture count
    "scan_order_log": (0x0099, 0x1006),  # Scan order report
    "other_bones_json": (0x0099, 0x1007),  # JSON-encoded clavicle/scapula findings
}


def encode_findings_json(findings: Sequence[RibFinding]) -> str:
    """Encode rib findings to JSON string.

    Args:
        findings: List of RibFinding objects.

    Returns:
        JSON string representation of findings.
    """
    findings_data = []
    for f in findings:
        data = {
            "rib_id": f.rib_id,
            "bbox": list(f.bbox),
            "centroid": {"x": f.centroid.x, "y": f.centroid.y},
            "detection_confidence": f.detection_confidence,
            "fracture_status": f.fracture_status,
            "fracture_confidence": f.fracture_confidence,
        }
        if f.fracture_type:
            data["fracture_type"] = f.fracture_type
        findings_data.append(data)

    return json.dumps(findings_data, indent=None)


def encode_other_bones_json(findings: Sequence[OtherBoneFinding]) -> str:
    """Encode other bone findings (clavicle, scapula) to JSON string.

    Args:
        findings: List of OtherBoneFinding objects.

    Returns:
        JSON string representation of findings.
    """
    findings_data = []
    for f in findings:
        data = {
            "bone_name": f.bone_name,
            "side": f.side,
            "bbox": list(f.bbox),
            "fracture_confidence": f.fracture_confidence,
        }
        findings_data.append(data)

    return json.dumps(findings_data, indent=None)


def encode_annotations_json(annotations: Sequence[ArrowAnnotation]) -> str:
    """Encode arrow annotations to JSON string.

    Args:
        annotations: List of ArrowAnnotation objects.

    Returns:
        JSON string representation of annotations.
    """
    annotations_data = []
    for a in annotations:
        data = {
            "target": {"x": a.target_point.x, "y": a.target_point.y},
            "origin": {"x": a.origin_point.x, "y": a.origin_point.y},
            "label": a.label,
            "color": list(a.color),
            "rib_id": a.associated_rib,
        }
        annotations_data.append(data)

    return json.dumps(annotations_data, indent=None)


def format_image_comments(
    rib_findings: Sequence[RibFinding],
    other_bone_findings: Sequence[OtherBoneFinding] | None = None,
) -> str:
    """Format findings as human-readable image comments.

    Args:
        rib_findings: List of RibFinding objects.
        other_bone_findings: Optional list of OtherBoneFinding objects.

    Returns:
        Human-readable comment string for ImageComments tag.
    """
    lines = ["TSXr2 AI Analysis Results:"]

    # Rib findings
    rib_fractures = [f for f in rib_findings if f.fracture_status == "fractured"]
    rib_suspicious = [f for f in rib_findings if f.fracture_status == "suspicious"]

    if rib_fractures:
        rib_ids = ", ".join(f.rib_id for f in rib_fractures)
        lines.append(f"RIB FRACTURES: {rib_ids}")

    if rib_suspicious:
        rib_ids = ", ".join(f.rib_id for f in rib_suspicious)
        lines.append(f"RIB SUSPICIOUS: {rib_ids}")

    # Other bone findings (clavicle, scapula)
    if other_bone_findings:
        # Classify by confidence threshold
        other_fractures = [
            f for f in other_bone_findings if f.fracture_confidence >= 0.75
        ]
        other_suspicious = [
            f for f in other_bone_findings
            if 0.5 <= f.fracture_confidence < 0.75
        ]

        if other_fractures:
            bone_ids = ", ".join(
                f"{f.side.upper()} {f.bone_name.upper()}" for f in other_fractures
            )
            lines.append(f"OTHER BONE FRACTURES: {bone_ids}")

        if other_suspicious:
            bone_ids = ", ".join(
                f"{f.side.upper()} {f.bone_name.upper()}" for f in other_suspicious
            )
            lines.append(f"OTHER BONE SUSPICIOUS: {bone_ids}")

    # Summary
    total_rib = len(rib_fractures) + len(rib_suspicious)
    total_other = 0
    if other_bone_findings:
        total_other = len([
            f for f in other_bone_findings if f.fracture_confidence >= 0.5
        ])

    if total_rib == 0 and total_other == 0:
        lines.append("No fractures detected.")
    else:
        lines.append(
            f"Total: {len(rib_fractures)} rib fractures, "
            f"{len(rib_suspicious)} rib suspicious, "
            f"{total_other} other bone findings"
        )

    lines.append("NOTE: AI findings require radiologist verification.")

    return "\n".join(lines)


def add_fracture_findings_to_dicom(
    dataset: Dataset,
    analysis_output: RibAnalysisOutput,
    annotations: Sequence[ArrowAnnotation] | None = None,
) -> Dataset:
    """Add AI fracture findings to DICOM metadata.

    Creates a deep copy of the dataset and adds private tags with
    the fracture analysis results. Does not modify the original dataset.

    Args:
        dataset: Original pydicom Dataset.
        analysis_output: RibAnalysisOutput from rib detection.
        annotations: Optional list of ArrowAnnotation for coordinates.

    Returns:
        New Dataset with added private tags.
    """
    # Create deep copy to preserve original
    modified = deepcopy(dataset)

    # Get timestamp in DICOM DT format: YYYYMMDDHHMMSS.FFFFFF
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S.%f")

    # Add private creator (required for private tags)
    modified.add_new(PRIVATE_TAGS["creator"], "LO", PRIVATE_CREATOR)

    # Add rib findings JSON
    findings_json = encode_findings_json(analysis_output.rib_findings)
    modified.add_new(PRIVATE_TAGS["findings_json"], "LT", findings_json)

    # Add other bone findings JSON (clavicle, scapula)
    if analysis_output.other_bone_findings:
        other_bones_json = encode_other_bones_json(analysis_output.other_bone_findings)
        modified.add_new(PRIVATE_TAGS["other_bones_json"], "LT", other_bones_json)

    # Add annotations JSON if provided
    if annotations:
        annotations_json = encode_annotations_json(annotations)
        modified.add_new(PRIVATE_TAGS["annotations_json"], "LT", annotations_json)

    # Add metadata
    modified.add_new(PRIVATE_TAGS["analysis_timestamp"], "DT", timestamp)
    modified.add_new(
        PRIVATE_TAGS["model_version"],
        "LO",
        analysis_output.metadata.model_version,
    )
    modified.add_new(
        PRIVATE_TAGS["fracture_count"],
        "IS",
        str(analysis_output.total_fracture_count),
    )

    # Add scan order log
    scan_log = "\n".join(analysis_output.scan_order_report)
    modified.add_new(PRIVATE_TAGS["scan_order_log"], "LT", scan_log)

    # Add human-readable image comments (standard tag)
    image_comments = format_image_comments(
        analysis_output.rib_findings,
        analysis_output.other_bone_findings,
    )
    modified.ImageComments = image_comments

    return modified


def extract_findings_from_dicom(dataset: Dataset) -> dict[str, Any] | None:
    """Extract AI findings from DICOM private tags.

    Args:
        dataset: pydicom Dataset that may contain TSXr2 findings.

    Returns:
        Dict with findings data, or None if no findings present.
    """
    # Check for private creator
    creator_tag = PRIVATE_TAGS["creator"]
    if creator_tag not in dataset:
        return None

    creator = dataset[creator_tag].value
    if creator != PRIVATE_CREATOR:
        return None

    result: dict[str, Any] = {"creator": creator}

    # Extract findings JSON
    findings_tag = PRIVATE_TAGS["findings_json"]
    if findings_tag in dataset:
        try:
            result["findings"] = json.loads(dataset[findings_tag].value)
        except json.JSONDecodeError:
            result["findings"] = None

    # Extract other bone findings JSON
    other_bones_tag = PRIVATE_TAGS["other_bones_json"]
    if other_bones_tag in dataset:
        try:
            result["other_bone_findings"] = json.loads(dataset[other_bones_tag].value)
        except json.JSONDecodeError:
            result["other_bone_findings"] = None

    # Extract annotations JSON
    annotations_tag = PRIVATE_TAGS["annotations_json"]
    if annotations_tag in dataset:
        try:
            result["annotations"] = json.loads(dataset[annotations_tag].value)
        except json.JSONDecodeError:
            result["annotations"] = None

    # Extract metadata
    if PRIVATE_TAGS["analysis_timestamp"] in dataset:
        result["timestamp"] = dataset[PRIVATE_TAGS["analysis_timestamp"]].value

    if PRIVATE_TAGS["model_version"] in dataset:
        result["model_version"] = dataset[PRIVATE_TAGS["model_version"]].value

    if PRIVATE_TAGS["fracture_count"] in dataset:
        result["fracture_count"] = int(dataset[PRIVATE_TAGS["fracture_count"]].value)

    if PRIVATE_TAGS["scan_order_log"] in dataset:
        result["scan_log"] = dataset[PRIVATE_TAGS["scan_order_log"]].value

    return result


def save_annotated_dicom(
    original_dataset: Dataset,
    output_path: Path | str,
    analysis_output: RibAnalysisOutput,
    annotations: Sequence[ArrowAnnotation] | None = None,
    new_series: bool = True,
) -> Path:
    """Save DICOM with embedded AI findings metadata.

    Args:
        original_dataset: Original pydicom Dataset.
        output_path: Path to save the annotated DICOM.
        analysis_output: RibAnalysisOutput from rib detection.
        annotations: Optional list of ArrowAnnotation.
        new_series: If True, generate new Series/SOP Instance UIDs.

    Returns:
        Path to the saved DICOM file.
    """
    output_path = Path(output_path)

    # Add findings to DICOM
    modified = add_fracture_findings_to_dicom(
        original_dataset, analysis_output, annotations
    )

    # Generate new UIDs if creating a new series
    if new_series:
        modified.SeriesInstanceUID = generate_uid()
        modified.SOPInstanceUID = generate_uid()
        modified.SeriesDescription = (
            f"{modified.get('SeriesDescription', 'Chest')} - TSXr2 AI Analysis"
        )

    # Save the modified DICOM
    modified.save_as(output_path)

    return output_path
