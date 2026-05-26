"""DICOM metadata writer module for AI findings.

Provides utilities for encoding AI fracture detection results
into DICOM private tags and standard metadata fields.
"""

from tsxr2.dicom_writer.metadata_encoder import (
    PRIVATE_CREATOR,
    PRIVATE_TAGS,
    add_fracture_findings_to_dicom,
    encode_annotations_json,
    encode_findings_json,
    encode_other_bones_json,
    extract_findings_from_dicom,
    format_image_comments,
    save_annotated_dicom,
)

__all__ = [
    "PRIVATE_CREATOR",
    "PRIVATE_TAGS",
    "add_fracture_findings_to_dicom",
    "encode_annotations_json",
    "encode_findings_json",
    "encode_other_bones_json",
    "extract_findings_from_dicom",
    "format_image_comments",
    "save_annotated_dicom",
]
