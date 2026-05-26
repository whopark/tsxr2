"""Input validation for DICOM files and other inputs.

Validates DICOM files meet requirements for chest X-ray analysis
before processing through the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class DicomValidationResult:
    """Result of DICOM file validation.

    Attributes:
        is_valid: Whether the file is valid for processing.
        errors: Critical errors that prevent processing.
        warnings: Non-critical issues to be aware of.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Modalities suitable for chest X-ray analysis
CHEST_XRAY_MODALITIES = {"DX", "CR", "DR"}

# Body parts that indicate chest imaging
CHEST_BODY_PARTS = {"CHEST", "THORAX", "LUNG", "LUNGS"}


def validate_dicom(path: Path | str) -> DicomValidationResult:
    """Validate a DICOM file for chest X-ray analysis.

    Checks that the file:
    1. Is a valid DICOM file
    2. Contains pixel data
    3. Has appropriate modality for chest X-ray
    4. Has required metadata

    Args:
        path: Path to the DICOM file.

    Returns:
        DicomValidationResult with validity status, errors, and warnings.
    """
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    # Check file exists
    if not path.exists():
        return DicomValidationResult(
            is_valid=False,
            errors=["File does not exist."],
        )

    # Try to read as DICOM
    try:
        ds = pydicom.dcmread(path, force=True)
    except (InvalidDicomError, Exception) as e:
        return DicomValidationResult(
            is_valid=False,
            errors=[f"Invalid DICOM file: {str(e)}"],
        )

    # Check for pixel data
    if not hasattr(ds, "PixelData") or ds.PixelData is None:
        errors.append("DICOM file has no pixel data.")

    # Check modality
    modality = getattr(ds, "Modality", None)
    if modality is None:
        warnings.append("Modality not specified in DICOM metadata.")
    elif modality not in CHEST_XRAY_MODALITIES:
        warnings.append(
            f"Modality '{modality}' is not a typical chest X-ray modality. "
            f"Expected one of: {', '.join(CHEST_XRAY_MODALITIES)}."
        )

    # Check body part
    body_part = getattr(ds, "BodyPartExamined", None)
    if body_part is not None and body_part.upper() not in CHEST_BODY_PARTS:
        warnings.append(
            f"Body part '{body_part}' may not be appropriate for chest X-ray analysis."
        )

    # Check required image attributes
    if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
        errors.append("DICOM file missing image dimensions (Rows/Columns).")

    is_valid = len(errors) == 0

    return DicomValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
    )
