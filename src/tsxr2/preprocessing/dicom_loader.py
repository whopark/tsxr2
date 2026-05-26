"""DICOM file loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from numpy.typing import NDArray


@dataclass
class DicomData:
    """Container for loaded DICOM data."""

    pixel_array: NDArray[np.uint16]
    metadata: dict[str, Any]
    original_dataset: pydicom.Dataset


def load_dicom(path: Path | str) -> DicomData:
    """Load a DICOM file and extract pixel data.

    Args:
        path: Path to the DICOM file.

    Returns:
        DicomData containing pixel array and metadata.

    Raises:
        FileNotFoundError: If the DICOM file doesn't exist.
        pydicom.errors.InvalidDicomError: If the file is not valid DICOM.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(path)

    # Extract key metadata
    metadata = {
        "modality": getattr(ds, "Modality", None),
        "body_part": getattr(ds, "BodyPartExamined", None),
        "view_position": getattr(ds, "ViewPosition", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "bits_stored": getattr(ds, "BitsStored", None),
        "window_center": getattr(ds, "WindowCenter", None),
        "window_width": getattr(ds, "WindowWidth", None),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", None),
    }

    return DicomData(
        pixel_array=ds.pixel_array,
        metadata=metadata,
        original_dataset=ds,
    )
