"""Pytest fixtures for TSXr2 tests."""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


@pytest.fixture
def sample_dicom_path(tmp_path: Path) -> Path:
    """Create a synthetic chest X-ray DICOM file for testing.

    This generates a minimal valid DICOM with:
    - Standard chest X-ray metadata
    - PHI fields (for anonymization testing)
    - 512x512 pixel array
    """
    # Create file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"  # Digital X-Ray
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create dataset
    ds = Dataset()
    ds.file_meta = file_meta

    # Patient info (PHI - to be anonymized)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345678"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "M"

    # Study info
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = "20231027"
    ds.StudyTime = "100000"
    ds.AccessionNumber = "ACC123456"
    ds.ReferringPhysicianName = "Dr^Referring"

    # Series info
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "DX"  # Digital X-Ray
    ds.BodyPartExamined = "CHEST"

    # Instance info
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = 1

    # Image info
    ds.ViewPosition = "PA"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Window settings for chest X-ray
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096

    # Create synthetic pixel data (simulated chest X-ray pattern)
    rng = np.random.default_rng(42)
    # Base lung field (darker in center)
    y, x = np.ogrid[:512, :512]
    center_y, center_x = 256, 256
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Simulate lung fields (darker) with some texture
    base = 2048 - (300 * np.exp(-distance / 200))
    noise = rng.integers(-100, 100, (512, 512))
    pixel_data = (base + noise).clip(0, 4095).astype(np.uint16)

    ds.PixelData = pixel_data.tobytes()

    # Save DICOM file with proper header
    dicom_path = tmp_path / "test_chest_xray.dcm"
    pydicom.dcmwrite(dicom_path, ds, enforce_file_format=True)

    return dicom_path


@pytest.fixture
def sample_dicom_dataset(sample_dicom_path: Path) -> Dataset:
    """Load the sample DICOM as a pydicom Dataset."""
    return pydicom.dcmread(sample_dicom_path)
