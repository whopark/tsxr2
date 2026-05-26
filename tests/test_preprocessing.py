"""Test preprocessing pipeline components."""

from pathlib import Path

import numpy as np
import pytest


def test_load_dicom_returns_pixel_array(sample_dicom_path: Path):
    """load_dicom should return pixel data as numpy array with correct shape."""
    from tsxr2.preprocessing import load_dicom

    result = load_dicom(sample_dicom_path)

    assert isinstance(result.pixel_array, np.ndarray)
    assert result.pixel_array.shape == (512, 512)
    assert result.pixel_array.dtype == np.uint16


def test_load_dicom_raises_on_missing_file():
    """load_dicom should raise FileNotFoundError for non-existent path."""
    from tsxr2.preprocessing import load_dicom

    with pytest.raises(FileNotFoundError):
        load_dicom("/nonexistent/path/file.dcm")


def test_load_dicom_extracts_metadata(sample_dicom_path: Path):
    """load_dicom should extract key metadata from DICOM."""
    from tsxr2.preprocessing import load_dicom

    result = load_dicom(sample_dicom_path)

    assert result.metadata["modality"] == "DX"
    assert result.metadata["body_part"] == "CHEST"
    assert result.metadata["view_position"] == "PA"
    assert result.metadata["rows"] == 512
    assert result.metadata["columns"] == 512


# --- PHI Removal Tests ---


def test_remove_phi_clears_patient_name(sample_dicom_path: Path):
    """remove_phi should clear PatientName from DICOM dataset."""
    import pydicom

    from tsxr2.preprocessing import remove_phi

    ds = pydicom.dcmread(sample_dicom_path)
    assert ds.PatientName == "Test^Patient"  # Verify PHI exists

    anonymized = remove_phi(ds)

    # PatientName should be empty or removed
    patient_name = getattr(anonymized, "PatientName", "")
    assert patient_name == "" or patient_name is None


def test_remove_phi_clears_all_identifiers(sample_dicom_path: Path):
    """remove_phi should clear all PHI identifiers from DICOM dataset."""
    import pydicom

    from tsxr2.preprocessing import remove_phi

    ds = pydicom.dcmread(sample_dicom_path)

    # Verify PHI exists before anonymization
    assert ds.PatientID == "12345678"
    assert ds.AccessionNumber == "ACC123456"
    assert str(ds.ReferringPhysicianName) == "Dr^Referring"

    anonymized = remove_phi(ds)

    # All PHI should be cleared
    assert anonymized.PatientID == "ANONYMOUS"
    assert anonymized.AccessionNumber == ""
    assert anonymized.ReferringPhysicianName == ""


def test_remove_phi_preserves_original_dataset(sample_dicom_path: Path):
    """remove_phi should not modify the original dataset."""
    import pydicom

    from tsxr2.preprocessing import remove_phi

    ds = pydicom.dcmread(sample_dicom_path)
    original_patient_name = str(ds.PatientName)

    _ = remove_phi(ds)

    # Original should be unchanged
    assert str(ds.PatientName) == original_patient_name


def test_remove_phi_preserves_clinical_data(sample_dicom_path: Path):
    """remove_phi should preserve non-PHI clinical data."""
    import pydicom

    from tsxr2.preprocessing import remove_phi

    ds = pydicom.dcmread(sample_dicom_path)
    anonymized = remove_phi(ds)

    # Clinical data should be preserved
    assert anonymized.Modality == "DX"
    assert anonymized.BodyPartExamined == "CHEST"
    assert anonymized.ViewPosition == "PA"
    assert anonymized.Rows == 512
    assert anonymized.Columns == 512
    # Pixel data should be preserved
    assert len(anonymized.PixelData) > 0


# --- Image Normalization Tests ---


def test_normalize_image_returns_correct_shape(sample_dicom_path: Path):
    """normalize_image should return a 512x512x3 uint8 array."""
    from tsxr2.preprocessing import load_dicom, normalize_image

    dicom_data = load_dicom(sample_dicom_path)
    result = normalize_image(dicom_data.pixel_array)

    assert result.shape == (512, 512, 3)
    assert result.dtype == np.uint8


def test_normalize_image_value_range(sample_dicom_path: Path):
    """normalize_image output should have values in [0, 255]."""
    from tsxr2.preprocessing import load_dicom, normalize_image

    dicom_data = load_dicom(sample_dicom_path)
    result = normalize_image(dicom_data.pixel_array)

    assert result.min() >= 0
    assert result.max() <= 255


def test_normalize_image_resizes_non_square():
    """normalize_image should resize non-square images correctly."""
    from tsxr2.preprocessing import normalize_image

    # Create a non-square test array
    test_array = np.random.randint(0, 4096, (256, 384), dtype=np.uint16)
    result = normalize_image(test_array, target_size=(512, 512))

    assert result.shape == (512, 512, 3)


def test_apply_window_clips_values():
    """apply_window should clip values outside window range."""
    from tsxr2.preprocessing.normalizer import apply_window

    # Array with values outside typical window
    test_array = np.array([[0, 1000, 2048, 3000, 4095]], dtype=np.uint16)

    # Window that clips some values
    result = apply_window(test_array, window_center=2048, window_width=2000)

    # Values should be scaled to 0-255 range
    assert result.min() >= 0
    assert result.max() <= 255
    # Center value (2048) should map to ~127
    assert 120 < result[0, 2] < 135
