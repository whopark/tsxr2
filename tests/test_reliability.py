"""Test reliability layer components."""

import pytest


# --- Confidence Assessment Tests ---


def test_assess_confidence_returns_high_for_confident_predictions():
    """assess_confidence should return 'high' when confidence_index >= 0.8."""
    from tsxr2.reliability import assess_confidence

    result = assess_confidence(confidence_index=0.85, abnormality_score=0.3)

    assert result.level == "high"
    assert len(result.warnings) == 0


def test_assess_confidence_returns_medium_for_moderate_confidence():
    """assess_confidence should return 'medium' when 0.5 <= confidence_index < 0.8."""
    from tsxr2.reliability import assess_confidence

    result = assess_confidence(confidence_index=0.65, abnormality_score=0.4)

    assert result.level == "medium"
    assert len(result.warnings) >= 1
    assert any("review" in w.lower() for w in result.warnings)


def test_assess_confidence_returns_low_for_uncertain_predictions():
    """assess_confidence should return 'low' when confidence_index < 0.5."""
    from tsxr2.reliability import assess_confidence

    result = assess_confidence(confidence_index=0.35, abnormality_score=0.5)

    assert result.level == "low"
    assert len(result.warnings) >= 1
    assert any("unreliable" in w.lower() or "caution" in w.lower() for w in result.warnings)


def test_assess_confidence_warns_on_borderline_abnormality():
    """assess_confidence should warn when abnormality_score is borderline (0.4-0.6)."""
    from tsxr2.reliability import assess_confidence

    result = assess_confidence(confidence_index=0.75, abnormality_score=0.5)

    assert any("borderline" in w.lower() for w in result.warnings)


def test_assess_confidence_returns_confidence_assessment():
    """assess_confidence should return a ConfidenceAssessment instance."""
    from tsxr2.reliability import ConfidenceAssessment, assess_confidence

    result = assess_confidence(confidence_index=0.9, abnormality_score=0.1)

    assert isinstance(result, ConfidenceAssessment)
    assert hasattr(result, "level")
    assert hasattr(result, "warnings")
    assert hasattr(result, "requires_review")


# --- Quality Validation Tests ---


def test_validate_image_quality_passes_adequate_image():
    """validate_image_quality should pass when quality checks are adequate."""
    from tsxr2.reliability import validate_image_quality
    from tsxr2.schemas.tsxr_output import QualityChecks

    quality = QualityChecks(rotation="low", inspiration="adequate")

    result = validate_image_quality(quality)

    assert result.is_acceptable is True
    assert len(result.issues) == 0


def test_validate_image_quality_flags_high_rotation():
    """validate_image_quality should flag images with high rotation."""
    from tsxr2.reliability import validate_image_quality
    from tsxr2.schemas.tsxr_output import QualityChecks

    quality = QualityChecks(rotation="high", inspiration="adequate")

    result = validate_image_quality(quality)

    assert result.is_acceptable is False
    assert any("rotation" in issue.lower() for issue in result.issues)


def test_validate_image_quality_flags_inadequate_inspiration():
    """validate_image_quality should flag images with inadequate inspiration."""
    from tsxr2.reliability import validate_image_quality
    from tsxr2.schemas.tsxr_output import QualityChecks

    quality = QualityChecks(rotation="none", inspiration="inadequate")

    result = validate_image_quality(quality)

    assert result.is_acceptable is False
    assert any("inspiration" in issue.lower() for issue in result.issues)


# --- Fallback Report Tests ---


def test_generate_fallback_report_returns_gemini_report():
    """generate_fallback_report should return a valid GeminiReport."""
    from tsxr2.reliability import generate_fallback_report
    from tsxr2.schemas import GeminiReport, TSXrOutput

    tsxr_output = TSXrOutput(
        metadata={"model_version": "tsxr-v2.1", "timestamp": "2024-01-01T00:00:00Z"},
        image_info={"dimensions": [512, 512], "view": "PA"},
        findings=[
            {"label": "Cardiomegaly", "probability": 0.85, "severity": "moderate", "side": "central"},
        ],
        global_scores={"abnormality_score": 0.85, "confidence_index": 0.78},
        quality_checks={"rotation": "low", "inspiration": "adequate"},
    )

    report = generate_fallback_report(tsxr_output)

    assert isinstance(report, GeminiReport)
    assert "Cardiomegaly" in report.findings
    assert report.impression != ""
    assert report.recommendations != ""


def test_generate_fallback_report_includes_all_findings():
    """generate_fallback_report should list all detected findings."""
    from tsxr2.reliability import generate_fallback_report
    from tsxr2.schemas import TSXrOutput

    tsxr_output = TSXrOutput(
        metadata={"model_version": "tsxr-v2.1", "timestamp": "2024-01-01T00:00:00Z"},
        image_info={"dimensions": [512, 512], "view": "PA"},
        findings=[
            {"label": "Pneumonia", "probability": 0.75, "severity": "moderate", "side": "bilateral"},
            {"label": "Pleural Effusion", "probability": 0.65, "severity": "mild", "side": "left"},
        ],
        global_scores={"abnormality_score": 0.75, "confidence_index": 0.70},
        quality_checks={"rotation": "none", "inspiration": "adequate"},
    )

    report = generate_fallback_report(tsxr_output)

    assert "Pneumonia" in report.findings
    assert "Pleural Effusion" in report.findings


def test_generate_fallback_report_handles_no_findings():
    """generate_fallback_report should handle normal studies with no findings."""
    from tsxr2.reliability import generate_fallback_report
    from tsxr2.schemas import TSXrOutput

    tsxr_output = TSXrOutput(
        metadata={"model_version": "tsxr-v2.1", "timestamp": "2024-01-01T00:00:00Z"},
        image_info={"dimensions": [512, 512], "view": "PA"},
        findings=[],
        global_scores={"abnormality_score": 0.1, "confidence_index": 0.95},
        quality_checks={"rotation": "none", "inspiration": "adequate"},
    )

    report = generate_fallback_report(tsxr_output)

    assert "no significant" in report.findings.lower() or "normal" in report.findings.lower()
    assert "normal" in report.impression.lower() or "no acute" in report.impression.lower()


def test_generate_fallback_report_includes_disclaimer():
    """generate_fallback_report should include AI disclaimer."""
    from tsxr2.reliability import generate_fallback_report
    from tsxr2.schemas import TSXrOutput

    tsxr_output = TSXrOutput(
        metadata={"model_version": "tsxr-v2.1", "timestamp": "2024-01-01T00:00:00Z"},
        image_info={"dimensions": [512, 512], "view": "PA"},
        findings=[],
        global_scores={"abnormality_score": 0.1, "confidence_index": 0.95},
        quality_checks={"rotation": "none", "inspiration": "adequate"},
    )

    report = generate_fallback_report(tsxr_output)

    # Should include disclaimer about AI-generated content
    full_text = f"{report.findings} {report.impression} {report.recommendations}"
    assert "ai" in full_text.lower() or "automated" in full_text.lower()


# --- Input Validation Tests ---


def test_validate_dicom_accepts_chest_xray(sample_dicom_path):
    """validate_dicom should accept valid chest X-ray DICOM."""
    from tsxr2.reliability import validate_dicom

    result = validate_dicom(sample_dicom_path)

    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_dicom_rejects_non_dicom_file(tmp_path):
    """validate_dicom should reject non-DICOM files."""
    from tsxr2.reliability import validate_dicom

    # Create a fake file
    fake_file = tmp_path / "fake.dcm"
    fake_file.write_text("This is not a DICOM file")

    result = validate_dicom(fake_file)

    assert result.is_valid is False
    assert len(result.errors) >= 1


def test_validate_dicom_rejects_missing_pixel_data(tmp_path):
    """validate_dicom should reject DICOM without pixel data."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from tsxr2.reliability import validate_dicom

    # Create DICOM without pixel data
    ds = Dataset()
    ds.PatientName = "Test^Patient"
    ds.Modality = "DX"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta

    dcm_path = tmp_path / "no_pixels.dcm"
    pydicom.dcmwrite(dcm_path, ds, enforce_file_format=True)

    result = validate_dicom(dcm_path)

    assert result.is_valid is False
    assert any("pixel" in err.lower() for err in result.errors)


def test_validate_dicom_warns_non_chest_modality(tmp_path):
    """validate_dicom should warn for non-chest modalities."""
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from tsxr2.reliability import validate_dicom

    # Create CT DICOM (not chest X-ray)
    ds = Dataset()
    ds.PatientName = "Test^Patient"
    ds.Modality = "CT"  # CT instead of DX/CR
    ds.BodyPartExamined = "ABDOMEN"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((512, 512), dtype=np.uint16).tobytes()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = file_meta

    dcm_path = tmp_path / "ct_scan.dcm"
    pydicom.dcmwrite(dcm_path, ds, enforce_file_format=True)

    result = validate_dicom(dcm_path)

    # Should still be valid but with warnings
    assert len(result.warnings) >= 1
    assert any("modality" in warn.lower() or "chest" in warn.lower() for warn in result.warnings)
