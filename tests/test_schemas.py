"""Test Pydantic schemas match PRD specifications."""

import pytest


def test_tsxr_output_schema_validates_correct_json():
    """TSXrOutput should validate a JSON payload matching PRD spec."""
    from tsxr2.schemas import TSXrOutput

    valid_payload = {
        "metadata": {
            "model_version": "tsxr-v2.1",
            "timestamp": "2023-10-27T10:00:00Z",
        },
        "image_info": {
            "dimensions": [512, 512],
            "view": "PA",
        },
        "findings": [
            {
                "label": "Pneumonia",
                "probability": 0.89,
                "severity": "moderate",
                "side": "right",
                "bbox": [120, 210, 160, 260],
            }
        ],
        "global_scores": {
            "abnormality_score": 0.92,
            "confidence_index": 0.88,
        },
        "quality_checks": {
            "rotation": "low",
            "inspiration": "adequate",
        },
    }

    result = TSXrOutput.model_validate(valid_payload)

    assert result.metadata.model_version == "tsxr-v2.1"
    assert len(result.findings) == 1
    assert result.findings[0].label == "Pneumonia"
    assert result.global_scores.abnormality_score == 0.92


def test_tsxr_output_rejects_invalid_probability():
    """TSXrOutput should reject probability values outside [0.0, 1.0]."""
    from pydantic import ValidationError

    from tsxr2.schemas import TSXrOutput

    invalid_payload = {
        "metadata": {"model_version": "tsxr-v2.1", "timestamp": "2023-10-27T10:00:00Z"},
        "image_info": {"dimensions": [512, 512], "view": "PA"},
        "findings": [
            {
                "label": "Pneumonia",
                "probability": 1.5,  # Invalid: > 1.0
                "severity": "moderate",
                "side": "right",
            }
        ],
        "global_scores": {"abnormality_score": 0.92, "confidence_index": 0.88},
        "quality_checks": {"rotation": "low", "inspiration": "adequate"},
    }

    with pytest.raises(ValidationError):
        TSXrOutput.model_validate(invalid_payload)


def test_gemini_report_schema_validates_correct_json():
    """GeminiReport should validate a JSON payload matching PRD spec."""
    from tsxr2.schemas import GeminiReport

    valid_payload = {
        "findings": "Right lower lobe opacity consistent with pneumonia.",
        "impression": "Findings suggestive of community-acquired pneumonia.",
        "recommendations": "Clinical correlation recommended. Consider CT if no improvement.",
    }

    result = GeminiReport.model_validate(valid_payload)

    assert "pneumonia" in result.findings.lower()
    assert result.recommendations is not None


def test_gemini_report_rejects_missing_required_fields():
    """GeminiReport should reject payloads missing required fields."""
    from pydantic import ValidationError

    from tsxr2.schemas import GeminiReport

    incomplete_payload = {
        "findings": "Some findings",
        # Missing: impression, recommendations
    }

    with pytest.raises(ValidationError):
        GeminiReport.model_validate(incomplete_payload)
