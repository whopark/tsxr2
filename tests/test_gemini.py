"""Test Gemini reasoning layer components."""

import os
from unittest.mock import MagicMock, patch

import pytest


# --- Gemini Client Tests ---


def test_gemini_client_initializes_with_api_key():
    """GeminiClient should initialize with API key from env or parameter."""
    from tsxr2.gemini import GeminiClient

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key"}):
        client = GeminiClient()

    assert client.api_key == "test-api-key"


def test_gemini_client_raises_without_api_key():
    """GeminiClient should raise ValueError if no API key provided."""
    from tsxr2.gemini import GeminiClient

    with patch.dict(os.environ, {}, clear=True):
        # Remove GEMINI_API_KEY if it exists
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(ValueError, match="API key"):
            GeminiClient()


# --- Prompt Template Tests ---


def test_build_report_prompt_includes_findings():
    """build_report_prompt should include TSXr findings in the prompt."""
    from tsxr2.gemini import build_report_prompt
    from tsxr2.schemas import TSXrOutput

    # Create a sample TSXrOutput
    tsxr_output = TSXrOutput(
        metadata={"model_version": "tsxr-v2.1", "timestamp": "2024-01-01T00:00:00Z"},
        image_info={"dimensions": [512, 512], "view": "PA"},
        findings=[
            {"label": "Cardiomegaly", "probability": 0.85, "severity": "moderate", "side": "central"},
            {"label": "Pleural Effusion", "probability": 0.72, "severity": "moderate", "side": "bilateral"},
        ],
        global_scores={"abnormality_score": 0.85, "confidence_index": 0.78},
        quality_checks={"rotation": "low", "inspiration": "adequate"},
    )

    prompt = build_report_prompt(tsxr_output)

    # Should include key findings
    assert "Cardiomegaly" in prompt
    assert "Pleural Effusion" in prompt
    assert "0.85" in prompt or "85%" in prompt
    assert "PA" in prompt


def test_build_report_prompt_includes_clinical_context():
    """build_report_prompt should include instructions for clinical report format."""
    from tsxr2.gemini import build_report_prompt
    from tsxr2.schemas import TSXrOutput

    tsxr_output = TSXrOutput(
        metadata={"model_version": "tsxr-v2.1", "timestamp": "2024-01-01T00:00:00Z"},
        image_info={"dimensions": [512, 512], "view": "PA"},
        findings=[],
        global_scores={"abnormality_score": 0.1, "confidence_index": 0.95},
        quality_checks={"rotation": "none", "inspiration": "adequate"},
    )

    prompt = build_report_prompt(tsxr_output)

    # Should include clinical report structure
    assert "findings" in prompt.lower() or "Findings" in prompt
    assert "impression" in prompt.lower() or "Impression" in prompt
    assert "recommendation" in prompt.lower() or "Recommendation" in prompt


# --- Response Parser Tests ---


def test_parse_gemini_response_extracts_sections():
    """parse_gemini_response should extract findings, impression, recommendations."""
    from tsxr2.gemini import parse_gemini_response

    response_text = """## Findings
The chest X-ray shows evidence of cardiomegaly with mild bilateral pleural effusions.
The cardiac silhouette is enlarged.

## Impression
1. Cardiomegaly
2. Bilateral pleural effusions, mild

## Recommendations
- Recommend follow-up echocardiogram to assess cardiac function
- Clinical correlation advised
"""

    report = parse_gemini_response(response_text)

    assert "cardiomegaly" in report.findings.lower()
    assert "Cardiomegaly" in report.impression
    assert "echocardiogram" in report.recommendations.lower()


def test_parse_gemini_response_handles_missing_sections():
    """parse_gemini_response should handle responses with missing sections gracefully."""
    from tsxr2.gemini import parse_gemini_response

    # Minimal response without all sections
    response_text = """## Findings
Normal chest X-ray with no acute abnormalities.
"""

    report = parse_gemini_response(response_text)

    assert "Normal" in report.findings
    # Missing sections should have placeholder text
    assert report.impression != ""
    assert report.recommendations != ""


def test_parse_gemini_response_returns_gemini_report():
    """parse_gemini_response should return a GeminiReport instance."""
    from tsxr2.gemini import parse_gemini_response
    from tsxr2.schemas import GeminiReport

    response_text = """## Findings
Test finding

## Impression
Test impression

## Recommendations
Test recommendation
"""

    report = parse_gemini_response(response_text)

    assert isinstance(report, GeminiReport)
    assert report.findings is not None
    assert report.impression is not None
    assert report.recommendations is not None


# --- Retry Logic Tests ---


def test_retry_decorator_retries_on_failure():
    """retry_with_backoff should retry failed calls up to max_retries."""
    from tsxr2.gemini.retry import retry_with_backoff

    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Simulated API failure")
        return "success"

    result = failing_function()

    assert result == "success"
    assert call_count == 3  # Failed twice, succeeded on third


def test_retry_decorator_raises_after_max_retries():
    """retry_with_backoff should raise after exhausting retries."""
    from tsxr2.gemini.retry import retry_with_backoff

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def always_fails():
        raise ConnectionError("Always fails")

    with pytest.raises(ConnectionError):
        always_fails()


def test_retry_decorator_does_not_retry_on_success():
    """retry_with_backoff should not retry successful calls."""
    from tsxr2.gemini.retry import retry_with_backoff

    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def succeeds_immediately():
        nonlocal call_count
        call_count += 1
        return "immediate success"

    result = succeeds_immediately()

    assert result == "immediate success"
    assert call_count == 1
