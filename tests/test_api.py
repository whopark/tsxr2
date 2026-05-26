"""Test API endpoints."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    """GET /health should return HTTP 200 with status healthy."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_preprocess_endpoint_accepts_dicom(sample_dicom_path: Path):
    """POST /preprocess should accept DICOM upload and return processed data."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/preprocess",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    assert response.status_code == 200
    data = response.json()
    assert "image_base64" in data
    assert "metadata" in data
    assert data["metadata"]["modality"] == "DX"


@pytest.mark.asyncio
async def test_analyze_endpoint_returns_tsxr_output(sample_dicom_path: Path):
    """POST /analyze should return TSXrOutput schema."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    assert response.status_code == 200
    data = response.json()

    # Should match TSXrOutput schema
    assert "metadata" in data
    assert "image_info" in data
    assert "findings" in data
    assert "global_scores" in data
    assert "quality_checks" in data

    # Validate structure
    assert data["metadata"]["model_version"] == "tsxr-v2.1"
    assert data["image_info"]["dimensions"] == [512, 512]
    assert 0.0 <= data["global_scores"]["abnormality_score"] <= 1.0


@pytest.mark.asyncio
async def test_report_endpoint_returns_gemini_report(sample_dicom_path: Path, monkeypatch):
    """POST /report should return GeminiReport schema."""
    from tsxr2.api.main import app

    # Mock Gemini API response
    mock_response = """## Findings
The AI analysis indicates a normal chest X-ray with no acute abnormalities detected.

## Impression
Normal chest radiograph.

## Recommendations
No follow-up imaging required.
"""

    def mock_generate(self, prompt, **kwargs):
        return mock_response

    # Patch the GeminiClient.generate method
    from tsxr2.gemini import GeminiClient
    monkeypatch.setattr(GeminiClient, "generate", mock_generate)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/report",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    assert response.status_code == 200
    data = response.json()

    # Should match GeminiReport schema
    assert "findings" in data
    assert "impression" in data
    assert "recommendations" in data
    assert "Normal" in data["findings"] or "normal" in data["findings"]


@pytest.mark.asyncio
async def test_full_report_endpoint_returns_comprehensive_response(sample_dicom_path: Path, monkeypatch):
    """POST /full-report should return TSXr analysis, Gemini report, and reliability info."""
    from tsxr2.api.main import app

    # Mock Gemini API response
    mock_response = """## Findings
Normal chest radiograph with no acute abnormalities.

## Impression
Normal study.

## Recommendations
No follow-up required.
"""

    def mock_generate(self, prompt, **kwargs):
        return mock_response

    from tsxr2.gemini import GeminiClient
    monkeypatch.setattr(GeminiClient, "generate", mock_generate)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/full-report",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    assert response.status_code == 200
    data = response.json()

    # Should have all sections
    assert "tsxr_output" in data
    assert "gemini_report" in data
    assert "reliability" in data
    assert "validation" in data

    # TSXr output structure
    assert "findings" in data["tsxr_output"]
    assert "global_scores" in data["tsxr_output"]

    # Gemini report structure
    assert "findings" in data["gemini_report"]
    assert "impression" in data["gemini_report"]

    # Reliability structure
    assert "confidence" in data["reliability"]
    assert "quality" in data["reliability"]


@pytest.mark.asyncio
async def test_full_report_uses_fallback_when_gemini_fails(sample_dicom_path: Path, monkeypatch):
    """POST /full-report should use fallback report when Gemini API fails."""
    from tsxr2.api.main import app

    def mock_generate_fails(self, prompt, **kwargs):
        raise ConnectionError("Gemini API unavailable")

    from tsxr2.gemini import GeminiClient
    monkeypatch.setattr(GeminiClient, "generate", mock_generate_fails)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/full-report",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    assert response.status_code == 200
    data = response.json()

    # Should still have all sections
    assert "gemini_report" in data
    assert "reliability" in data

    # Should indicate fallback was used
    assert data["reliability"]["used_fallback"] is True


@pytest.mark.asyncio
async def test_health_detailed_returns_component_status():
    """GET /health/detailed should return status of all components."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/detailed")

    assert response.status_code == 200
    data = response.json()

    # Should have component status
    assert "status" in data
    assert "components" in data
    assert "model" in data["components"]
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_detailed_shows_model_loaded():
    """GET /health/detailed should show model is loaded."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/detailed")

    data = response.json()

    # Model should be loaded and healthy
    assert data["components"]["model"]["status"] == "healthy"
    assert data["components"]["model"]["loaded"] is True


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_counters():
    """GET /metrics should return request counters and latencies."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    data = response.json()

    # Should have metrics
    assert "requests" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_openapi_schema_is_generated():
    """OpenAPI schema should be generated with all endpoints documented."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    # Should have API info
    assert "info" in schema
    assert schema["info"]["title"] == "TSXr2"

    # Should have all endpoints documented
    paths = schema["paths"]
    assert "/health" in paths
    assert "/health/detailed" in paths
    assert "/analyze" in paths
    assert "/report" in paths
    assert "/full-report" in paths


@pytest.mark.asyncio
async def test_openapi_has_endpoint_descriptions():
    """OpenAPI schema should have descriptions for all endpoints."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    schema = response.json()
    paths = schema["paths"]

    # Full-report endpoint should have description
    full_report = paths["/full-report"]["post"]
    assert "summary" in full_report or "description" in full_report
