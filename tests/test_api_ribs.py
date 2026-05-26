"""Tests for /analyze-ribs API endpoint."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_analyze_ribs_endpoint_returns_200(sample_dicom_path: Path):
    """POST /analyze-ribs should return HTTP 200 with valid DICOM."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_analyze_ribs_returns_rib_analysis(sample_dicom_path: Path):
    """POST /analyze-ribs should return rib analysis data."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    data = response.json()

    # Should have rib_analysis section
    assert "rib_analysis" in data
    assert "rib_findings" in data["rib_analysis"]
    assert "fractures_detected" in data["rib_analysis"]
    assert "scan_order_report" in data["rib_analysis"]


@pytest.mark.asyncio
async def test_analyze_ribs_systematic_scan_order(sample_dicom_path: Path):
    """Scan order should follow L1→L10, R1→R10 protocol."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    data = response.json()
    scan_report = data["rib_analysis"]["scan_order_report"]

    # First entry should be L1
    assert scan_report[0].startswith("L1:")
    # Entry 10 should be L10
    assert scan_report[9].startswith("L10:")
    # Entry 11 should be R1
    assert scan_report[10].startswith("R1:")


@pytest.mark.asyncio
async def test_analyze_ribs_returns_annotated_image(sample_dicom_path: Path):
    """Should return annotated image when include_annotations=True (default)."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"include_annotations": "true"},
            )

    data = response.json()

    # Should have annotated_image section
    assert "annotated_image" in data
    assert data["annotated_image"] is not None
    assert "original_image_base64" in data["annotated_image"]
    assert "annotated_image_base64" in data["annotated_image"]
    assert "annotations" in data["annotated_image"]


@pytest.mark.asyncio
async def test_analyze_ribs_no_annotations_when_disabled(sample_dicom_path: Path):
    """Should not return annotated image when include_annotations=False."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"include_annotations": "false"},
            )

    data = response.json()

    # annotated_image should be None
    assert data["annotated_image"] is None


@pytest.mark.asyncio
async def test_analyze_ribs_detects_fractures(sample_dicom_path: Path):
    """Simulation should detect some fractures."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    data = response.json()

    # Simulation has some fractures for testing
    assert data["rib_analysis"]["total_fracture_count"] > 0
    assert len(data["rib_analysis"]["fractures_detected"]) > 0


@pytest.mark.asyncio
async def test_analyze_ribs_has_processing_time(sample_dicom_path: Path):
    """Response should include processing time."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    data = response.json()

    assert "processing_time_ms" in data
    assert data["processing_time_ms"] > 0


@pytest.mark.asyncio
async def test_analyze_ribs_rib_findings_structure(sample_dicom_path: Path):
    """Rib findings should have expected structure."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    data = response.json()
    rib_findings = data["rib_analysis"]["rib_findings"]

    # Should have 20 rib findings (L1-L10, R1-R10)
    assert len(rib_findings) == 20

    # Each finding should have expected fields
    for finding in rib_findings:
        assert "rib_id" in finding
        assert "bbox" in finding
        assert "centroid" in finding
        assert "detection_confidence" in finding
        assert "fracture_status" in finding
        assert "fracture_confidence" in finding


@pytest.mark.asyncio
async def test_analyze_ribs_annotations_match_fractures(sample_dicom_path: Path):
    """Annotations should correspond to detected fractures/suspicious ribs."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"include_annotations": "true"},
            )

    data = response.json()

    # Get fractures and suspicious ribs
    non_intact = [
        f for f in data["rib_analysis"]["rib_findings"]
        if f["fracture_status"] in ("fractured", "suspicious")
    ]

    # Annotations should match
    annotations = data["annotated_image"]["annotations"]
    assert len(annotations) == len(non_intact)


@pytest.mark.asyncio
async def test_analyze_ribs_metadata(sample_dicom_path: Path):
    """Response should include metadata with model version."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
            )

    data = response.json()

    assert "metadata" in data["rib_analysis"]
    assert data["rib_analysis"]["metadata"]["model_version"] == "rib-detector-v1.0"


@pytest.mark.asyncio
async def test_analyze_ribs_rejects_no_filename():
    """Should reject request without filename."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/analyze-ribs",
            files={"file": ("", b"", "application/dicom")},
        )

    # FastAPI returns 422 (Unprocessable Entity) for validation errors
    # or 400 for our custom validation
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_analyze_ribs_in_openapi_schema():
    """OpenAPI schema should include /analyze-ribs endpoint."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    schema = response.json()
    assert "/analyze-ribs" in schema["paths"]
    assert "post" in schema["paths"]["/analyze-ribs"]


@pytest.mark.asyncio
async def test_analyze_ribs_embed_in_dicom(sample_dicom_path: Path):
    """POST /analyze-ribs with embed_in_dicom=True should return session_id."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"embed_in_dicom": "true"},
            )

    assert response.status_code == 200
    data = response.json()

    # Should have DICOM modified flag and session ID
    assert data["dicom_modified"] is True
    assert data["dicom_session_id"] is not None
    assert data["dicom_session_id"].startswith("rib_analysis_")


@pytest.mark.asyncio
async def test_download_modified_dicom(sample_dicom_path: Path):
    """GET /analyze-ribs/dicom/{session_id} should return DICOM file."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First, analyze with embed_in_dicom=True
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"embed_in_dicom": "true"},
            )

        data = response.json()
        session_id = data["dicom_session_id"]

        # Now download the modified DICOM
        download_response = await client.get(f"/analyze-ribs/dicom/{session_id}")

    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "application/dicom"
    assert "attachment" in download_response.headers["content-disposition"]

    # Should have valid DICOM content
    dicom_bytes = download_response.content
    assert len(dicom_bytes) > 0


@pytest.mark.asyncio
async def test_extract_dicom_findings(sample_dicom_path: Path):
    """GET /analyze-ribs/extract/{session_id} should return findings."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First, analyze with embed_in_dicom=True
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"embed_in_dicom": "true"},
            )

        data = response.json()
        session_id = data["dicom_session_id"]

        # Extract findings from cached DICOM
        extract_response = await client.get(f"/analyze-ribs/extract/{session_id}")

    assert extract_response.status_code == 200
    extract_data = extract_response.json()

    # Should have findings
    assert "findings" in extract_data
    assert extract_data["findings"]["creator"] == "TSXr2_AI"
    assert "findings" in extract_data["findings"]
    assert "model_version" in extract_data["findings"]

    # Should have human-readable image comments
    assert "image_comments" in extract_data
    assert "TSXr2 AI Analysis" in extract_data["image_comments"]


@pytest.mark.asyncio
async def test_download_invalid_session_returns_404():
    """GET /analyze-ribs/dicom/{session_id} with invalid ID should return 404."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/analyze-ribs/dicom/invalid_session_123")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_no_embed_returns_no_session_id(sample_dicom_path: Path):
    """POST /analyze-ribs without embed_in_dicom should not return session_id."""
    from tsxr2.api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(sample_dicom_path, "rb") as f:
            response = await client.post(
                "/analyze-ribs",
                files={"file": ("test.dcm", f, "application/dicom")},
                params={"embed_in_dicom": "false"},
            )

    data = response.json()
    assert data["dicom_modified"] is False
    assert data["dicom_session_id"] is None
