"""Test logging and audit trail components."""

import json
import logging
from io import StringIO
from unittest.mock import MagicMock

import pytest


# --- Audit Logger Tests ---


def test_audit_logger_logs_analysis_event():
    """AuditLogger should log analysis events with structured data."""
    from tsxr2.logging import AuditLogger

    # Capture log output
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = AuditLogger()
    logger._logger.addHandler(handler)
    logger._logger.setLevel(logging.INFO)

    logger.log_analysis(
        request_id="test-123",
        findings_count=3,
        abnormality_score=0.75,
        confidence_level="high",
    )

    log_output = log_stream.getvalue()
    assert "test-123" in log_output
    assert "analysis" in log_output.lower()


def test_audit_logger_logs_gemini_call():
    """AuditLogger should log Gemini API calls."""
    from tsxr2.logging import AuditLogger

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = AuditLogger()
    logger._logger.addHandler(handler)
    logger._logger.setLevel(logging.INFO)

    logger.log_gemini_call(
        request_id="test-456",
        success=True,
        latency_ms=250,
        used_fallback=False,
    )

    log_output = log_stream.getvalue()
    assert "test-456" in log_output
    assert "gemini" in log_output.lower()


def test_audit_logger_logs_validation_failure():
    """AuditLogger should log validation failures."""
    from tsxr2.logging import AuditLogger

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = AuditLogger()
    logger._logger.addHandler(handler)
    logger._logger.setLevel(logging.WARNING)

    logger.log_validation_failure(
        request_id="test-789",
        errors=["Invalid DICOM file", "Missing pixel data"],
    )

    log_output = log_stream.getvalue()
    assert "test-789" in log_output
    assert "validation" in log_output.lower()


# --- Request ID Middleware Tests ---


def test_generate_request_id_returns_unique_ids():
    """generate_request_id should return unique identifiers."""
    from tsxr2.logging import generate_request_id

    ids = [generate_request_id() for _ in range(100)]
    unique_ids = set(ids)

    assert len(unique_ids) == 100  # All IDs should be unique


def test_generate_request_id_has_valid_format():
    """generate_request_id should return properly formatted IDs."""
    from tsxr2.logging import generate_request_id

    request_id = generate_request_id()

    # Should be a non-empty string
    assert isinstance(request_id, str)
    assert len(request_id) > 0
    # Should contain alphanumeric characters and hyphens
    assert all(c.isalnum() or c == "-" for c in request_id)
