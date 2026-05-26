"""Audit logging for TSXr2 API operations.

Provides structured logging for analysis events, API calls,
and validation failures to support compliance and debugging.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any


class AuditLogger:
    """Structured audit logger for clinical analysis events.

    Logs events in JSON format for easy parsing and analysis.
    Supports compliance requirements for medical imaging systems.
    """

    def __init__(self, name: str = "tsxr2.audit") -> None:
        """Initialize audit logger.

        Args:
            name: Logger name for the Python logging module.
        """
        self._logger = logging.getLogger(name)

    def _log_event(
        self,
        event_type: str,
        level: int,
        **data: Any,
    ) -> None:
        """Log a structured event.

        Args:
            event_type: Type of event (analysis, gemini_call, validation, etc.)
            level: Logging level (INFO, WARNING, ERROR)
            **data: Event-specific data fields.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        self._logger.log(level, json.dumps(event))

    def log_analysis(
        self,
        request_id: str,
        findings_count: int,
        abnormality_score: float,
        confidence_level: str,
        **extra: Any,
    ) -> None:
        """Log an analysis completion event.

        Args:
            request_id: Unique request identifier.
            findings_count: Number of findings detected.
            abnormality_score: Overall abnormality score.
            confidence_level: Confidence assessment level.
            **extra: Additional data to log.
        """
        self._log_event(
            event_type="analysis_complete",
            level=logging.INFO,
            request_id=request_id,
            findings_count=findings_count,
            abnormality_score=abnormality_score,
            confidence_level=confidence_level,
            **extra,
        )

    def log_gemini_call(
        self,
        request_id: str,
        success: bool,
        latency_ms: float,
        used_fallback: bool,
        **extra: Any,
    ) -> None:
        """Log a Gemini API call event.

        Args:
            request_id: Unique request identifier.
            success: Whether the call succeeded.
            latency_ms: Call latency in milliseconds.
            used_fallback: Whether fallback was used.
            **extra: Additional data to log.
        """
        level = logging.INFO if success else logging.WARNING
        self._log_event(
            event_type="gemini_call",
            level=level,
            request_id=request_id,
            success=success,
            latency_ms=latency_ms,
            used_fallback=used_fallback,
            **extra,
        )

    def log_validation_failure(
        self,
        request_id: str,
        errors: list[str],
        **extra: Any,
    ) -> None:
        """Log a validation failure event.

        Args:
            request_id: Unique request identifier.
            errors: List of validation error messages.
            **extra: Additional data to log.
        """
        self._log_event(
            event_type="validation_failure",
            level=logging.WARNING,
            request_id=request_id,
            errors=errors,
            **extra,
        )

    def log_error(
        self,
        request_id: str,
        error_type: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Log an error event.

        Args:
            request_id: Unique request identifier.
            error_type: Type of error (e.g., "processing_error").
            message: Error message.
            **extra: Additional data to log.
        """
        self._log_event(
            event_type="error",
            level=logging.ERROR,
            request_id=request_id,
            error_type=error_type,
            message=message,
            **extra,
        )


def generate_request_id() -> str:
    """Generate a unique request identifier.

    Returns:
        UUID string for request tracking.
    """
    return str(uuid.uuid4())
