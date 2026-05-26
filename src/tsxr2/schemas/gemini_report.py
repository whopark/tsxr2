"""Gemini report output schema matching PRD specification."""

from pydantic import BaseModel, Field


class GeminiReport(BaseModel):
    """Gemini-generated clinical report per PRD specification."""

    findings: str = Field(
        ..., description="Structured description of visual findings based on TSXr hints"
    )
    impression: str = Field(..., description="Clinical summary and primary diagnostic conclusion")
    recommendations: str = Field(
        ..., description="Suggested follow-up actions (e.g., CT correlation, follow-up in 6 weeks)"
    )
