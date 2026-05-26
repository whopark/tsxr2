"""Fallback report generation when Gemini API is unavailable.

Provides template-based report generation using TSXr findings
to ensure clinical workflow continuity.
"""

from tsxr2.schemas import GeminiReport, TSXrOutput


def generate_fallback_report(tsxr_output: TSXrOutput) -> GeminiReport:
    """Generate a fallback clinical report from TSXr findings.

    Creates a structured report using template-based generation
    when the Gemini API is unavailable. The report includes all
    detected findings with appropriate clinical context.

    Args:
        tsxr_output: TSXr model output with findings and scores.

    Returns:
        GeminiReport with findings, impression, and recommendations.
    """
    # Build findings section
    if tsxr_output.findings:
        findings_lines = [
            "AI-detected findings on chest radiograph:",
            "",
        ]
        for finding in tsxr_output.findings:
            severity = finding.severity.capitalize()
            side = finding.side.capitalize()
            probability = finding.probability * 100
            findings_lines.append(
                f"- {finding.label}: {severity} severity, {side} "
                f"(confidence: {probability:.0f}%)"
            )
        findings_text = "\n".join(findings_lines)
    else:
        findings_text = (
            "No significant abnormalities detected by AI analysis.\n"
            "The chest radiograph appears within normal limits."
        )

    # Build impression
    if tsxr_output.findings:
        # Sort findings by probability for impression
        sorted_findings = sorted(
            tsxr_output.findings,
            key=lambda f: f.probability,
            reverse=True,
        )
        impression_items = [f.label for f in sorted_findings[:3]]  # Top 3
        impression_text = (
            f"AI-detected abnormalities: {', '.join(impression_items)}.\n"
            "Radiologist correlation recommended."
        )
    else:
        impression_text = (
            "Normal chest radiograph. No acute cardiopulmonary abnormality detected."
        )

    # Build recommendations
    if tsxr_output.findings:
        recommendations_text = (
            "1. Radiologist review of AI-detected findings is recommended.\n"
            "2. Clinical correlation with patient history and symptoms.\n"
            "3. Follow-up imaging as clinically indicated.\n\n"
            "NOTE: This is an automated AI-generated report. "
            "All findings require verification by a qualified radiologist."
        )
    else:
        recommendations_text = (
            "No specific follow-up required based on AI analysis.\n"
            "Continue routine care as clinically appropriate.\n\n"
            "NOTE: This is an automated AI-generated report. "
            "Clinical judgment should guide patient management."
        )

    return GeminiReport(
        findings=findings_text,
        impression=impression_text,
        recommendations=recommendations_text,
    )
