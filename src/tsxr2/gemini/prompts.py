"""Prompt templates for Gemini clinical report generation.

Provides structured prompts that format TSXr model output
for Gemini to generate clinical radiology reports.
"""

from tsxr2.schemas import TSXrOutput

# System prompt for clinical report generation
SYSTEM_PROMPT = """You are an expert radiologist assistant. Based on the AI-detected findings
from a chest X-ray analysis system, generate a structured clinical radiology report.

Your report must be:
1. Clinically accurate and professional
2. Structured with clear sections
3. Appropriately cautious (these are AI-detected findings requiring human verification)

Output format (use these exact section headers):
## Findings
[Describe each detected abnormality with location and severity]

## Impression
[Concise summary of the most clinically significant findings]

## Recommendations
[Suggest follow-up actions based on findings]
"""


def build_report_prompt(tsxr_output: TSXrOutput) -> str:
    """Build a prompt for Gemini from TSXr model output.

    Formats the structured TSXr output into a detailed prompt
    that Gemini can use to generate a clinical report.

    Args:
        tsxr_output: TSXrOutput schema from TSXr model inference.

    Returns:
        Formatted prompt string for Gemini.
    """
    # Build findings section
    findings_text = ""
    if tsxr_output.findings:
        for finding in tsxr_output.findings:
            findings_text += (
                f"- {finding.label}: probability={finding.probability:.2f}, "
                f"severity={finding.severity}, side={finding.side}\n"
            )
    else:
        findings_text = "- No significant abnormalities detected\n"

    # Build image info
    view = tsxr_output.image_info.view
    dimensions = tsxr_output.image_info.dimensions

    # Build quality assessment
    rotation = tsxr_output.quality_checks.rotation
    inspiration = tsxr_output.quality_checks.inspiration

    # Build global scores
    abnormality_score = tsxr_output.global_scores.abnormality_score
    confidence = tsxr_output.global_scores.confidence_index

    prompt = f"""{SYSTEM_PROMPT}

## AI Analysis Results

**Image Information:**
- View: {view}
- Dimensions: {dimensions[0]}x{dimensions[1]}

**Quality Assessment:**
- Rotation: {rotation}
- Inspiration: {inspiration}

**AI-Detected Findings:**
{findings_text}
**Global Scores:**
- Abnormality Score: {abnormality_score:.2f}
- Confidence Index: {confidence:.2f}

Based on these AI-detected findings, generate a clinical radiology report with Findings, Impression, and Recommendations sections.
"""
    return prompt
