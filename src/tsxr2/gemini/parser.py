"""Response parser for Gemini clinical report output.

Parses markdown-formatted Gemini responses into structured
GeminiReport schema.
"""

import re

from tsxr2.schemas import GeminiReport


def parse_gemini_response(response_text: str) -> GeminiReport:
    """Parse Gemini markdown response into GeminiReport schema.

    Extracts Findings, Impression, and Recommendations sections
    from the markdown-formatted response.

    Args:
        response_text: Raw text response from Gemini.

    Returns:
        GeminiReport with parsed sections.
    """
    # Pattern to extract sections (case-insensitive headers)
    # Matches ## Findings, ## Impression, ## Recommendations
    section_pattern = r"##\s*(Findings|Impression|Recommendations?)\s*\n(.*?)(?=##\s*\w+|$)"

    sections = {
        "findings": "No findings provided.",
        "impression": "No impression provided.",
        "recommendations": "No recommendations provided.",
    }

    # Find all sections
    matches = re.findall(section_pattern, response_text, re.IGNORECASE | re.DOTALL)

    for header, content in matches:
        header_lower = header.lower()
        content_stripped = content.strip()

        if header_lower == "findings":
            sections["findings"] = content_stripped
        elif header_lower == "impression":
            sections["impression"] = content_stripped
        elif header_lower in ("recommendations", "recommendation"):
            sections["recommendations"] = content_stripped

    return GeminiReport(
        findings=sections["findings"],
        impression=sections["impression"],
        recommendations=sections["recommendations"],
    )
