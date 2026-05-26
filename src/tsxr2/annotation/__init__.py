"""Annotation module for fracture visualization.

Provides arrow rendering and visual annotation utilities for
marking detected fractures on chest X-ray images.
"""

from tsxr2.annotation.arrow_renderer import (
    annotate_other_bone_findings,
    annotate_rib_findings,
    calculate_arrow_offset,
    draw_arrow,
    draw_fracture_arrow,
    draw_label,
)
from tsxr2.annotation.styles import (
    AnnotationConfig,
    ArrowStyle,
    FRACTURE_STYLES,
    OffsetDirection,
    SEVERITY_STYLES,
    get_style_for_severity,
    get_style_for_status,
)

__all__ = [
    # Arrow rendering
    "annotate_other_bone_findings",
    "annotate_rib_findings",
    "calculate_arrow_offset",
    "draw_arrow",
    "draw_fracture_arrow",
    "draw_label",
    # Styles
    "AnnotationConfig",
    "ArrowStyle",
    "FRACTURE_STYLES",
    "OffsetDirection",
    "SEVERITY_STYLES",
    "get_style_for_severity",
    "get_style_for_status",
]
