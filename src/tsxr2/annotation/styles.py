"""Visual styles for fracture annotations.

Defines colors, sizes, and other visual parameters for arrow annotations
on chest X-ray images.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ArrowStyle:
    """Configuration for arrow annotation rendering.

    Attributes:
        color: RGB tuple for arrow color.
        line_width: Width of the arrow line in pixels.
        head_length: Length of the arrowhead in pixels.
        head_angle: Angle of arrowhead wings in degrees.
        label_font_size: Font size for annotation labels.
        label_color: RGB tuple for label text color.
        label_bg_color: RGB tuple for label background (None for transparent).
        label_padding: Padding around label text in pixels.
    """

    color: tuple[int, int, int] = (255, 0, 0)  # Red
    line_width: int = 3
    head_length: int = 15
    head_angle: float = 30.0
    label_font_size: int = 12
    label_color: tuple[int, int, int] = (255, 255, 255)  # White
    label_bg_color: tuple[int, int, int] | None = (0, 0, 0)  # Black background
    label_padding: int = 4


# Predefined styles for different fracture statuses
FRACTURE_STYLES: dict[str, ArrowStyle] = {
    "fractured": ArrowStyle(
        color=(255, 0, 0),  # Red - definite fracture
        line_width=4,
        head_length=18,
        label_bg_color=(180, 0, 0),
    ),
    "suspicious": ArrowStyle(
        color=(255, 165, 0),  # Orange - suspicious
        line_width=3,
        head_length=15,
        label_bg_color=(180, 100, 0),
    ),
    "osteoporosis": ArrowStyle(
        color=(180, 0, 180),  # Purple/Magenta - osteoporotic changes
        line_width=3,
        head_length=15,
        label_bg_color=(120, 0, 120),
    ),
    "intact": ArrowStyle(
        color=(0, 200, 0),  # Green - normal (optional display)
        line_width=2,
        head_length=12,
        label_bg_color=(0, 120, 0),
    ),
}

# Severity-based styles (alternative)
SEVERITY_STYLES: dict[str, ArrowStyle] = {
    "severe": ArrowStyle(
        color=(255, 0, 0),
        line_width=5,
        head_length=20,
    ),
    "moderate": ArrowStyle(
        color=(255, 140, 0),
        line_width=4,
        head_length=16,
    ),
    "mild": ArrowStyle(
        color=(255, 200, 0),
        line_width=3,
        head_length=14,
    ),
}

# Arrow offset directions
OffsetDirection = Literal["left", "right", "top", "bottom", "auto"]


@dataclass
class AnnotationConfig:
    """Global configuration for annotation rendering.

    Attributes:
        default_style: Default arrow style to use.
        show_intact_ribs: Whether to show annotations for intact ribs.
        show_labels: Whether to show text labels on arrows.
        arrow_offset: Distance from target to arrow origin in pixels.
        min_arrow_length: Minimum arrow length in pixels.
        avoid_overlap: Whether to adjust arrows to avoid overlapping.
        overlap_threshold: Distance threshold for overlap detection.
    """

    default_style: ArrowStyle = field(default_factory=ArrowStyle)
    show_intact_ribs: bool = False
    show_labels: bool = True
    arrow_offset: int = 60
    min_arrow_length: int = 30
    avoid_overlap: bool = True
    overlap_threshold: int = 40


def get_style_for_status(status: str) -> ArrowStyle:
    """Get the appropriate arrow style for a fracture status.

    Args:
        status: Fracture status ("intact", "fractured", "suspicious", "osteoporosis").

    Returns:
        ArrowStyle for the given status.
    """
    return FRACTURE_STYLES.get(status, ArrowStyle())


def get_style_for_severity(severity: str) -> ArrowStyle:
    """Get the appropriate arrow style for a severity level.

    Args:
        severity: Severity level ("mild", "moderate", "severe").

    Returns:
        ArrowStyle for the given severity.
    """
    return SEVERITY_STYLES.get(severity, ArrowStyle())
