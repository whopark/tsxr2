"""Arrow annotation rendering for fracture visualization.

Uses Pillow (PIL) to draw arrows and labels on chest X-ray images
pointing to detected fracture locations.
"""

import math
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from tsxr2.annotation.styles import (
    AnnotationConfig,
    ArrowStyle,
    OffsetDirection,
    get_style_for_status,
)
from tsxr2.schemas.rib_finding import (
    ArrowAnnotation,
    CoordinatePoint,
    OtherBoneFinding,
    RibFinding,
)


def calculate_arrow_offset(
    target: tuple[int, int],
    image_size: tuple[int, int],
    offset_distance: int = 60,
    direction: OffsetDirection = "auto",
) -> tuple[int, int]:
    """Calculate the origin point for an arrow pointing to the target.

    Args:
        target: Target point (x, y) the arrow points to.
        image_size: Image dimensions (width, height).
        offset_distance: Distance from target to arrow origin.
        direction: Preferred direction ("auto" chooses best).

    Returns:
        Tuple (x, y) for the arrow origin point.
    """
    tx, ty = target
    w, h = image_size

    if direction == "auto":
        # Choose direction based on position in image
        # Arrow comes from the side with more space
        left_space = tx
        right_space = w - tx
        top_space = ty
        bottom_space = h - ty

        # Prefer horizontal arrows for rib annotations
        if left_space > right_space:
            direction = "left"
        else:
            direction = "right"

    # Calculate origin based on direction
    if direction == "left":
        return (tx - offset_distance, ty)
    elif direction == "right":
        return (tx + offset_distance, ty)
    elif direction == "top":
        return (tx, ty - offset_distance)
    elif direction == "bottom":
        return (tx, ty + offset_distance)
    else:
        return (tx - offset_distance, ty)  # Default to left


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    style: ArrowStyle,
) -> None:
    """Draw an arrow from start to end point.

    Args:
        draw: PIL ImageDraw object.
        start: Arrow start point (x, y).
        end: Arrow end point (x, y) - where the arrowhead points.
        style: ArrowStyle configuration.
    """
    # Draw the main line
    draw.line([start, end], fill=style.color, width=style.line_width)

    # Calculate arrowhead
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_angle_rad = math.radians(style.head_angle)
    head_length = style.head_length

    # Arrowhead points
    left_x = end[0] - head_length * math.cos(angle - head_angle_rad)
    left_y = end[1] - head_length * math.sin(angle - head_angle_rad)
    right_x = end[0] - head_length * math.cos(angle + head_angle_rad)
    right_y = end[1] - head_length * math.sin(angle + head_angle_rad)

    # Draw arrowhead as filled triangle
    draw.polygon(
        [(end[0], end[1]), (int(left_x), int(left_y)), (int(right_x), int(right_y))],
        fill=style.color,
    )


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    style: ArrowStyle,
    anchor: str = "mm",
) -> None:
    """Draw a label at the specified position.

    Args:
        draw: PIL ImageDraw object.
        position: Label center position (x, y).
        text: Label text to display.
        style: ArrowStyle configuration.
        anchor: Text anchor point (default "mm" = middle-middle).
    """
    # Try to use a TrueType font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", style.label_font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font, anchor=anchor)

    # Draw background if specified
    if style.label_bg_color is not None:
        padding = style.label_padding
        draw.rectangle(
            [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ],
            fill=style.label_bg_color,
        )

    # Draw text
    draw.text(position, text, fill=style.label_color, font=font, anchor=anchor)


def draw_fracture_arrow(
    image: Image.Image,
    target_point: tuple[int, int],
    label: str,
    style: ArrowStyle | None = None,
    offset_direction: OffsetDirection = "auto",
    offset_distance: int = 60,
) -> tuple[Image.Image, ArrowAnnotation]:
    """Draw a single fracture arrow on the image.

    Args:
        image: PIL Image to annotate.
        target_point: Point the arrow points to (x, y).
        label: Text label for the annotation.
        style: ArrowStyle to use (default: red arrow).
        offset_direction: Direction for arrow origin.
        offset_distance: Distance from target to origin.

    Returns:
        Tuple of (annotated image copy, ArrowAnnotation metadata).
    """
    if style is None:
        style = ArrowStyle()

    # Create a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Calculate arrow origin
    origin = calculate_arrow_offset(
        target_point,
        image.size,
        offset_distance,
        offset_direction,
    )

    # Draw the arrow
    draw_arrow(draw, origin, target_point, style)

    # Draw label near the origin
    label_offset_x = -20 if origin[0] < target_point[0] else 20
    label_pos = (origin[0] + label_offset_x, origin[1])
    draw_label(draw, label_pos, label, style)

    # Create annotation metadata
    annotation = ArrowAnnotation(
        target_point=CoordinatePoint(x=target_point[0], y=target_point[1]),
        origin_point=CoordinatePoint(x=origin[0], y=origin[1]),
        label=label,
        color=style.color,
        associated_rib=label.split()[0] if " " in label else label,
    )

    return annotated, annotation


def annotate_rib_findings(
    image: Image.Image,
    findings: Sequence[RibFinding],
    config: AnnotationConfig | None = None,
) -> tuple[Image.Image, list[ArrowAnnotation]]:
    """Annotate multiple rib findings with arrows.

    Args:
        image: PIL Image to annotate.
        findings: List of RibFinding to annotate.
        config: AnnotationConfig for rendering options.

    Returns:
        Tuple of (annotated image, list of ArrowAnnotation metadata).
    """
    if config is None:
        config = AnnotationConfig()

    # Create a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    annotations: list[ArrowAnnotation] = []

    # Track arrow positions to avoid overlap
    used_positions: list[tuple[int, int]] = []

    for finding in findings:
        # Skip intact ribs unless configured to show them
        if finding.fracture_status == "intact" and not config.show_intact_ribs:
            continue

        # Get style for this fracture status
        style = get_style_for_status(finding.fracture_status)

        # Target is the centroid of the detected rib
        target = (finding.centroid.x, finding.centroid.y)

        # Determine arrow direction based on rib side
        if finding.rib_id.startswith("L"):
            # Left ribs (patient's left = image right) - arrow from right
            direction: OffsetDirection = "right"
        else:
            # Right ribs (patient's right = image left) - arrow from left
            direction = "left"

        # Calculate origin with overlap avoidance
        origin = calculate_arrow_offset(
            target,
            image.size,
            config.arrow_offset,
            direction,
        )

        # Adjust if overlapping with existing arrows
        if config.avoid_overlap:
            origin = _adjust_for_overlap(
                origin, used_positions, config.overlap_threshold
            )

        used_positions.append(origin)

        # Draw arrow
        draw_arrow(draw, origin, target, style)

        # Draw label if enabled
        label = f"{finding.rib_id} {finding.fracture_status.upper()}"
        if config.show_labels:
            label_offset_x = -30 if origin[0] < target[0] else 30
            label_pos = (origin[0] + label_offset_x, origin[1])
            draw_label(draw, label_pos, label, style)

        # Create annotation metadata
        annotation = ArrowAnnotation(
            target_point=CoordinatePoint(x=target[0], y=target[1]),
            origin_point=CoordinatePoint(x=origin[0], y=origin[1]),
            label=label,
            color=style.color,
            associated_rib=finding.rib_id,
        )
        annotations.append(annotation)

        # Also update the finding's annotation field
        finding.annotation = annotation

    return annotated, annotations


def _adjust_for_overlap(
    position: tuple[int, int],
    used_positions: list[tuple[int, int]],
    threshold: int,
) -> tuple[int, int]:
    """Adjust position to avoid overlapping with existing arrows.

    Args:
        position: Proposed position (x, y).
        used_positions: List of already used positions.
        threshold: Minimum distance between positions.

    Returns:
        Adjusted position that doesn't overlap.
    """
    x, y = position

    for used_x, used_y in used_positions:
        distance = math.sqrt((x - used_x) ** 2 + (y - used_y) ** 2)
        if distance < threshold:
            # Shift vertically to avoid overlap
            y += threshold

    return (x, y)


def annotate_other_bone_findings(
    image: Image.Image,
    findings: Sequence[OtherBoneFinding],
    config: AnnotationConfig | None = None,
    fracture_status_map: dict[str, str] | None = None,
) -> tuple[Image.Image, list[ArrowAnnotation]]:
    """Annotate clavicle, scapula, and other bone findings with arrows.

    Args:
        image: PIL Image to annotate.
        findings: List of OtherBoneFinding to annotate.
        config: AnnotationConfig for rendering options.
        fracture_status_map: Map of bone identifier to fracture status.
            Keys should be like "clavicle_left", "scapula_right".
            If not provided, uses confidence thresholds.

    Returns:
        Tuple of (annotated image, list of ArrowAnnotation metadata).
    """
    if config is None:
        config = AnnotationConfig()

    # Create a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    annotations: list[ArrowAnnotation] = []

    # Track arrow positions to avoid overlap
    used_positions: list[tuple[int, int]] = []

    for finding in findings:
        # Determine fracture status from map or confidence
        bone_id = f"{finding.bone_name}_{finding.side}"
        if fracture_status_map and bone_id in fracture_status_map:
            status = fracture_status_map[bone_id]
        else:
            # Infer status from confidence
            if finding.fracture_confidence >= 0.75:
                status = "fractured"
            elif finding.fracture_confidence >= 0.5:
                status = "suspicious"
            else:
                status = "intact"

        # Skip intact bones unless configured to show them
        if status == "intact" and not config.show_intact_ribs:
            continue

        # Get style for this fracture status
        style = get_style_for_status(status)

        # Calculate centroid from bounding box
        cx = (finding.bbox[0] + finding.bbox[2]) // 2
        cy = (finding.bbox[1] + finding.bbox[3]) // 2
        target = (cx, cy)

        # Determine arrow direction based on bone side
        # Clavicles are at top, so arrows come from above/below
        # Scapulae are lateral, so arrows come from sides
        if finding.bone_name == "clavicle":
            direction: OffsetDirection = "bottom"  # Arrow from below
        elif finding.side == "left":
            direction = "right"  # Arrow from right for left-side bones
        else:
            direction = "left"  # Arrow from left for right-side bones

        # Calculate origin
        origin = calculate_arrow_offset(
            target,
            image.size,
            config.arrow_offset,
            direction,
        )

        # Adjust if overlapping with existing arrows
        if config.avoid_overlap:
            origin = _adjust_for_overlap(
                origin, used_positions, config.overlap_threshold
            )

        used_positions.append(origin)

        # Draw arrow
        draw_arrow(draw, origin, target, style)

        # Draw label if enabled
        label = f"{finding.side.upper()} {finding.bone_name.upper()} {status.upper()}"
        if config.show_labels:
            label_offset_x = -30 if origin[0] < target[0] else 30
            label_offset_y = -15 if direction == "bottom" else 0
            label_pos = (origin[0] + label_offset_x, origin[1] + label_offset_y)
            draw_label(draw, label_pos, label, style)

        # Create annotation metadata
        annotation = ArrowAnnotation(
            target_point=CoordinatePoint(x=target[0], y=target[1]),
            origin_point=CoordinatePoint(x=origin[0], y=origin[1]),
            label=label,
            color=style.color,
            associated_rib=bone_id,  # Use bone_id as identifier
        )
        annotations.append(annotation)

        # Update the finding's annotation field
        finding.annotation = annotation

    return annotated, annotations
