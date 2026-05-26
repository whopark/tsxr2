"""Tests for annotation module."""

import pytest
from PIL import Image

from tsxr2.annotation import (
    AnnotationConfig,
    ArrowStyle,
    FRACTURE_STYLES,
    annotate_rib_findings,
    calculate_arrow_offset,
    draw_fracture_arrow,
    get_style_for_status,
)
from tsxr2.schemas.rib_finding import CoordinatePoint, RibFinding


class TestArrowStyle:
    """Tests for ArrowStyle configuration."""

    def test_default_style(self):
        """Default style should have red color."""
        style = ArrowStyle()
        assert style.color == (255, 0, 0)
        assert style.line_width == 3

    def test_custom_style(self):
        """Custom style should override defaults."""
        style = ArrowStyle(color=(0, 255, 0), line_width=5)
        assert style.color == (0, 255, 0)
        assert style.line_width == 5


class TestFractureStyles:
    """Tests for predefined fracture styles."""

    def test_fractured_style_is_red(self):
        """Fractured status should use red color."""
        style = get_style_for_status("fractured")
        assert style.color == (255, 0, 0)

    def test_suspicious_style_is_orange(self):
        """Suspicious status should use orange color."""
        style = get_style_for_status("suspicious")
        assert style.color == (255, 165, 0)

    def test_intact_style_is_green(self):
        """Intact status should use green color."""
        style = get_style_for_status("intact")
        assert style.color == (0, 200, 0)

    def test_osteoporosis_style_is_purple(self):
        """Osteoporosis status should use purple/magenta color."""
        style = get_style_for_status("osteoporosis")
        assert style.color == (180, 0, 180)

    def test_unknown_status_returns_default(self):
        """Unknown status should return default style."""
        style = get_style_for_status("unknown")
        assert style.color == (255, 0, 0)  # Default red


class TestArrowOffsetCalculation:
    """Tests for arrow offset calculation."""

    def test_auto_offset_prefers_more_space(self):
        """Auto direction should prefer side with more space."""
        # Target on right side of image (400/512) - more space on LEFT (400 pixels)
        target = (400, 256)
        image_size = (512, 512)
        origin = calculate_arrow_offset(target, image_size, offset_distance=50)
        # Origin should be to the left of target (more space on left)
        assert origin[0] < target[0]

    def test_left_offset(self):
        """Left direction should offset to the left."""
        target = (300, 256)
        image_size = (512, 512)
        origin = calculate_arrow_offset(target, image_size, direction="left")
        assert origin[0] < target[0]
        assert origin[1] == target[1]

    def test_right_offset(self):
        """Right direction should offset to the right."""
        target = (200, 256)
        image_size = (512, 512)
        origin = calculate_arrow_offset(target, image_size, direction="right")
        assert origin[0] > target[0]
        assert origin[1] == target[1]

    def test_top_offset(self):
        """Top direction should offset upward."""
        target = (256, 300)
        image_size = (512, 512)
        origin = calculate_arrow_offset(target, image_size, direction="top")
        assert origin[0] == target[0]
        assert origin[1] < target[1]

    def test_bottom_offset(self):
        """Bottom direction should offset downward."""
        target = (256, 200)
        image_size = (512, 512)
        origin = calculate_arrow_offset(target, image_size, direction="bottom")
        assert origin[0] == target[0]
        assert origin[1] > target[1]


class TestDrawFractureArrow:
    """Tests for single arrow drawing."""

    @pytest.fixture
    def blank_image(self) -> Image.Image:
        """Create a blank 512x512 gray image."""
        return Image.new("RGB", (512, 512), color=(128, 128, 128))

    def test_returns_annotated_image(self, blank_image):
        """Should return both annotated image and annotation metadata."""
        annotated, annotation = draw_fracture_arrow(
            blank_image,
            target_point=(256, 256),
            label="L5 Fracture",
        )
        assert isinstance(annotated, Image.Image)
        assert annotation.label == "L5 Fracture"

    def test_does_not_modify_original(self, blank_image):
        """Original image should not be modified."""
        original_data = list(blank_image.getdata())
        draw_fracture_arrow(blank_image, (256, 256), "Test")
        assert list(blank_image.getdata()) == original_data

    def test_arrow_is_drawn(self, blank_image):
        """Arrow should add non-gray pixels to image."""
        annotated, _ = draw_fracture_arrow(
            blank_image,
            target_point=(256, 256),
            label="Test",
            style=ArrowStyle(color=(255, 0, 0)),
        )
        # Check that some red pixels exist
        pixels = list(annotated.getdata())
        red_pixels = [p for p in pixels if p[0] > 200 and p[1] < 100 and p[2] < 100]
        assert len(red_pixels) > 0

    def test_annotation_has_correct_coordinates(self, blank_image):
        """Annotation should have correct target and origin coordinates."""
        target = (256, 256)
        _, annotation = draw_fracture_arrow(
            blank_image,
            target_point=target,
            label="Test",
        )
        assert annotation.target_point.x == target[0]
        assert annotation.target_point.y == target[1]


class TestAnnotateRibFindings:
    """Tests for multiple rib annotation."""

    @pytest.fixture
    def blank_image(self) -> Image.Image:
        """Create a blank 512x512 gray image."""
        return Image.new("RGB", (512, 512), color=(128, 128, 128))

    @pytest.fixture
    def sample_findings(self) -> list[RibFinding]:
        """Create sample rib findings."""
        return [
            RibFinding(
                rib_id="L5",
                bbox=(280, 200, 350, 230),
                centroid=CoordinatePoint(x=315, y=215),
                detection_confidence=0.95,
                fracture_status="fractured",
                fracture_confidence=0.88,
            ),
            RibFinding(
                rib_id="R3",
                bbox=(150, 150, 220, 180),
                centroid=CoordinatePoint(x=185, y=165),
                detection_confidence=0.92,
                fracture_status="suspicious",
                fracture_confidence=0.65,
            ),
            RibFinding(
                rib_id="L2",
                bbox=(300, 100, 370, 130),
                centroid=CoordinatePoint(x=335, y=115),
                detection_confidence=0.98,
                fracture_status="intact",
                fracture_confidence=0.95,
            ),
        ]

    def test_annotates_fractures_by_default(self, blank_image, sample_findings):
        """By default, only fractured and suspicious ribs should be annotated."""
        config = AnnotationConfig(show_intact_ribs=False)
        annotated, annotations = annotate_rib_findings(
            blank_image, sample_findings, config
        )
        # Should have 2 annotations (L5 fractured, R3 suspicious)
        assert len(annotations) == 2
        labels = [a.associated_rib for a in annotations]
        assert "L5" in labels
        assert "R3" in labels
        assert "L2" not in labels  # Intact, should not be annotated

    def test_annotates_all_when_configured(self, blank_image, sample_findings):
        """Should annotate all ribs when show_intact_ribs is True."""
        config = AnnotationConfig(show_intact_ribs=True)
        annotated, annotations = annotate_rib_findings(
            blank_image, sample_findings, config
        )
        assert len(annotations) == 3

    def test_returns_annotated_image(self, blank_image, sample_findings):
        """Should return annotated PIL Image."""
        annotated, _ = annotate_rib_findings(blank_image, sample_findings)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == blank_image.size

    def test_left_ribs_arrow_from_right(self, blank_image, sample_findings):
        """Left ribs should have arrows coming from the right."""
        _, annotations = annotate_rib_findings(blank_image, sample_findings)
        l5_annotation = next(a for a in annotations if a.associated_rib == "L5")
        # Origin should be to the right of target (arrow from right)
        assert l5_annotation.origin_point.x > l5_annotation.target_point.x

    def test_right_ribs_arrow_from_left(self, blank_image, sample_findings):
        """Right ribs should have arrows coming from the left."""
        _, annotations = annotate_rib_findings(blank_image, sample_findings)
        r3_annotation = next(a for a in annotations if a.associated_rib == "R3")
        # Origin should be to the left of target (arrow from left)
        assert r3_annotation.origin_point.x < r3_annotation.target_point.x


class TestAnnotationConfig:
    """Tests for AnnotationConfig."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = AnnotationConfig()
        assert config.show_intact_ribs is False
        assert config.show_labels is True
        assert config.avoid_overlap is True

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = AnnotationConfig(
            show_intact_ribs=True,
            show_labels=False,
            arrow_offset=100,
        )
        assert config.show_intact_ribs is True
        assert config.show_labels is False
        assert config.arrow_offset == 100
