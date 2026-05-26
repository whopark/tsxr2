"""Tests for rib detection module."""

import numpy as np
import pytest

from tsxr2.rib_detection import (
    FRACTURE_CLASSES,
    SCAN_ORDER,
    VISIBLE_LEFT_RIBS,
    VISIBLE_RIB_LABELS,
    VISIBLE_RIGHT_RIBS,
    format_scan_log_entry,
    get_rib_number,
    get_rib_side,
    is_valid_rib_id,
    run_full_rib_analysis,
    simulate_rib_detection,
)
from tsxr2.schemas import RibFinding


class TestRibLabels:
    """Tests for rib labeling constants and utilities."""

    def test_visible_left_ribs_order(self):
        """Left ribs should be L1 through L10."""
        assert VISIBLE_LEFT_RIBS == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"]

    def test_visible_right_ribs_order(self):
        """Right ribs should be R1 through R10."""
        assert VISIBLE_RIGHT_RIBS == ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]

    def test_scan_order_follows_clinical_protocol(self):
        """Scan order should be L1→L10, R1→R10, then other bones."""
        # First 10 should be left ribs
        assert SCAN_ORDER[:10] == VISIBLE_LEFT_RIBS
        # Next 10 should be right ribs
        assert SCAN_ORDER[10:20] == VISIBLE_RIGHT_RIBS
        # Last item should be other_bones
        assert SCAN_ORDER[-1] == "other_bones"

    def test_visible_rib_labels_has_20_entries(self):
        """Should have 20 visible ribs (L1-L10 + R1-R10)."""
        assert len(VISIBLE_RIB_LABELS) == 20

    def test_fracture_classes(self):
        """Fracture classes should be intact, fractured, suspicious."""
        assert "intact" in FRACTURE_CLASSES
        assert "fractured" in FRACTURE_CLASSES
        assert "suspicious" in FRACTURE_CLASSES


class TestRibIdValidation:
    """Tests for rib ID validation functions."""

    @pytest.mark.parametrize("rib_id,expected", [
        ("L1", True),
        ("L5", True),
        ("L10", True),
        ("L12", True),
        ("R1", True),
        ("R5", True),
        ("R10", True),
        ("R12", True),
    ])
    def test_valid_rib_ids(self, rib_id: str, expected: bool):
        """Valid rib IDs should return True."""
        assert is_valid_rib_id(rib_id) == expected

    @pytest.mark.parametrize("rib_id", [
        "X1",  # Invalid side
        "L0",  # Zero not valid
        "L13",  # Beyond 12
        "R13",
        "L",  # Missing number
        "5",  # Missing side
        "",  # Empty
        "LL1",  # Double letter
    ])
    def test_invalid_rib_ids(self, rib_id: str):
        """Invalid rib IDs should return False."""
        assert is_valid_rib_id(rib_id) is False

    def test_get_rib_side_left(self):
        """Left ribs should return 'left'."""
        assert get_rib_side("L1") == "left"
        assert get_rib_side("L10") == "left"

    def test_get_rib_side_right(self):
        """Right ribs should return 'right'."""
        assert get_rib_side("R1") == "right"
        assert get_rib_side("R10") == "right"

    def test_get_rib_side_invalid_raises(self):
        """Invalid rib ID should raise ValueError."""
        with pytest.raises(ValueError):
            get_rib_side("X5")

    def test_get_rib_number(self):
        """Should extract numeric part of rib ID."""
        assert get_rib_number("L1") == 1
        assert get_rib_number("R10") == 10
        assert get_rib_number("L12") == 12

    def test_get_rib_number_invalid_raises(self):
        """Invalid rib ID should raise ValueError."""
        with pytest.raises(ValueError):
            get_rib_number("L")


class TestScanLogFormatting:
    """Tests for scan log entry formatting."""

    def test_format_intact_rib(self):
        """Intact rib should show 'intact' status."""
        entry = format_scan_log_entry("L5", "intact", 0.95)
        assert "L5" in entry
        assert "intact" in entry
        assert "0.95" in entry

    def test_format_fractured_rib(self):
        """Fractured rib should show 'FRACTURE DETECTED'."""
        entry = format_scan_log_entry("R3", "fractured", 0.87)
        assert "R3" in entry
        assert "FRACTURE" in entry
        assert "0.87" in entry

    def test_format_suspicious_rib(self):
        """Suspicious rib should show 'SUSPICIOUS'."""
        entry = format_scan_log_entry("L8", "suspicious", 0.65)
        assert "L8" in entry
        assert "SUSPICIOUS" in entry


class TestRibFindingSchema:
    """Tests for RibFinding Pydantic schema."""

    def test_valid_rib_finding(self):
        """Valid RibFinding should be created successfully."""
        from tsxr2.schemas.rib_finding import CoordinatePoint

        finding = RibFinding(
            rib_id="L5",
            bbox=(100, 200, 150, 250),
            centroid=CoordinatePoint(x=125, y=225),
            detection_confidence=0.95,
            fracture_status="fractured",
            fracture_confidence=0.88,
        )
        assert finding.rib_id == "L5"
        assert finding.fracture_status == "fractured"

    def test_rib_id_pattern_validation(self):
        """Invalid rib ID pattern should raise validation error."""
        from pydantic import ValidationError
        from tsxr2.schemas.rib_finding import CoordinatePoint

        with pytest.raises(ValidationError):
            RibFinding(
                rib_id="X5",  # Invalid pattern
                bbox=(100, 200, 150, 250),
                centroid=CoordinatePoint(x=125, y=225),
                detection_confidence=0.95,
                fracture_status="intact",
                fracture_confidence=0.95,
            )

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        from pydantic import ValidationError
        from tsxr2.schemas.rib_finding import CoordinatePoint

        with pytest.raises(ValidationError):
            RibFinding(
                rib_id="L5",
                bbox=(100, 200, 150, 250),
                centroid=CoordinatePoint(x=125, y=225),
                detection_confidence=1.5,  # Invalid: > 1.0
                fracture_status="intact",
                fracture_confidence=0.95,
            )


class TestSimulatedDetection:
    """Tests for simulated rib detection."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample 512x512 RGB image."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    def test_simulate_returns_20_ribs(self, sample_image):
        """Simulation should detect all 20 visible ribs."""
        results = simulate_rib_detection(sample_image)
        assert len(results) == 20

    def test_simulate_includes_all_rib_ids(self, sample_image):
        """Simulation should include all visible rib IDs."""
        results = simulate_rib_detection(sample_image)
        detected_ids = {r.rib_id for r in results}
        expected_ids = set(VISIBLE_RIB_LABELS)
        assert detected_ids == expected_ids

    def test_simulate_has_valid_bboxes(self, sample_image):
        """All bounding boxes should be within image bounds."""
        results = simulate_rib_detection(sample_image)
        h, w = sample_image.shape[:2]

        for r in results:
            x1, y1, x2, y2 = r.bbox
            assert 0 <= x1 < w
            assert 0 <= x2 <= w
            assert 0 <= y1 < h
            assert 0 <= y2 <= h
            assert x1 < x2
            assert y1 < y2


class TestFullRibAnalysis:
    """Tests for complete rib analysis pipeline."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample 512x512 RGB image."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    def test_full_analysis_returns_rib_analysis_output(self, sample_image):
        """Full analysis should return RibAnalysisOutput."""
        from tsxr2.schemas.rib_finding import RibAnalysisOutput

        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
        )
        assert isinstance(result, RibAnalysisOutput)

    def test_full_analysis_has_scan_order_report(self, sample_image):
        """Analysis should include systematic scan report."""
        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
        )
        assert len(result.scan_order_report) > 0
        # First entry should be for L1
        assert result.scan_order_report[0].startswith("L1:")

    def test_full_analysis_detects_fractures(self, sample_image):
        """Simulation should detect some fractures for testing."""
        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
        )
        # Simulation is configured to have L5 and R7 as fractured
        assert result.total_fracture_count > 0
        assert len(result.fractures_detected) > 0

    def test_full_analysis_metadata(self, sample_image):
        """Analysis should include metadata with model version."""
        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
        )
        assert result.metadata.model_version == "rib-detector-v1.0"
        assert result.metadata.scan_duration_ms is not None
