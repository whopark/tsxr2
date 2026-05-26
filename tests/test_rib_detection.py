"""Tests for rib detection module."""

import numpy as np
import pytest

from tsxr2.rib_detection import (
    FRACTURE_CLASSES,
    SCAN_ORDER,
    VISIBLE_LEFT_RIBS,
    VISIBLE_RIB_LABELS,
    VISIBLE_RIGHT_RIBS,
    format_other_bone_log_entry,
    format_scan_log_entry,
    get_rib_number,
    get_rib_side,
    is_valid_rib_id,
    run_full_rib_analysis,
    simulate_other_bone_detection,
    simulate_rib_detection,
)
from tsxr2.schemas import OtherBoneFinding, RibFinding


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


class TestOtherBoneDetection:
    """Tests for clavicle and scapula detection."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample 512x512 RGB image."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    def test_simulate_other_bones_returns_7_findings(self, sample_image):
        """Simulation should detect 7 other bones (2 clavicles, 2 scapulae, 3 spine regions)."""
        results = simulate_other_bone_detection(sample_image)
        assert len(results) == 7

    def test_simulate_includes_clavicles(self, sample_image):
        """Simulation should include both clavicles."""
        results = simulate_other_bone_detection(sample_image)
        clavicles = [r for r in results if r.bone_name == "clavicle"]
        assert len(clavicles) == 2
        sides = {c.side for c in clavicles}
        assert sides == {"left", "right"}

    def test_simulate_includes_scapulae(self, sample_image):
        """Simulation should include both scapulae."""
        results = simulate_other_bone_detection(sample_image)
        scapulae = [r for r in results if r.bone_name == "scapula"]
        assert len(scapulae) == 2
        sides = {s.side for s in scapulae}
        assert sides == {"left", "right"}

    def test_simulate_has_valid_bboxes(self, sample_image):
        """All bounding boxes should be within image bounds."""
        results = simulate_other_bone_detection(sample_image)
        h, w = sample_image.shape[:2]

        for r in results:
            x1, y1, x2, y2 = r.bbox
            assert 0 <= x1 < w
            assert 0 <= x2 <= w
            assert 0 <= y1 < h
            assert 0 <= y2 <= h
            assert x1 < x2
            assert y1 < y2

    def test_simulate_includes_spine_regions(self, sample_image):
        """Simulation should include 3 thoracic spine regions."""
        results = simulate_other_bone_detection(sample_image)
        spine = [r for r in results if r.bone_name == "spine"]
        assert len(spine) == 3
        regions = {s.side for s in spine}
        assert regions == {"upper_thoracic", "mid_thoracic", "lower_thoracic"}

    def test_simulate_includes_fractures(self, sample_image):
        """Simulation should include fracture, suspicious, and osteoporosis findings."""
        results = simulate_other_bone_detection(sample_image)
        non_intact = [r for r in results if r.fracture_status != "intact"]
        # Simulation has: right clavicle fractured, right scapula suspicious,
        # mid_thoracic spine fractured, lower_thoracic spine osteoporosis
        assert len(non_intact) >= 4

    def test_simulate_includes_osteoporosis(self, sample_image):
        """Simulation should include osteoporosis finding in lower thoracic spine."""
        results = simulate_other_bone_detection(sample_image)
        osteoporosis = [r for r in results if r.fracture_status == "osteoporosis"]
        assert len(osteoporosis) == 1
        assert osteoporosis[0].bone_name == "spine"
        assert osteoporosis[0].side == "lower_thoracic"


class TestOtherBoneFindingSchema:
    """Tests for OtherBoneFinding Pydantic schema."""

    def test_valid_clavicle_finding(self):
        """Valid clavicle finding should be created successfully."""
        finding = OtherBoneFinding(
            bone_name="clavicle",
            side="left",
            bbox=(100, 50, 200, 80),
            fracture_confidence=0.85,
        )
        assert finding.bone_name == "clavicle"
        assert finding.side == "left"

    def test_valid_scapula_finding(self):
        """Valid scapula finding should be created successfully."""
        finding = OtherBoneFinding(
            bone_name="scapula",
            side="right",
            bbox=(50, 100, 150, 300),
            fracture_confidence=0.65,
        )
        assert finding.bone_name == "scapula"
        assert finding.side == "right"

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OtherBoneFinding(
                bone_name="clavicle",
                side="left",
                bbox=(100, 50, 200, 80),
                fracture_confidence=1.5,  # Invalid: > 1.0
            )

    def test_valid_spine_finding_with_region(self):
        """Spine finding with thoracic region should be valid."""
        finding = OtherBoneFinding(
            bone_name="spine",
            side="mid_thoracic",
            bbox=(200, 100, 280, 300),
            fracture_status="fractured",
            fracture_confidence=0.85,
        )
        assert finding.bone_name == "spine"
        assert finding.side == "mid_thoracic"
        assert finding.fracture_status == "fractured"

    def test_valid_spine_with_osteoporosis(self):
        """Spine finding with osteoporosis status should be valid."""
        finding = OtherBoneFinding(
            bone_name="spine",
            side="lower_thoracic",
            bbox=(200, 250, 280, 450),
            fracture_status="osteoporosis",
            fracture_confidence=0.72,
        )
        assert finding.fracture_status == "osteoporosis"
        assert finding.side == "lower_thoracic"

    def test_all_spine_regions_valid(self):
        """All spine region values should be accepted."""
        for region in ["upper_thoracic", "mid_thoracic", "lower_thoracic"]:
            finding = OtherBoneFinding(
                bone_name="spine",
                side=region,
                bbox=(200, 100, 280, 200),
                fracture_confidence=0.80,
            )
            assert finding.side == region


class TestOtherBoneLogFormatting:
    """Tests for other bone scan log formatting."""

    def test_format_intact_clavicle(self):
        """Intact clavicle should show 'intact' status."""
        entry = format_other_bone_log_entry("clavicle", "left", "intact", 0.92)
        assert "clavicle" in entry.lower()
        assert "left" in entry.lower()
        assert "intact" in entry.lower()

    def test_format_fractured_clavicle(self):
        """Fractured clavicle should show 'FRACTURE'."""
        entry = format_other_bone_log_entry("clavicle", "right", "fractured", 0.88)
        assert "clavicle" in entry.lower()
        assert "right" in entry.lower()
        assert "FRACTURE" in entry

    def test_format_suspicious_scapula(self):
        """Suspicious scapula should show 'SUSPICIOUS'."""
        entry = format_other_bone_log_entry("scapula", "right", "suspicious", 0.62)
        assert "scapula" in entry.lower()
        assert "right" in entry.lower()
        assert "SUSPICIOUS" in entry

    def test_format_spine_region(self):
        """Spine with thoracic region should format correctly."""
        entry = format_other_bone_log_entry("spine", "mid_thoracic", "fractured", 0.85)
        assert "MID THORACIC SPINE" in entry
        assert "FRACTURE" in entry

    def test_format_spine_osteoporosis(self):
        """Spine with osteoporosis should show 'OSTEOPOROSIS'."""
        entry = format_other_bone_log_entry("spine", "lower_thoracic", "osteoporosis", 0.72)
        assert "LOWER THORACIC SPINE" in entry
        assert "OSTEOPOROSIS" in entry


class TestFullAnalysisWithOtherBones:
    """Tests for complete analysis including clavicle and scapula."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample 512x512 RGB image."""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    def test_full_analysis_includes_other_bone_findings(self, sample_image):
        """Full analysis should include other bone findings."""
        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
            include_other_bones=True,
        )
        assert result.other_bone_findings is not None
        assert len(result.other_bone_findings) == 7  # 2 clavicles + 2 scapulae + 3 spine

    def test_full_analysis_scan_order_includes_other_bones(self, sample_image):
        """Scan order report should include other bone entries."""
        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
            include_other_bones=True,
        )
        # Find entries containing clavicle, scapula, or spine
        other_bone_entries = [
            e for e in result.scan_order_report
            if "clavicle" in e.lower() or "scapula" in e.lower() or "spine" in e.lower()
        ]
        assert len(other_bone_entries) >= 7  # At least 7 other bone entries

    def test_full_analysis_without_other_bones(self, sample_image):
        """Analysis without other bones should have no other_bone_findings."""
        result = run_full_rib_analysis(
            model=None,
            image=sample_image,
            use_simulation=True,
            include_other_bones=False,
        )
        assert result.other_bone_findings is None or len(result.other_bone_findings) == 0
