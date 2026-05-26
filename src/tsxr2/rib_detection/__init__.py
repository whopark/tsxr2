"""Rib fracture detection module for systematic chest X-ray analysis.

This module provides:
- YOLOv8-based rib fracture detection (recommended)
- FCOS-based rib detection and localization (alternative)
- Fracture classification per individual rib
- Systematic scanning: L1→L10, R1→R10, then other bones
"""

from tsxr2.rib_detection.detector import (
    FCOSRibDetector,
    RibFractureHead,
    load_rib_detector,
)
from tsxr2.rib_detection.yolo_detector import (
    YOLOv8RibDetector,
    load_yolo_detector,
)
from tsxr2.rib_detection.inference import (
    OtherBoneDetectionResult,
    RibDetectionResult,
    format_other_bone_log_entry,
    run_full_rib_analysis,
    run_rib_detection,
    run_yolo_rib_detection,
    simulate_other_bone_detection,
    simulate_rib_detection,
    systematic_rib_scan,
)
from tsxr2.rib_detection.rib_labels import (
    FRACTURE_CLASSES,
    FRACTURE_TYPES,
    LEFT_RIBS,
    NUM_FRACTURE_CLASSES,
    NUM_RIB_CLASSES,
    OTHER_BONES,
    RIB_LABELS,
    RIGHT_RIBS,
    SCAN_ORDER,
    VISIBLE_LEFT_RIBS,
    VISIBLE_RIB_LABELS,
    VISIBLE_RIGHT_RIBS,
    format_scan_log_entry,
    get_rib_number,
    get_rib_side,
    is_valid_rib_id,
)

__all__ = [
    # YOLO Model (recommended)
    "YOLOv8RibDetector",
    "load_yolo_detector",
    # FCOS Model (alternative)
    "FCOSRibDetector",
    "RibFractureHead",
    "load_rib_detector",
    # Inference
    "OtherBoneDetectionResult",
    "RibDetectionResult",
    "format_other_bone_log_entry",
    "run_full_rib_analysis",
    "run_rib_detection",
    "run_yolo_rib_detection",
    "simulate_other_bone_detection",
    "simulate_rib_detection",
    "systematic_rib_scan",
    # Constants
    "FRACTURE_CLASSES",
    "FRACTURE_TYPES",
    "LEFT_RIBS",
    "NUM_FRACTURE_CLASSES",
    "NUM_RIB_CLASSES",
    "OTHER_BONES",
    "RIB_LABELS",
    "RIGHT_RIBS",
    "SCAN_ORDER",
    "VISIBLE_LEFT_RIBS",
    "VISIBLE_RIB_LABELS",
    "VISIBLE_RIGHT_RIBS",
    # Functions
    "format_scan_log_entry",
    "get_rib_number",
    "get_rib_side",
    "is_valid_rib_id",
]
