"""Rib identification labels and systematic scanning order.

Defines the standard labeling convention for chest X-ray rib analysis
following the clinical protocol: L1→L10, R1→R10, then other bones.
"""

from typing import Literal

# All rib labels (L1-L12 left, R1-R12 right)
# Note: L11/L12 and R11/R12 are floating ribs, often not visible on CXR
LEFT_RIBS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12"]
RIGHT_RIBS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"]

# Combined rib labels (24 total)
RIB_LABELS = LEFT_RIBS + RIGHT_RIBS

# Typically visible ribs on standard chest X-ray (PA/AP view)
VISIBLE_LEFT_RIBS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"]
VISIBLE_RIGHT_RIBS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
VISIBLE_RIB_LABELS = VISIBLE_LEFT_RIBS + VISIBLE_RIGHT_RIBS

# Systematic scan order per clinical protocol (CLAUDE.md requirement)
# Order: L1→L10, R1→R10, then other bones
SCAN_ORDER = VISIBLE_LEFT_RIBS + VISIBLE_RIGHT_RIBS + ["other_bones"]

# Fracture classification states
FractureStatus = Literal["intact", "fractured", "suspicious"]

FRACTURE_CLASSES = ["intact", "fractured", "suspicious"]

# Fracture types for detailed classification
FRACTURE_TYPES = [
    "simple",  # Single break, non-displaced
    "displaced",  # Bone fragments separated
    "comminuted",  # Multiple fragments
    "greenstick",  # Incomplete fracture (rare in adults)
    "healing",  # Evidence of prior fracture with callus
    "pathological",  # Fracture through diseased bone
]

# Other bones to check after ribs
OTHER_BONES = [
    "clavicle_left",
    "clavicle_right",
    "scapula_left",
    "scapula_right",
    "sternum",
    "thoracic_spine",
]

# Number of classes for detection model
NUM_RIB_CLASSES = len(VISIBLE_RIB_LABELS)  # 20 (L1-L10, R1-R10)
NUM_FRACTURE_CLASSES = len(FRACTURE_CLASSES)  # 3 (intact, fractured, suspicious)


def get_rib_side(rib_id: str) -> Literal["left", "right"]:
    """Get the anatomical side from rib ID.

    Args:
        rib_id: Rib identifier (e.g., "L5", "R3")

    Returns:
        "left" or "right"

    Raises:
        ValueError: If rib_id format is invalid
    """
    if rib_id.startswith("L"):
        return "left"
    elif rib_id.startswith("R"):
        return "right"
    else:
        raise ValueError(f"Invalid rib ID format: {rib_id}")


def get_rib_number(rib_id: str) -> int:
    """Get the rib number from rib ID.

    Args:
        rib_id: Rib identifier (e.g., "L5", "R3")

    Returns:
        Integer rib number (1-12)

    Raises:
        ValueError: If rib_id format is invalid
    """
    try:
        return int(rib_id[1:])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid rib ID format: {rib_id}") from e


def is_valid_rib_id(rib_id: str) -> bool:
    """Check if rib ID is valid.

    Args:
        rib_id: Rib identifier to validate

    Returns:
        True if valid, False otherwise
    """
    if not rib_id or len(rib_id) < 2:
        return False
    if rib_id[0] not in ("L", "R"):
        return False
    try:
        num = int(rib_id[1:])
        return 1 <= num <= 12
    except ValueError:
        return False


def format_scan_log_entry(rib_id: str, status: str, confidence: float) -> str:
    """Format a scan log entry for the systematic scan report.

    Args:
        rib_id: Rib identifier (e.g., "L5")
        status: Fracture status ("intact", "fractured", "suspicious")
        confidence: Detection confidence (0.0-1.0)

    Returns:
        Formatted log entry string
    """
    status_marker = ""
    if status == "fractured":
        status_marker = "FRACTURE DETECTED"
    elif status == "suspicious":
        status_marker = "SUSPICIOUS"
    else:
        status_marker = "intact"

    return f"{rib_id}: {status_marker} (conf: {confidence:.2f})"
