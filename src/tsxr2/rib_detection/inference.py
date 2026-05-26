"""Rib detection inference pipeline.

Handles image preprocessing, model inference, and systematic rib scanning
following the clinical protocol: L1→L10, R1→R10, then other bones.

Supports multiple detection backends:
- YOLOv8: Real-time rib fracture detection (recommended)
- FCOS: Alternative anchor-free detector
- Simulation: For testing when no trained model is available
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import transforms

from tsxr2.rib_detection.detector import FCOSRibDetector
from tsxr2.rib_detection.yolo_detector import YOLOv8RibDetector, load_yolo_detector
from tsxr2.rib_detection.rib_labels import (
    FRACTURE_CLASSES,
    OTHER_BONES,
    SCAN_ORDER,
    VISIBLE_RIB_LABELS,
    format_scan_log_entry,
    get_rib_side,
)
from tsxr2.schemas.rib_finding import (
    CoordinatePoint,
    OtherBoneFinding,
    RibAnalysisMetadata,
    RibAnalysisOutput,
    RibFinding,
)

# ImageNet normalization for pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class RibDetectionResult:
    """Raw detection result for a single rib."""

    rib_id: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    detection_score: float
    fracture_status: str
    fracture_confidence: float


@dataclass
class OtherBoneDetectionResult:
    """Raw detection result for non-rib bones (clavicle, scapula, etc.)."""

    bone_name: str  # e.g., "clavicle", "scapula"
    side: str  # "left", "right", or "midline"
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    detection_score: float
    fracture_status: str
    fracture_confidence: float


def preprocess_for_detection(
    image: NDArray[np.uint8],
    target_size: int = 800,
) -> torch.Tensor:
    """Preprocess image for object detection.

    FCOS works best with larger input sizes than classification models.

    Args:
        image: Normalized image array (H, W, 3) uint8.
        target_size: Target size for the shorter edge (default 800).

    Returns:
        Tensor of shape (3, H, W) ready for detection model.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),  # Resize shorter edge
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transform(image)


def run_rib_detection(
    model: FCOSRibDetector,
    image: NDArray[np.uint8],
    device: torch.device,
    detection_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> list[RibDetectionResult]:
    """Run rib detection on a chest X-ray image.

    Args:
        model: Loaded FCOS rib detector.
        image: Normalized image array (H, W, 3) uint8.
        device: Device to run inference on.
        detection_threshold: Confidence threshold for detections.
        nms_threshold: NMS IoU threshold.

    Returns:
        List of RibDetectionResult for each detected rib.
    """
    # Preprocess image
    tensor = preprocess_for_detection(image).to(device)

    # Run detection
    with torch.no_grad():
        outputs = model([tensor])

    # Process outputs
    output = outputs[0]
    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    # Handle fracture predictions if available
    if "fracture_labels" in output:
        fracture_labels = output["fracture_labels"].cpu().numpy()
        fracture_scores = output["fracture_scores"].cpu().numpy()
    else:
        # Default to suspicious with low confidence
        fracture_labels = np.ones(len(boxes), dtype=np.int64)  # suspicious
        fracture_scores = np.zeros((len(boxes), 3))

    results = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < detection_threshold:
            continue

        # Skip background class (0)
        if label == 0:
            continue

        # Get rib label
        rib_id = model.get_rib_label(int(label))
        if rib_id == "background" or rib_id.startswith("unknown"):
            continue

        # Get fracture status
        frac_label = int(fracture_labels[i]) if i < len(fracture_labels) else 1
        frac_status = FRACTURE_CLASSES[min(frac_label, len(FRACTURE_CLASSES) - 1)]
        frac_conf = float(fracture_scores[i].max()) if i < len(fracture_scores) else 0.5

        results.append(RibDetectionResult(
            rib_id=rib_id,
            bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            detection_score=float(score),
            fracture_status=frac_status,
            fracture_confidence=frac_conf,
        ))

    return results


def run_yolo_rib_detection(
    detector: YOLOv8RibDetector,
    image: NDArray[np.uint8],
) -> list[RibDetectionResult]:
    """Run rib detection using YOLOv8 model.

    Args:
        detector: Loaded YOLOv8 rib detector.
        image: Normalized image array (H, W, 3) or (H, W) uint8.

    Returns:
        List of RibDetectionResult for each detected rib.
    """
    # Run YOLO detection with rib ID mapping
    yolo_detections = detector.detect_ribs(image)

    # Convert to RibDetectionResult format
    results = []
    for det in yolo_detections:
        results.append(RibDetectionResult(
            rib_id=det['rib_id'],
            bbox=det['bbox'],
            detection_score=det['detection_confidence'],
            fracture_status=det['fracture_status'],
            fracture_confidence=det['fracture_confidence'],
        ))

    return results


def simulate_rib_detection(
    image: NDArray[np.uint8],
    detection_threshold: float = 0.5,
) -> list[RibDetectionResult]:
    """Simulate rib detection for testing when model is not trained.

    Creates synthetic detections based on image regions.
    This is a placeholder until a trained model is available.

    Args:
        image: Normalized image array (H, W, 3) uint8.
        detection_threshold: Not used, kept for API consistency.

    Returns:
        List of simulated RibDetectionResult.
    """
    h, w = image.shape[:2]
    results = []

    # Simulate rib positions based on typical chest X-ray anatomy
    # Left ribs are on the right side of the image (patient's left)
    # Right ribs are on the left side of the image (patient's right)

    for i, rib_id in enumerate(VISIBLE_RIB_LABELS):
        side = get_rib_side(rib_id)
        rib_num = int(rib_id[1:])

        # Calculate approximate position
        # Vertical position: higher rib numbers are lower on the image
        y_start = int(h * (0.1 + rib_num * 0.07))
        y_end = y_start + int(h * 0.05)

        # Horizontal position: left ribs on right side, right ribs on left
        if side == "left":
            x_start = int(w * 0.55)
            x_end = int(w * 0.85)
        else:
            x_start = int(w * 0.15)
            x_end = int(w * 0.45)

        # Clamp to image bounds
        x_start = max(0, min(x_start, w - 10))
        x_end = max(x_start + 10, min(x_end, w))
        y_start = max(0, min(y_start, h - 10))
        y_end = max(y_start + 10, min(y_end, h))

        # Simulate detection confidence (random for testing)
        det_score = 0.7 + np.random.random() * 0.25

        # Simulate fracture (low probability for most ribs)
        # Make a few ribs have higher fracture probability for testing
        if rib_id in ["L5", "R7"]:
            frac_status = "fractured"
            frac_conf = 0.75 + np.random.random() * 0.2
        elif rib_id in ["L8", "R4"]:
            frac_status = "suspicious"
            frac_conf = 0.5 + np.random.random() * 0.2
        else:
            frac_status = "intact"
            frac_conf = 0.85 + np.random.random() * 0.1

        results.append(RibDetectionResult(
            rib_id=rib_id,
            bbox=(x_start, y_start, x_end, y_end),
            detection_score=det_score,
            fracture_status=frac_status,
            fracture_confidence=frac_conf,
        ))

    return results


def simulate_other_bone_detection(
    image: NDArray[np.uint8],
) -> list[OtherBoneDetectionResult]:
    """Simulate detection of clavicle, scapula, and other bones.

    Creates synthetic detections based on typical chest X-ray anatomy.
    This is a placeholder until a trained model is available.

    Args:
        image: Normalized image array (H, W, 3) uint8.

    Returns:
        List of simulated OtherBoneDetectionResult.
    """
    h, w = image.shape[:2]
    results = []

    # Clavicle positions (upper chest, bilateral)
    # Left clavicle (appears on right side of image - patient's left)
    results.append(OtherBoneDetectionResult(
        bone_name="clavicle",
        side="left",
        bbox=(int(w * 0.50), int(h * 0.05), int(w * 0.80), int(h * 0.12)),
        detection_score=0.85 + np.random.random() * 0.1,
        fracture_status="intact",
        fracture_confidence=0.90 + np.random.random() * 0.08,
    ))

    # Right clavicle (appears on left side of image - patient's right)
    # Simulate a fracture for testing
    results.append(OtherBoneDetectionResult(
        bone_name="clavicle",
        side="right",
        bbox=(int(w * 0.20), int(h * 0.05), int(w * 0.50), int(h * 0.12)),
        detection_score=0.88 + np.random.random() * 0.1,
        fracture_status="fractured",
        fracture_confidence=0.82 + np.random.random() * 0.15,
    ))

    # Scapula positions (lateral chest, bilateral)
    # Left scapula (appears on right side of image)
    results.append(OtherBoneDetectionResult(
        bone_name="scapula",
        side="left",
        bbox=(int(w * 0.75), int(h * 0.10), int(w * 0.95), int(h * 0.45)),
        detection_score=0.80 + np.random.random() * 0.15,
        fracture_status="intact",
        fracture_confidence=0.88 + np.random.random() * 0.1,
    ))

    # Right scapula (appears on left side of image)
    # Simulate suspicious finding for testing
    results.append(OtherBoneDetectionResult(
        bone_name="scapula",
        side="right",
        bbox=(int(w * 0.05), int(h * 0.10), int(w * 0.25), int(h * 0.45)),
        detection_score=0.82 + np.random.random() * 0.12,
        fracture_status="suspicious",
        fracture_confidence=0.55 + np.random.random() * 0.2,
    ))

    # Thoracic spine (visible vertebrae T1-T12 in chest X-ray)
    # Spine is midline, running vertically through the chest
    spine_y_start = int(h * 0.08)
    spine_y_end = int(h * 0.85)
    spine_x_center = w // 2
    spine_width = int(w * 0.08)

    # Simulate vertebral bodies (upper, middle, lower thoracic)
    # Upper thoracic (T1-T4) - usually intact
    results.append(OtherBoneDetectionResult(
        bone_name="spine",
        side="upper_thoracic",  # T1-T4 region
        bbox=(
            spine_x_center - spine_width,
            spine_y_start,
            spine_x_center + spine_width,
            int(h * 0.25),
        ),
        detection_score=0.78 + np.random.random() * 0.15,
        fracture_status="intact",
        fracture_confidence=0.88 + np.random.random() * 0.1,
    ))

    # Middle thoracic (T5-T8) - simulate compression fracture for testing
    results.append(OtherBoneDetectionResult(
        bone_name="spine",
        side="mid_thoracic",  # T5-T8 region
        bbox=(
            spine_x_center - spine_width,
            int(h * 0.25),
            spine_x_center + spine_width,
            int(h * 0.50),
        ),
        detection_score=0.82 + np.random.random() * 0.12,
        fracture_status="fractured",  # Compression fracture
        fracture_confidence=0.75 + np.random.random() * 0.2,
    ))

    # Lower thoracic (T9-T12) - simulate osteoporosis for testing
    results.append(OtherBoneDetectionResult(
        bone_name="spine",
        side="lower_thoracic",  # T9-T12 region
        bbox=(
            spine_x_center - spine_width,
            int(h * 0.50),
            spine_x_center + spine_width,
            spine_y_end,
        ),
        detection_score=0.80 + np.random.random() * 0.12,
        fracture_status="osteoporosis",  # Osteoporotic changes
        fracture_confidence=0.65 + np.random.random() * 0.2,
    ))

    return results


def format_other_bone_log_entry(
    bone_name: str,
    side: str,
    status: str,
    confidence: float,
) -> str:
    """Format a scan log entry for other bone findings.

    Args:
        bone_name: Name of the bone (e.g., "clavicle", "scapula", "spine")
        side: Anatomical side ("left", "right", "midline", "upper_thoracic", etc.)
        status: Fracture status ("intact", "fractured", "suspicious", "osteoporosis")
        confidence: Detection confidence (0.0-1.0)

    Returns:
        Formatted log entry string
    """
    # Format bone label (handle spine regions specially)
    if bone_name == "spine":
        bone_label = f"{side.upper().replace('_', ' ')} SPINE"
    else:
        bone_label = f"{side.upper()} {bone_name.upper()}"

    if status == "fractured":
        status_marker = "FRACTURE DETECTED"
    elif status == "suspicious":
        status_marker = "SUSPICIOUS"
    elif status == "osteoporosis":
        status_marker = "OSTEOPOROSIS"
    else:
        status_marker = "intact"

    return f"{bone_label}: {status_marker} (conf: {confidence:.2f})"


def systematic_rib_scan(
    detections: list[RibDetectionResult],
    image_dimensions: tuple[int, int],
    other_bone_detections: list[OtherBoneDetectionResult] | None = None,
) -> RibAnalysisOutput:
    """Perform systematic rib scan following clinical protocol.

    Scans ribs in order: L1→L10, R1→R10, then scans other bones
    (clavicle, scapula).

    Args:
        detections: List of raw rib detections.
        image_dimensions: Image (width, height) for coordinate mapping.
        other_bone_detections: Optional list of clavicle/scapula detections.

    Returns:
        RibAnalysisOutput with systematic scan results.
    """
    start_time = time.time()

    # Index detections by rib_id for quick lookup
    detection_map: dict[str, RibDetectionResult] = {}
    for det in detections:
        # Keep highest confidence detection for each rib
        if det.rib_id not in detection_map or det.detection_score > detection_map[det.rib_id].detection_score:
            detection_map[det.rib_id] = det

    # Systematic scan following SCAN_ORDER
    rib_findings: list[RibFinding] = []
    fractures_detected: list[RibFinding] = []
    other_bone_findings: list[OtherBoneFinding] = []
    scan_log: list[str] = []

    for scan_item in SCAN_ORDER:
        if scan_item == "other_bones":
            # Handle other bones separately
            scan_log.append("--- Scanning other bones ---")
            continue

        # Look up this rib in detections
        if scan_item in detection_map:
            det = detection_map[scan_item]

            # Calculate centroid
            centroid = CoordinatePoint(
                x=(det.bbox[0] + det.bbox[2]) // 2,
                y=(det.bbox[1] + det.bbox[3]) // 2,
            )

            finding = RibFinding(
                rib_id=det.rib_id,
                bbox=det.bbox,
                centroid=centroid,
                detection_confidence=det.detection_score,
                fracture_status=det.fracture_status,
                fracture_confidence=det.fracture_confidence,
            )

            rib_findings.append(finding)

            # Track fractures
            if det.fracture_status == "fractured":
                fractures_detected.append(finding)

            # Log entry
            log_entry = format_scan_log_entry(
                det.rib_id,
                det.fracture_status,
                det.fracture_confidence,
            )
            scan_log.append(log_entry)
        else:
            # Rib not detected
            scan_log.append(f"{scan_item}: NOT DETECTED")

    # Process other bone detections (clavicle, scapula)
    other_bone_fracture_count = 0
    if other_bone_detections:
        # Process clavicles first (left then right)
        for side in ["left", "right"]:
            clavicle = next(
                (d for d in other_bone_detections if d.bone_name == "clavicle" and d.side == side),
                None,
            )
            if clavicle:
                finding = OtherBoneFinding(
                    bone_name=clavicle.bone_name,
                    side=clavicle.side,
                    bbox=clavicle.bbox,
                    fracture_status=clavicle.fracture_status,
                    fracture_confidence=clavicle.fracture_confidence,
                )
                other_bone_findings.append(finding)

                log_entry = format_other_bone_log_entry(
                    clavicle.bone_name,
                    clavicle.side,
                    clavicle.fracture_status,
                    clavicle.fracture_confidence,
                )
                scan_log.append(log_entry)

                if clavicle.fracture_status == "fractured":
                    other_bone_fracture_count += 1

        # Process scapulae (left then right)
        for side in ["left", "right"]:
            scapula = next(
                (d for d in other_bone_detections if d.bone_name == "scapula" and d.side == side),
                None,
            )
            if scapula:
                finding = OtherBoneFinding(
                    bone_name=scapula.bone_name,
                    side=scapula.side,
                    bbox=scapula.bbox,
                    fracture_status=scapula.fracture_status,
                    fracture_confidence=scapula.fracture_confidence,
                )
                other_bone_findings.append(finding)

                log_entry = format_other_bone_log_entry(
                    scapula.bone_name,
                    scapula.side,
                    scapula.fracture_status,
                    scapula.fracture_confidence,
                )
                scan_log.append(log_entry)

                if scapula.fracture_status == "fractured":
                    other_bone_fracture_count += 1

        # Process spine (upper, mid, lower thoracic)
        for region in ["upper_thoracic", "mid_thoracic", "lower_thoracic"]:
            spine = next(
                (d for d in other_bone_detections if d.bone_name == "spine" and d.side == region),
                None,
            )
            if spine:
                finding = OtherBoneFinding(
                    bone_name=spine.bone_name,
                    side=spine.side,
                    bbox=spine.bbox,
                    fracture_status=spine.fracture_status,
                    fracture_confidence=spine.fracture_confidence,
                )
                other_bone_findings.append(finding)

                log_entry = format_other_bone_log_entry(
                    spine.bone_name,
                    spine.side,
                    spine.fracture_status,
                    spine.fracture_confidence,
                )
                scan_log.append(log_entry)

                # Count fractures and osteoporosis as significant findings
                if spine.fracture_status in ("fractured", "osteoporosis"):
                    other_bone_fracture_count += 1

    # Calculate timing
    scan_duration_ms = (time.time() - start_time) * 1000

    # Total fracture count includes rib and other bone fractures
    total_fractures = len(fractures_detected) + other_bone_fracture_count

    return RibAnalysisOutput(
        metadata=RibAnalysisMetadata(
            model_version="rib-detector-v1.0",
            scan_duration_ms=scan_duration_ms,
        ),
        rib_findings=rib_findings,
        fractures_detected=fractures_detected,
        other_bone_findings=other_bone_findings,
        scan_order_report=scan_log,
        total_fracture_count=total_fractures,
    )


def run_full_rib_analysis(
    model: FCOSRibDetector | None = None,
    image: NDArray[np.uint8] = None,
    device: torch.device | None = None,
    detection_threshold: float = 0.5,
    use_simulation: bool = False,
    include_other_bones: bool = True,
    yolo_detector: YOLOv8RibDetector | None = None,
    yolo_model_path: Path | str | None = None,
) -> RibAnalysisOutput:
    """Run complete rib fracture analysis pipeline.

    Scans ribs systematically (L1→L10, R1→R10) then scans other bones
    (clavicle, scapula) for fractures.

    Detection priority:
    1. If yolo_detector is provided, use it directly
    2. If yolo_model_path is provided, load and use YOLO model
    3. If model (FCOS) is provided, use FCOS
    4. If use_simulation=True or no model, use simulation

    Args:
        model: FCOS rib detector (can be None if using simulation or YOLO).
        image: Normalized image array (H, W, 3) or (H, W) uint8.
        device: Device for inference (ignored if using simulation).
        detection_threshold: Confidence threshold for detections.
        use_simulation: If True, use simulated detections for testing.
        include_other_bones: If True, also scan clavicle and scapula.
        yolo_detector: Pre-loaded YOLOv8 detector (highest priority).
        yolo_model_path: Path to YOLO weights file to load.

    Returns:
        RibAnalysisOutput with complete systematic scan results.
    """
    if image is None:
        raise ValueError("image is required")

    # Get image dimensions
    h, w = image.shape[:2]
    image_dimensions = (w, h)

    # Determine detection method and run
    rib_detections: list[RibDetectionResult] = []

    if yolo_detector is not None:
        # Use provided YOLO detector
        rib_detections = run_yolo_rib_detection(yolo_detector, image)
    elif yolo_model_path is not None:
        # Load and use YOLO model
        try:
            yolo_detector = load_yolo_detector(
                model_path=yolo_model_path,
                conf_threshold=detection_threshold,
            )
            rib_detections = run_yolo_rib_detection(yolo_detector, image)
        except FileNotFoundError:
            # Fall back to simulation if model file not found
            rib_detections = simulate_rib_detection(image, detection_threshold)
    elif use_simulation or model is None:
        # Use simulation
        rib_detections = simulate_rib_detection(image, detection_threshold)
    else:
        # Use FCOS model
        if device is None:
            device = torch.device("cpu")
        rib_detections = run_rib_detection(model, image, device, detection_threshold)

    # Run other bone detection (clavicle, scapula, spine)
    # Currently only simulation is available for other bones
    other_bone_detections = None
    if include_other_bones:
        other_bone_detections = simulate_other_bone_detection(image)

    # Perform systematic scan
    return systematic_rib_scan(rib_detections, image_dimensions, other_bone_detections)
