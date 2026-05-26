"""Rib detection inference pipeline.

Handles image preprocessing, model inference, and systematic rib scanning
following the clinical protocol: L1→L10, R1→R10, then other bones.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import transforms

from tsxr2.rib_detection.detector import FCOSRibDetector
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

    return results


def format_other_bone_log_entry(
    bone_name: str,
    side: str,
    status: str,
    confidence: float,
) -> str:
    """Format a scan log entry for other bone findings.

    Args:
        bone_name: Name of the bone (e.g., "clavicle", "scapula")
        side: Anatomical side ("left", "right", "midline")
        status: Fracture status ("intact", "fractured", "suspicious")
        confidence: Detection confidence (0.0-1.0)

    Returns:
        Formatted log entry string
    """
    bone_label = f"{side.upper()} {bone_name.upper()}"

    if status == "fractured":
        status_marker = "FRACTURE DETECTED"
    elif status == "suspicious":
        status_marker = "SUSPICIOUS"
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
    model: FCOSRibDetector | None,
    image: NDArray[np.uint8],
    device: torch.device | None = None,
    detection_threshold: float = 0.5,
    use_simulation: bool = False,
    include_other_bones: bool = True,
) -> RibAnalysisOutput:
    """Run complete rib fracture analysis pipeline.

    Scans ribs systematically (L1→L10, R1→R10) then scans other bones
    (clavicle, scapula) for fractures.

    Args:
        model: FCOS rib detector (can be None if using simulation).
        image: Normalized image array (H, W, 3) uint8.
        device: Device for inference (ignored if using simulation).
        detection_threshold: Confidence threshold for detections.
        use_simulation: If True, use simulated detections for testing.
        include_other_bones: If True, also scan clavicle and scapula.

    Returns:
        RibAnalysisOutput with complete systematic scan results.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    image_dimensions = (w, h)

    # Run rib detection
    if use_simulation or model is None:
        rib_detections = simulate_rib_detection(image, detection_threshold)
    else:
        if device is None:
            device = torch.device("cpu")
        rib_detections = run_rib_detection(model, image, device, detection_threshold)

    # Run other bone detection (clavicle, scapula)
    other_bone_detections = None
    if include_other_bones:
        if use_simulation or model is None:
            other_bone_detections = simulate_other_bone_detection(image)
        # TODO: Add real model inference for other bones when available

    # Perform systematic scan
    return systematic_rib_scan(rib_detections, image_dimensions, other_bone_detections)
