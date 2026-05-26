"""YOLOv8-based rib fracture detector.

Uses a pre-trained YOLOv8 model for rib detection and fracture classification.
Maps detections to anatomical rib IDs (L1-L10, R1-R10) based on position.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from tsxr2.rib_detection.rib_labels import VISIBLE_RIB_LABELS

# Default model path
DEFAULT_MODEL_PATH = Path("c:/Users/ks_P/Downloads/YOLO26best.pt")


class YOLOv8RibDetector:
    """YOLOv8-based detector for rib fractures.

    Loads a YOLOv8 model trained on chest X-rays and maps detections
    to anatomical rib identifiers based on their spatial position.

    Attributes:
        model: Loaded YOLOv8 model.
        class_names: Class names from the model (e.g., 'normal', 'fx').
        conf_threshold: Minimum confidence for detections.
        invert_image: Whether to invert image intensity.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.01,
        invert_image: bool = True,
        input_size: int = 640,
    ):
        """Initialize YOLOv8 rib detector.

        Args:
            model_path: Path to YOLOv8 weights (.pt file).
            conf_threshold: Minimum confidence threshold for detections.
            invert_image: Whether to invert image (bones dark -> bright).
            input_size: Model input size (default 640x640).
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.invert_image = invert_image
        self.input_size = input_size

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names

        # Determine class indices
        self.normal_class = None
        self.fracture_class = None
        for idx, name in self.class_names.items():
            if name.lower() in ('normal', 'intact'):
                self.normal_class = idx
            elif name.lower() in ('fx', 'fracture', 'fractured'):
                self.fracture_class = idx

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Preprocess image for YOLO inference.

        Args:
            image: Input image array (H, W) or (H, W, 3).

        Returns:
            Preprocessed RGB image array (H, W, 3).
        """
        # Normalize to 8-bit if needed
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Invert if configured (some models expect inverted X-rays)
        if self.invert_image:
            image = 255 - image

        # Convert to PIL for resizing
        if len(image.shape) == 2:
            img = Image.fromarray(image)
        else:
            img = Image.fromarray(image)

        # Resize to model input size
        img = img.resize((self.input_size, self.input_size))

        # Convert to RGB
        img = img.convert('RGB')

        return np.array(img)

    def detect(
        self,
        image: NDArray[np.uint8],
        original_size: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Run detection on an image.

        Args:
            image: Input image array (H, W) or (H, W, 3).
            original_size: Original image size (width, height) for scaling boxes.

        Returns:
            List of detection dicts with keys:
                - bbox: (x1, y1, x2, y2) in original image coordinates
                - class_name: 'normal' or 'fx'
                - confidence: detection confidence
                - center: (cx, cy) center point
        """
        # Store original size for scaling
        if original_size is None:
            if len(image.shape) == 2:
                original_size = (image.shape[1], image.shape[0])
            else:
                original_size = (image.shape[1], image.shape[0])

        orig_w, orig_h = original_size

        # Preprocess
        processed = self.preprocess(image)

        # Run inference
        results = self.model(processed, conf=self.conf_threshold, verbose=False)

        # Parse results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # Scale coordinates back to original image size
                scale_x = orig_w / self.input_size
                scale_y = orig_h / self.input_size

                x1 = int(xyxy[0] * scale_x)
                y1 = int(xyxy[1] * scale_y)
                x2 = int(xyxy[2] * scale_x)
                y2 = int(xyxy[3] * scale_y)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls,
                    'class_name': self.class_names[cls],
                    'confidence': conf,
                    'center': (cx, cy),
                })

        return detections

    def map_to_rib_id(
        self,
        center: tuple[int, int],
        image_size: tuple[int, int],
    ) -> str | None:
        """Map a detection center point to a rib ID based on position.

        Uses anatomical knowledge:
        - Left side of image (patient's right) = R1-R10
        - Right side of image (patient's left) = L1-L10
        - Vertical position determines rib number (1=top, 10=bottom)

        Args:
            center: Detection center (x, y) in image coordinates.
            image_size: Image dimensions (width, height).

        Returns:
            Rib ID string (e.g., 'L5', 'R3') or None if outside rib area.
        """
        cx, cy = center
        w, h = image_size

        # Normalize coordinates to 0-1
        norm_x = cx / w
        norm_y = cy / h

        # Exclude regions unlikely to be ribs
        # Top 8% (clavicles) and bottom 15% (below ribs)
        if norm_y < 0.08 or norm_y > 0.85:
            return None

        # Exclude center 15% (spine/mediastinum)
        if 0.42 < norm_x < 0.58:
            return None

        # Determine side
        # Left side of image = patient's right (R ribs)
        # Right side of image = patient's left (L ribs)
        if norm_x < 0.5:
            side = 'R'
        else:
            side = 'L'

        # Map vertical position to rib number (1-10)
        # Ribs are roughly evenly spaced between y=0.08 and y=0.85
        rib_zone_top = 0.08
        rib_zone_bottom = 0.85
        rib_zone_height = rib_zone_bottom - rib_zone_top

        relative_y = (norm_y - rib_zone_top) / rib_zone_height
        rib_num = int(relative_y * 10) + 1
        rib_num = max(1, min(10, rib_num))  # Clamp to 1-10

        return f"{side}{rib_num}"

    def detect_ribs(
        self,
        image: NDArray[np.uint8],
    ) -> list[dict]:
        """Detect ribs and map to anatomical IDs.

        Args:
            image: Input chest X-ray image.

        Returns:
            List of rib detections with rib_id, bbox, fracture status.
        """
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]

        # Run detection
        detections = self.detect(image, original_size=(w, h))

        # Map to rib IDs and determine fracture status
        rib_detections = []
        for det in detections:
            rib_id = self.map_to_rib_id(det['center'], (w, h))

            if rib_id is None:
                continue

            # Determine fracture status from class
            if det['class_id'] == self.fracture_class:
                fracture_status = 'fractured'
            elif det['class_id'] == self.normal_class:
                fracture_status = 'intact'
            else:
                fracture_status = 'suspicious'

            rib_detections.append({
                'rib_id': rib_id,
                'bbox': det['bbox'],
                'center': det['center'],
                'detection_confidence': det['confidence'],
                'fracture_status': fracture_status,
                'fracture_confidence': det['confidence'],
            })

        return rib_detections


def load_yolo_detector(
    model_path: Path | str | None = None,
    conf_threshold: float = 0.01,
) -> YOLOv8RibDetector:
    """Load YOLOv8 rib detector.

    Args:
        model_path: Path to model weights. Uses default if None.
        conf_threshold: Detection confidence threshold.

    Returns:
        Initialized YOLOv8RibDetector.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    return YOLOv8RibDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
    )
