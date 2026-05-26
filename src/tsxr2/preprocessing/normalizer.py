"""Image normalization utilities for model input preparation.

Converts DICOM pixel arrays to normalized images suitable for
vision model inference (TSXr DenseNet121/ViT).
"""

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def apply_window(
    pixel_array: NDArray[np.uint16],
    window_center: float = 2048.0,
    window_width: float = 4096.0,
) -> NDArray[np.uint8]:
    """Apply windowing (contrast adjustment) to pixel array.

    Windowing maps the pixel values to a display range based on
    window center and width, commonly used for chest X-ray visualization.

    Args:
        pixel_array: Raw pixel data from DICOM.
        window_center: Center of the window (default: 2048 for 12-bit data).
        window_width: Width of the window (default: 4096 for full range).

    Returns:
        Uint8 array with values scaled to 0-255.
    """
    # Calculate window bounds
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    # Apply window and scale to 0-255
    windowed = np.clip(pixel_array, lower, upper)
    scaled = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)

    return scaled


def normalize_image(
    pixel_array: NDArray[np.uint16],
    target_size: tuple[int, int] = (512, 512),
    window_center: float | None = None,
    window_width: float | None = None,
) -> NDArray[np.uint8]:
    """Normalize a DICOM pixel array for model input.

    Performs the following transformations:
    1. Apply windowing for optimal contrast
    2. Resize to target dimensions
    3. Convert to 3-channel RGB (grayscale repeated)

    Args:
        pixel_array: Raw pixel data from DICOM (typically 12-16 bit).
        target_size: Output dimensions (height, width). Default (512, 512).
        window_center: Window center for contrast. If None, uses sensible default.
        window_width: Window width for contrast. If None, uses sensible default.

    Returns:
        Normalized uint8 array of shape (height, width, 3).
    """
    # Determine window parameters
    if window_center is None:
        # Auto-detect based on data range
        window_center = float(pixel_array.max() + pixel_array.min()) / 2
    if window_width is None:
        window_width = float(pixel_array.max() - pixel_array.min())
        # Ensure minimum window width to avoid division by zero
        window_width = max(window_width, 1.0)

    # Apply windowing
    windowed = apply_window(pixel_array, window_center, window_width)

    # Resize if needed
    if windowed.shape[:2] != target_size:
        img = Image.fromarray(windowed, mode="L")
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        windowed = np.array(img)

    # Convert grayscale to 3-channel RGB
    rgb = np.stack([windowed, windowed, windowed], axis=-1)

    return rgb
