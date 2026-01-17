# src/analytics/teams/assigner.py

"""
Jersey color feature extraction for team assignment.

Provides stateless functions for extracting color features from
player jersey regions. Does not perform team assignment or clustering.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


# Minimum saturation threshold to exclude grayscale/background pixels
SATURATION_THRESHOLD = 30

# Minimum valid pixels required for reliable color extraction
MIN_VALID_PIXELS = 10


def extract_jersey_color(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    """
    Extract dominant color feature from player jersey region with confidence metrics.
    
    Crops the upper-body region from the bounding box and computes
    a robust color feature in LAB color space, filtering out background
    and low-saturation pixels.
    
    Args:
        frame: Video frame (BGR format, numpy array)
        bbox: Bounding box (x1, y1, x2, y2) in pixels
    
    Returns:
        Dictionary containing:
            - 'color': Optional[Tuple[float, float, float]] - (L, A, B) or None if invalid
            - 'valid_pixel_count': int - Number of pixels used for computation
            - 'confidence': float - Quality metric (0.0 to 1.0)
            - 'valid': bool - Whether extraction succeeded
    
    Notes:
        - Uses upper 40% of bbox as jersey region (geometry-blind assumption)
        - Filters low-saturation pixels to exclude background/shadows
        - Uses median for robustness against outliers
        - Returns valid=False if insufficient colorful pixels
        - Confidence is RELATIVE, not absolute - downstream must threshold
        - Variance normalization (500.0) is heuristic and data-dependent
    
    Limitations:
        - Fixed 40% crop fails on crouching/partial boxes/camera tilt
        - Confidence scale is comparative, not calibrated
        - Low confidence values (e.g., 0.1-0.3) still mark valid=True
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop upper body region (top 40% of bbox for jersey)
    # Assumption: standing upright player, standard camera angle
    height = y2 - y1
    jersey_y2 = y1 + int(height * 0.4)
    jersey_region = frame[y1:jersey_y2, x1:x2]
    
    # Handle edge case of empty crop
    if jersey_region.size == 0:
        return {
            'color': None,
            'valid_pixel_count': 0,
            'confidence': 0.0,
            'valid': False
        }
    
    # Convert to LAB for perceptually uniform color space
    lab_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
    
    # Convert to HSV to filter low-saturation pixels (grayscale/background)
    hsv_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
    saturation = hsv_region[:, :, 1]
    
    # Create mask: keep only pixels with sufficient saturation (colorful regions)
    saturation_mask = saturation > SATURATION_THRESHOLD
    
    # Apply mask to LAB channels
    valid_pixels = lab_region[saturation_mask]
    valid_count = len(valid_pixels)
    
    # Reject extraction if insufficient colorful pixels found
    if valid_count < MIN_VALID_PIXELS:
        return {
            'color': None,
            'valid_pixel_count': valid_count,
            'confidence': 0.0,
            'valid': False
        }
    
    # Use median for robust color estimation (resistant to outliers)
    median_color = np.median(valid_pixels, axis=0)
    
    # Compute color variance to assess reliability
    # High variance indicates mixed colors (multiple players, shadows, etc.)
    color_variance = np.var(valid_pixels, axis=0).mean()
    
    # Confidence based on pixel count and color consistency
    total_pixels = lab_region.shape[0] * lab_region.shape[1]
    
    # Pixel count factor: ratio of valid to total pixels
    pixel_factor = min(1.0, valid_count / max(1, total_pixels * 0.3))
    
    # Variance factor: lower variance = higher confidence
    # Heuristic normalization - 500.0 is data-dependent, not universal
    # This produces RELATIVE confidence for comparison, not calibrated probability
    variance_factor = max(0.0, 1.0 - (color_variance / 500.0))
    
    # Combined confidence: both factors must be high
    # NOTE: Values like 0.2-0.4 are still marked valid=True
    # Downstream logic must apply own thresholds (e.g., confidence > 0.5)
    confidence = pixel_factor * variance_factor
    
    return {
        'color': (float(median_color[0]), float(median_color[1]), float(median_color[2])),
        'valid_pixel_count': int(valid_count),
        'confidence': float(confidence),
        'valid': True
    }