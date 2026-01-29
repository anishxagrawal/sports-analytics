# src/spatial/image_to_field.py

"""
Image-to-field projection module for sports analytics system.

Provides simple, approximate conversion from pixel coordinates to normalized
field-space coordinates. This is NOT camera calibration or perspective correction.

What this projection is:
- A consistent, resolution-independent normalization
- A bridge between image space and a canonical field representation
- Intentionally approximate and relative, not geometrically accurate

What this projection is NOT:
- Not homography or camera calibration
- Not aware of pitch dimensions or real-world scale
- Not perspective-corrected or distortion-aware
- Not dependent on field lines or other geometric cues

The output provides a stable coordinate system for spatial analytics while
accepting that the projection itself is a simplified approximation.
"""

from typing import Optional, Tuple


def pixel_to_field(
    pixel_pos: Tuple[float, float],
    frame_shape: Tuple[int, int]
) -> Optional[Tuple[float, float]]:
    """
    Convert pixel coordinate to normalized field-space coordinate.
    
    Performs simple normalization: left→right becomes 0→1 horizontally,
    bottom→top becomes 0→1 vertically (near→far in field terms).
    
    This is an approximate, stateless projection that provides consistency
    across different video resolutions but makes no claims about geometric
    accuracy or real-world scale.
    
    Args:
        pixel_pos: Pixel coordinate (x, y) where (0, 0) is top-left
        frame_shape: Frame dimensions (height, width)
    
    Returns:
        Normalized field position (fx, fy) in [0, 1] × [0, 1], or None on invalid input.
        - fx: horizontal position (0=left, 1=right)
        - fy: vertical position (0=near/bottom, 1=far/top)
    
    Example:
        >>> pixel_to_field((960, 540), (1080, 1920))
        (0.5, 0.5)
        >>> pixel_to_field((0, 1080), (1080, 1920))
        (0.0, 0.0)
    """
    try:
        px, py = pixel_pos
        height, width = frame_shape
        
        # Validate inputs
        if width <= 0 or height <= 0:
            return None
        
        if not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
            return None
        
        # Normalize horizontal: left (0) → right (width) becomes 0 → 1
        fx = px / width
        
        # Normalize vertical: invert so bottom (height) → top (0) becomes 0 → 1
        # This makes fy=0 the "near" side (bottom of image) and fy=1 the "far" side (top)
        fy = 1.0 - (py / height)
        
        # Clamp to valid range [0, 1]
        fx = max(0.0, min(1.0, fx))
        fy = max(0.0, min(1.0, fy))
        
        return (fx, fy)
    
    except (TypeError, ValueError, ZeroDivisionError):
        # Silent failure on invalid input
        return None


def field_to_pixel(
    field_pos: Tuple[float, float],
    frame_shape: Tuple[int, int]
) -> Optional[Tuple[float, float]]:
    """
    Convert normalized field-space coordinate back to pixel coordinate.
    
    Inverse of pixel_to_field(). Useful for visualization or debugging.
    
    Args:
        field_pos: Normalized field position (fx, fy) in [0, 1] × [0, 1]
        frame_shape: Frame dimensions (height, width)
    
    Returns:
        Pixel coordinate (x, y) or None on invalid input.
    
    Example:
        >>> field_to_pixel((0.5, 0.5), (1080, 1920))
        (960.0, 540.0)
    """
    try:
        fx, fy = field_pos
        height, width = frame_shape
        
        # Validate inputs
        if width <= 0 or height <= 0:
            return None
        
        if not isinstance(fx, (int, float)) or not isinstance(fy, (int, float)):
            return None
        
        # Clamp field coordinates to valid range
        fx = max(0.0, min(1.0, fx))
        fy = max(0.0, min(1.0, fy))
        
        # Denormalize horizontal
        px = fx * width
        
        # Denormalize vertical (invert back)
        py = (1.0 - fy) * height
        
        return (px, py)
    
    except (TypeError, ValueError, ZeroDivisionError):
        # Silent failure on invalid input
        return None