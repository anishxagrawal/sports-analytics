# src/spatial/field_lines.py

"""
Field line detection module for sports analytics system.

Extracts weak geometric cues from football pitch markings (white lines) in a
single video frame. These cues are optional hints, not authoritative geometry.

The output provides soft suggestions for downstream spatial anchoring modules
but makes no claims about specific pitch features or camera calibration.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2


class LineCue:
    """
    Represents a weak geometric cue from a detected pitch line.
    
    A LineCue is a suggestion, not a fact. It indicates that a straight white
    line was detected in the image, along with its approximate orientation,
    position, and relative prominence.
    
    Attributes:
        orientation: Line angle in radians (-π/2 to π/2), where 0 is horizontal
        midpoint: Normalized (x, y) position in [0, 1] image coordinates
        length: Normalized length relative to image diagonal
        confidence: Relative confidence score in [0, 1] based on dominance
    """
    
    def __init__(
        self,
        orientation: float,
        midpoint: Tuple[float, float],
        length: float,
        confidence: float
    ):
        """
        Initialize a line cue.
        
        Args:
            orientation: Angle in radians (-π/2 to π/2)
            midpoint: Normalized (x, y) in [0, 1]
            length: Normalized length relative to diagonal
            confidence: Score in [0, 1]
        """
        self.orientation = orientation
        self.midpoint = midpoint
        self.length = length
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'orientation': float(self.orientation),
            'midpoint': self.midpoint,
            'length': float(self.length),
            'confidence': float(self.confidence)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<LineCue angle={np.degrees(self.orientation):.1f}° "
            f"pos={self.midpoint} len={self.length:.3f} "
            f"conf={self.confidence:.2f}>"
        )


def detect_field_lines(
    frame: np.ndarray,
    min_line_length_ratio: float = 0.05,
    max_line_gap: int = 10,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 50,
    top_n_lines: int = 20
) -> List[LineCue]:
    """
    Detect weak geometric cues from pitch lines in a single frame.
    
    Extracts straight white line candidates from the image using edge detection
    and line fitting. Returns normalized, confidence-weighted cues suitable for
    downstream spatial anchoring.
    
    This function is stateless and defensive:
    - Returns empty list on any preprocessing failure
    - Filters aggressively to avoid spurious detections
    - Normalizes all outputs to be resolution-independent
    - Makes no assumptions about pitch orientation or specific features
    
    Args:
        frame: Input frame (BGR, RGB, or grayscale)
        min_line_length_ratio: Minimum line length as fraction of image diagonal
        max_line_gap: Maximum gap in pixels between line segments
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        hough_threshold: Accumulator threshold for Hough line detection
        top_n_lines: Maximum number of strongest lines to return
    
    Returns:
        List of LineCue objects, empty if detection fails or no lines found.
        Lines are sorted by confidence (highest first).
    
    Example:
        >>> frame = cv2.imread("field.jpg")
        >>> cues = detect_field_lines(frame)
        >>> for cue in cues:
        ...     print(f"Line at {np.degrees(cue.orientation):.1f}°")
    """
    try:
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if gray.size == 0:
            return []
        
        h, w = gray.shape
        diagonal = np.sqrt(h**2 + w**2)
        min_line_length_pixels = int(diagonal * min_line_length_ratio)
        
        # Enhance white markings
        enhanced = _enhance_white_markings(gray)
        if enhanced is None:
            return []
        
        # Detect edges
        edges = cv2.Canny(enhanced, canny_low, canny_high, apertureSize=3)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length_pixels,
            maxLineGap=max_line_gap
        )
        
        if lines is None or len(lines) == 0:
            return []
        
        # Convert to LineCue objects
        cues = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cue = _line_to_cue(x1, y1, x2, y2, w, h, diagonal)
            if cue is not None:
                cues.append(cue)
        
        if len(cues) == 0:
            return []
        
        # Sort by confidence and take top N
        cues.sort(key=lambda c: c.confidence, reverse=True)
        return cues[:top_n_lines]
    
    except Exception:
        # Silent failure - return empty list
        return []


def _enhance_white_markings(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Preprocess frame to emphasize white pitch markings.
    
    Uses adaptive thresholding and morphological operations to isolate
    bright linear structures that likely correspond to pitch lines.
    
    Args:
        gray: Grayscale input frame
    
    Returns:
        Enhanced binary-like image or None on failure
    """
    try:
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Isolate bright regions (white lines)
        _, bright = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
        
        # Morphological closing to connect nearby line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    except Exception:
        return None


def _line_to_cue(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_width: int,
    img_height: int,
    diagonal: float
) -> Optional[LineCue]:
    """
    Convert pixel line coordinates to normalized LineCue.
    
    Args:
        x1, y1, x2, y2: Line endpoints in pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        diagonal: Image diagonal length in pixels
    
    Returns:
        LineCue object or None if line is degenerate
    """
    try:
        # Compute line properties
        dx = x2 - x1
        dy = y2 - y1
        length_pixels = np.sqrt(dx**2 + dy**2)
        
        if length_pixels < 1.0:
            return None
        
        # Normalized length
        length_normalized = length_pixels / diagonal
        
        # Midpoint in normalized coordinates
        mid_x = ((x1 + x2) / 2.0) / img_width
        mid_y = ((y1 + y2) / 2.0) / img_height
        midpoint = (mid_x, mid_y)
        
        # Orientation angle in radians (normalized to -π/2 to π/2)
        angle = np.arctan2(dy, dx)
        # Normalize to [-π/2, π/2] range (treat line as undirected)
        if angle > np.pi / 2:
            angle -= np.pi
        elif angle < -np.pi / 2:
            angle += np.pi
        
        # Confidence based on relative length and position
        # Longer lines and lines near image center get higher confidence
        length_score = min(1.0, length_normalized / 0.3)  # Cap at 30% of diagonal
        
        # Distance from image center (normalized)
        center_x, center_y = 0.5, 0.5
        dist_from_center = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
        centrality_score = max(0, 1.0 - dist_from_center * 2.0)  # Decay from center
        
        # Combined confidence (weighted toward length)
        confidence = 0.7 * length_score + 0.3 * centrality_score
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return LineCue(
            orientation=angle,
            midpoint=midpoint,
            length=length_normalized,
            confidence=confidence
        )
    
    except Exception:
        return None


def filter_parallel_lines(
    cues: List[LineCue],
    angle_tolerance: float = np.radians(5)
) -> List[LineCue]:
    """
    Filter out redundant near-parallel lines, keeping only the strongest.
    
    When multiple lines have similar orientations and positions, keeps only
    the one with highest confidence. Useful for reducing noise from duplicate
    detections of the same physical line.
    
    Args:
        cues: List of LineCue objects
        angle_tolerance: Angular tolerance in radians for considering lines parallel
    
    Returns:
        Filtered list of LineCue objects
    """
    if len(cues) <= 1:
        return cues
    
    try:
        # Sort by confidence (highest first)
        sorted_cues = sorted(cues, key=lambda c: c.confidence, reverse=True)
        
        filtered = []
        for cue in sorted_cues:
            # Check if this cue is too similar to any already accepted cue
            is_redundant = False
            for accepted in filtered:
                angle_diff = abs(cue.orientation - accepted.orientation)
                # Handle angle wrapping
                if angle_diff > np.pi / 2:
                    angle_diff = np.pi - angle_diff
                
                # Check if angles are similar
                if angle_diff < angle_tolerance:
                    # Check if positions are close
                    pos_dist = np.sqrt(
                        (cue.midpoint[0] - accepted.midpoint[0])**2 +
                        (cue.midpoint[1] - accepted.midpoint[1])**2
                    )
                    if pos_dist < 0.1:  # Within 10% of image size
                        is_redundant = True
                        break
            
            if not is_redundant:
                filtered.append(cue)
        
        return filtered
    
    except Exception:
        # On any error, return original list
        return cues